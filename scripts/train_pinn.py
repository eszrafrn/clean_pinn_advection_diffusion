import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

import sys
root = Path(__file__).resolve().parent.parent
src = root/"src"
sys.path.insert(0, str(src))

from advection_diffusion.initial_conditions import gaussian_pulse_np
from advection_diffusion.metrics import compute_mass
from advection_diffusion.pinn_losses import gaussian_quadrature, total_pinn_loss
from advection_diffusion.pinn_model import PINN, save_checkpoint
from advection_diffusion.pinn_sampling import prepare_training_data


def parse_layers(text):
    return [int(v) for v in text.split(",")]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["vanilla", "conservative"], default="vanilla")
    parser.add_argument("--bc", choices=["dirichlet", "periodic", "zero_flux"], default="periodic")
    parser.add_argument("--pe", type=float, default=1.0)
    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--D", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--x0", type=float, default=0.5)
    parser.add_argument("--layers", default="2,50,50,50,50,1")
    parser.add_argument("--n-r", type=int, default=20000)
    parser.add_argument("--n-ic", type=int, default=4000)
    parser.add_argument("--n-bc", type=int, default=3000)
    parser.add_argument("--epochs-adam", type=int, default=10000)
    parser.add_argument("--epochs-lbfgs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-ic", type=float, default=100.0)
    parser.add_argument("--lambda-bc", type=float, default=10.0)
    parser.add_argument("--lambda-pde", type=float, default=1.0)
    parser.add_argument("--lambda-mass", type=float, default=10.0)
    parser.add_argument("--n-quad", type=int, default=100)
    parser.add_argument("--n-time-mass", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="outputs/models")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v = args.pe * args.D / args.L
    layers = parse_layers(args.layers)

    data = prepare_training_data(
        n_r=args.n_r,
        n_ic=args.n_ic,
        n_bc=args.n_bc,
        L=args.L,
        T=args.T,
        bc_type=args.bc,
        device=device,
        seed=args.seed,
    )

    ic_params = {"amplitude": 1.0, "x0": args.x0, "sigma": args.sigma}
    params = {
        "L": args.L,
        "T": args.T,
        "v": v,
        "D": args.D,
        "Pe": args.pe,
        "bc_type": args.bc,
        "variant": args.variant,
        "ic_params": ic_params,
        "lambda_ic": args.lambda_ic,
        "lambda_bc": args.lambda_bc,
        "lambda_pde": args.lambda_pde,
        "lambda_mass": args.lambda_mass if args.variant == "conservative" else 0.0,
    }

    model = PINN(layers=layers, L=args.L, T=args.T).to(device)

    mass_data = None
    if args.variant == "conservative":
        x_quad, w_quad = gaussian_quadrature(args.n_quad, L=args.L, device=device)
        t_samples = torch.linspace(0.0, args.T, args.n_time_mass, device=device).reshape(-1, 1)

        c0_quad_np = gaussian_pulse_np(
            x_quad.detach().cpu().numpy().reshape(-1),
            amplitude=1.0,
            x0=args.x0,
            sigma=args.sigma,
        )

        c0_quad = torch.tensor(
            c0_quad_np,
            dtype=torch.float32,
            device=device
        ).reshape(-1, 1)

        M0 = torch.sum(c0_quad * w_quad).item()

        print(f"M0 target for mass penalty = {M0:.12e}")

        mass_data = {
            "x_quad": x_quad,
            "w_quad": w_quad,
            "t_samples": t_samples,
            "M0": M0,
        }

        params["M0"] = M0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    progress = trange(1, args.epochs_adam + 1, desc=f"ADAM {args.variant}/{args.bc}")
    for epoch in progress:
        optimizer.zero_grad()
        loss, parts = total_pinn_loss(model, data, params, mass_data=mass_data)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs_adam:
            row = {
                "stage": "adam",
                "iter": epoch,
                "total": float(loss.detach().cpu()),
                **{k: float(vv.detach().cpu()) for k, vv in parts.items()},
            }
            history.append(row)
            progress.set_postfix(total=f"{row['total']:.3e}", pde=f"{row['L_pde']:.3e}")

    if args.epochs_lbfgs > 0:
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=args.epochs_lbfgs,
            max_eval=int(args.epochs_lbfgs * 1.25),
            history_size=100,
            tolerance_grad=1e-8,
            tolerance_change=1e-10,
            line_search_fn="strong_wolfe",
        )
        state = {"iter": 0}

        def closure():
            optimizer_lbfgs.zero_grad()
            loss, parts = total_pinn_loss(model, data, params, mass_data=mass_data)
            loss.backward()
            if state["iter"] % 50 == 0:
                row = {
                    "stage": "lbfgs",
                    "iter": state["iter"],
                    "total": float(loss.detach().cpu()),
                    **{k: float(vv.detach().cpu()) for k, vv in parts.items()},
                }
                history.append(row)
                print(
                    f"LBFGS {state['iter']:04d}: total={row['total']:.3e}, "
                    f"pde={row['L_pde']:.3e}, mass={row['L_mass']:.3e}"
                )
            state["iter"] += 1
            return loss

        optimizer_lbfgs.step(closure)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"pinn_{args.variant}_{args.bc}_Pe{args.pe:g}_T{args.T:g}"
    model_path = outdir / f"{tag}.pt"
    save_checkpoint(model_path, model, params, history=history)

    log_path = outdir / f"{tag}_loss.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "iter", "total", "L_ic", "L_bc", "L_pde", "L_mass"])
        writer.writeheader()
        writer.writerows(history)

    print(f"Saved model: {model_path}")
    print(f"Saved loss log: {log_path}")


if __name__ == "__main__":
    main()


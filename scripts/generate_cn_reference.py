import argparse
import json
from pathlib import Path

import numpy as np

from advection_diffusion.cn_solver import CNConfig, CrankNicolson1D


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc", choices=["dirichlet", "periodic", "zero_flux"], default="periodic")
    parser.add_argument("--pe", type=float, default=1.0)
    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--D", type=float, default=0.01)
    parser.add_argument("--nx", type=int, default=1000)
    parser.add_argument("--nt", type=int, default=1000)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--x0", type=float, default=0.5)
    parser.add_argument("--outdir", default="outputs/reference")
    return parser.parse_args()


def main():
    args = parse_args()
    v = args.pe * args.D / args.L
    cfg = CNConfig(
        L=args.L,
        T=args.T,
        nx=args.nx,
        nt=args.nt,
        v=v,
        D=args.D,
        bc_type=args.bc,
        ic_x0=args.x0,
        ic_sigma=args.sigma,
    )
    solver = CrankNicolson1D(cfg)
    result = solver.solve(save_history=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"reference_{args.bc}_Pe{args.pe:g}_T{args.T:g}"
    npz_path = outdir / f"{tag}.npz"
    np.savez(
        npz_path,
        x=result["x"],
        t=solver.t,
        c=result["c"],
        mass=result["mass"],
        L=args.L,
        T=args.T,
        nx=args.nx,
        nt=args.nt,
        v=v,
        D=args.D,
        Pe=args.pe,
        bc_type=args.bc,
        grid_type=result["grid_type"],
        ic_x0=args.x0,
        ic_sigma=args.sigma,
    )

    meta = {
        "path": str(npz_path),
        "bc_type": args.bc,
        "Pe": args.pe,
        "v": v,
        "D": args.D,
        "mass_initial": float(result["mass"][0]),
        "mass_final": float(result["mass"][-1]),
        "relative_mass_drift": float(abs(result["mass"][-1] - result["mass"][0]) / abs(result["mass"][0])),
    }
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()


import argparse
from pathlib import Path

import numpy as np
import torch

from advection_diffusion.pinn_model import load_checkpoint
from advection_diffusion.plotting import animate_cn_vs_pinn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--outdir", default="outputs/animations")
    return parser.parse_args()


def predict_model(model, x, t, device):
    model.eval()
    x_tensor = torch.tensor(x.reshape(-1, 1), dtype=torch.float32, device=device)
    pred = []
    with torch.no_grad():
        for tv in t:
            t_tensor = torch.full_like(x_tensor, float(tv))
            c = model(torch.cat([x_tensor, t_tensor], dim=1)).detach().cpu().numpy().reshape(-1)
            pred.append(c)
    return np.asarray(pred)


def main():
    args = parse_args()
    ref = np.load(args.reference, allow_pickle=True)
    x = ref["x"]
    t = ref["t"]
    c_cn = ref["c"]
    c_pinn = None
    name = Path(args.reference).stem

    if args.model:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = load_checkpoint(args.model, map_location=device)
        model = model.to(device)
        c_pinn = predict_model(model, x, t, device)
        name = Path(args.model).stem + "__vs__" + name

    out_path = Path(args.outdir) / f"{name}.gif"
    animate_cn_vs_pinn(x, t, c_cn, c_pinn=c_pinn, save_path=out_path)
    print(f"Saved animation: {out_path}")


if __name__ == "__main__":
    main()


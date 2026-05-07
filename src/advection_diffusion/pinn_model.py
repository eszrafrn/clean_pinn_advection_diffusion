import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, layers=None, L=1.0, T=1.0):
        super().__init__()
        if layers is None:
            layers = [2, 50, 50, 50, 50, 50, 1]
        self.layers = list(layers)
        self.register_buffer("lower", torch.tensor([[0.0, 0.0]], dtype=torch.float32))
        self.register_buffer("upper", torch.tensor([[float(L), float(T)]], dtype=torch.float32))

        modules = []
        for i in range(len(self.layers) - 2):
            modules.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(self.layers[-2], self.layers[-1]))
        self.net = nn.Sequential(*modules)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, xt):
        scaled = 2.0 * (xt - self.lower) / (self.upper - self.lower) - 1.0
        return self.net(scaled)


def save_checkpoint(path, model, config, history=None):
    payload = {
        "state_dict": model.state_dict(),
        "layers": model.layers,
        "config": config,
        "history": history or {},
    }
    torch.save(payload, path)


def load_checkpoint(path, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    cfg = payload.get("config", {})
    model = PINN(
        layers=payload.get("layers", [2, 64, 64, 64, 64, 1]),
        L=cfg.get("L", 1.0),
        T=cfg.get("T", 1.0),
    )
    model.load_state_dict(payload["state_dict"])
    return model, payload


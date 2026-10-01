import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    """Gated MLP: down(silu(gate(x)) * up(x))."""

    def __init__(self, config: dict):
        super().__init__()
        hidden = config["hidden_size"]
        inter = config["intermediate_size"]
        bias = config.get("mlp_bias", False)
        self.gate_proj = nn.Linear(hidden, inter, bias=bias)
        self.up_proj = nn.Linear(hidden, inter, bias=bias)
        self.down_proj = nn.Linear(inter, hidden, bias=bias)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))

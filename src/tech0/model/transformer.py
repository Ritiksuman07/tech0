"""Tech0 decoder-only transformer (~50M params), pure PyTorch."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from tech0.model.block import DecoderBlock
from tech0.model.rmsnorm import RMSNorm


@dataclass
class Tech0Config:
    name: str = "tech0-50m"
    vocab_size: int = 32000
    hidden_size: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 64
    intermediate_size: int = 1376
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False
    initializer_range: float = 0.02
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    use_cache: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Tech0Config":
        if "model" in data and isinstance(data["model"], dict):
            data = data["model"]
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Tech0Config":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _checkpointed_layer(hidden_states: torch.Tensor, layer: nn.Module) -> torch.Tensor:
    """Module-level so the checkpoint recompute binds the correct layer.

    A closure over the loop variable would resolve to the last layer during the
    backward recomputation and silently produce wrong gradients.
    """
    return layer(hidden_states, None, None, False)[0]


class Tech0ForCausalLM(nn.Module):
    def __init__(self, config: Tech0Config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(
            [DecoderBlock(config.to_dict(), layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.post_init()

    def post_init(self) -> None:
        self.apply(self._init_weights)
        std = self.config.initializer_range
        for name, param in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=std / math.sqrt(2.0 * self.config.num_hidden_layers))

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    # ---- helpers -------------------------------------------------------
    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def num_parameters(self, only_trainable: bool = True) -> int:
        params = self.parameters()
        if only_trainable:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def enable_gradient_checkpointing(self, enable: bool = True) -> None:
        self.gradient_checkpointing = enable

    # ---- forward -------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ):
        bsz, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        new_caches = [] if use_cache else None
        for idx, layer in enumerate(self.layers):
            past = past_key_values[idx] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training and not use_cache and attention_mask is None:
                hidden_states = checkpoint(
                    _checkpointed_layer, hidden_states, layer, use_reentrant=False
                )
                cache = None
            else:
                hidden_states, cache = layer(
                    hidden_states, attention_mask=attention_mask, past_key_value=past, use_cache=use_cache
                )
            if use_cache:
                new_caches.append(cache)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            return (logits, loss, new_caches)
        return {"logits": logits, "loss": loss, "past_key_values": new_caches}

    # ---- generation ----------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        eos_token_id = self.config.eos_token_id if eos_token_id is None else eos_token_id
        past = None
        for _ in range(max_new_tokens):
            out = self.forward(
                input_ids if past is None else input_ids[:, -1:],
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = out["past_key_values"]
            logits = out["logits"][:, -1, :].float()
            if temperature and temperature != 1.0:
                logits = logits / temperature
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) if temperature and temperature > 0 else logits.argmax(-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return input_ids

    # ---- persistence ---------------------------------------------------
    def save_pretrained(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_dir / "model.pt")
        self.config.save(save_dir / "config.json")

    @classmethod
    def from_pretrained(cls, save_dir: str | Path, map_location="cpu") -> "Tech0ForCausalLM":
        save_dir = Path(save_dir)
        config = Tech0Config.load(save_dir / "config.json")
        model = cls(config)
        state = torch.load(save_dir / "model.pt", map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        return model

"""Single teacher: Llama-3.2-3B.

Used in two ways:
  * sequence-level KD (default): generate high-density rationale/response text.
  * logit-level KD (optional): expose top-k soft targets. Requires the student
    and teacher to share a vocabulary (see `vocab_aligned`), otherwise the loss
    is undefined -- `distill.py` guards this.

The teacher is auto-disabled when available VRAM is below `min_vram_gb`
(the local MX330 has only 2 GB, so it never loads there).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tech0.train.trainer import resolve_device, vram_gb


@dataclass
class TeacherConfig:
    model_name: str = "meta-llama/Llama-3.2-3B"
    load_in_4bit: bool = True
    min_vram_gb: float = 8.0
    device: str = "auto"


class HFTeacher:
    def __init__(self, config: TeacherConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self) -> None:
        available = vram_gb()
        if self.device == "cuda" and available and available < self.config.min_vram_gb:
            raise RuntimeError(
                f"Teacher disabled: {available:.1f}GB VRAM < min_vram_gb={self.config.min_vram_gb}. "
                "Load it on a 16GB Colab/Kaggle GPU instead."
            )
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = {"torch_dtype": torch.float16 if self.device == "cuda" else torch.float32}
        if self.config.load_in_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = {"": self.device}

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name, **kwargs)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def vocab_size(self) -> int:
        return int(self.model.config.vocab_size)

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        batch_size: int = 8,
    ) -> list[str]:
        outputs: list[str] = []
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=1024
            ).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            for i in range(len(batch)):
                prompt_len = int(enc["input_ids"][i].shape[0])
                text = self.tokenizer.decode(gen[i][prompt_len:], skip_special_tokens=True)
                outputs.append(text.strip())
        return outputs

    @torch.no_grad()
    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids.to(self.model.device)).logits.float()

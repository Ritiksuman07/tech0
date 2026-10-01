"""Tech0: a 50M parameter decoder-only language model built from scratch in PyTorch."""

__version__ = "0.1.0"

from tech0.model.transformer import Tech0Config, Tech0ForCausalLM

__all__ = ["Tech0Config", "Tech0ForCausalLM", "__version__"]

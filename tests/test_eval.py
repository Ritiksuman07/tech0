from tech0.eval.code_eval import extract_code
from tech0.eval.mc_harness import evaluate_mc, synthetic_mc
from tech0.model import Tech0Config, Tech0ForCausalLM


class DigitTokenizer:
    """Minimal tokenizer for harness tests: every token is an integer id."""

    bos_token_id = 0

    def encode(self, text: str) -> list[int]:
        ids = [int(t) for t in text.split() if t.isdigit()]
        return ids or [1]


def tiny_model(vocab: int = 64) -> Tech0ForCausalLM:
    cfg = Tech0Config.from_dict({"model": {
        "vocab_size": vocab, "num_hidden_layers": 2, "hidden_size": 64,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
        "intermediate_size": 176, "max_position_embeddings": 64}})
    return Tech0ForCausalLM(cfg)


def test_extract_code_fenced():
    text = "Here you go:\n```python\nprint(1)\n```\ndone"
    assert extract_code(text) == "print(1)"


def test_extract_code_plain():
    assert extract_code("print(2)") == "print(2)"


def test_mc_evaluator_returns_valid_accuracy():
    model = tiny_model()
    examples = [{"context": "1 2", "choices": ["3", "4"], "gold": 0}]
    acc = evaluate_mc(model, DigitTokenizer(), examples, device="cpu")
    assert 0.0 <= acc <= 1.0

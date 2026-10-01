import pytest
import torch

from tech0.model import Tech0Config, Tech0ForCausalLM

EXPECTED_PARAMS = 49_623_552


def build_model(**overrides):
    cfg = Tech0Config.from_dict(overrides)
    return cfg, Tech0ForCausalLM(cfg)


def test_parameter_budget():
    _, model = build_model()
    assert model.num_parameters() == EXPECTED_PARAMS, model.num_parameters()


def test_config_roundtrip(tmp_path):
    cfg, _ = build_model()
    cfg.save(tmp_path / "config.json")
    loaded = Tech0Config.load(tmp_path / "config.json")
    assert loaded.vocab_size == cfg.vocab_size
    assert loaded.hidden_size == cfg.hidden_size
    assert loaded.tie_word_embeddings is True


def test_tied_embeddings_share_storage():
    _, model = build_model()
    assert model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()


def test_gqa_head_counts():
    _, model = build_model()
    attn = model.layers[0].self_attn
    assert attn.num_heads == 8
    assert attn.num_kv_heads == 2
    assert attn.num_key_value_groups == 4
    assert attn.head_dim == 64


def test_forward_shapes():
    _, model = build_model()
    ids = torch.randint(0, 100, (2, 16))
    out = model(ids)
    assert out["logits"].shape == (2, 16, 32000)
    assert out["loss"] is None


def test_loss_is_finite():
    _, model = build_model()
    ids = torch.randint(0, 100, (2, 16))
    out = model(ids, labels=ids)
    assert out["loss"] is not None
    assert torch.isfinite(out["loss"])


def test_generate_runs():
    cfg, model = build_model(max_position_embeddings=64)
    ids = torch.randint(0, 100, (1, 4))
    out = model.generate(ids, max_new_tokens=8, temperature=0.0)
    assert out.shape[0] == 1
    assert out.shape[1] == 12

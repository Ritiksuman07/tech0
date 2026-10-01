import copy

import torch

from tech0.model import Tech0Config, Tech0ForCausalLM


def tiny_config():
    return Tech0Config.from_dict(
        {"num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 4,
         "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 176,
         "vocab_size": 128, "max_position_embeddings": 32}
    )


def test_save_load_roundtrip(tmp_path):
    model = Tech0ForCausalLM(tiny_config())
    model.save_pretrained(tmp_path / "ckpt")
    loaded = Tech0ForCausalLM.from_pretrained(tmp_path / "ckpt")
    ids = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        a = model(ids)["logits"]
        b = loaded(ids)["logits"]
    assert torch.allclose(a, b, atol=1e-5)


def test_gradient_checkpointing_matches_plain():
    torch.manual_seed(0)
    cfg = tiny_config()
    plain = Tech0ForCausalLM(cfg)
    ckpt = copy.deepcopy(plain)
    ckpt.enable_gradient_checkpointing(True)
    ids = torch.randint(0, 128, (2, 8))

    plain.train()
    ckpt.train()
    loss_plain = plain(ids, labels=ids)["loss"]
    loss_plain.backward()
    loss_ckpt = ckpt(ids, labels=ids)["loss"]
    loss_ckpt.backward()

    assert torch.allclose(loss_plain, loss_ckpt, atol=1e-5)
    for (n1, p1), (n2, p2) in zip(plain.named_parameters(), ckpt.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1.grad, p2.grad, atol=1e-4), n1

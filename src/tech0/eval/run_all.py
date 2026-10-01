"""Run the full evaluation matrix and write eval_results/results.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tech0.eval.code_eval import evaluate_code
from tech0.eval.mc_harness import evaluate_mc, load_mc_task
from tech0.eval.perplexity import evaluate_perplexity
from tech0.model import Tech0Config, Tech0ForCausalLM
from tech0.tokenizer.build_bpe import Tech0Tokenizer
from tech0.train.trainer import resolve_device
from tech0.utils.config import apply_dotted_overrides, load_config
from tech0.utils.seed import set_seed


def load_model(model_dir: str | Path, device: str) -> Tech0ForCausalLM:
    model_dir = Path(model_dir)
    if model_dir.is_dir() and (model_dir / "config.json").exists():
        model = Tech0ForCausalLM.from_pretrained(model_dir, map_location=device)
    else:
        state = torch.load(model_dir, map_location=device, weights_only=False)
        model = Tech0ForCausalLM(Tech0Config.from_dict({"model": state["config"]}))
        model.load_state_dict(state["model"])
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tech0 evaluation suite")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--tasks", nargs="*", default=None, help="Subset of MC task names")
    parser.add_argument("--skip-code", action="store_true")
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_dotted_overrides(cfg, args.override)
    set_seed(cfg.get("seed", 1234))
    device = resolve_device(cfg.get("device", "auto"))

    model_dir = args.model_dir or cfg["model_dir"]
    tokenizer_dir = args.tokenizer_dir or cfg.get("tokenizer_dir", "data/tokenizer")
    tokenizer = Tech0Tokenizer(tokenizer_dir)

    try:
        model = load_model(model_dir, device)
    except Exception as exc:  # noqa: BLE001
        print(f"[eval] could not load '{model_dir}' ({exc}); using a fresh random model for smoke.")
        state_path = Path(tokenizer_dir).parent / "meta.json"
        vocab = json.loads(state_path.read_text())["vocab_size"] if state_path.exists() else 1024
        model = Tech0ForCausalLM(Tech0Config.from_dict({"model": {"vocab_size": vocab,
                          "num_hidden_layers": 2, "hidden_size": 128, "num_attention_heads": 4,
                          "num_key_value_heads": 2, "head_dim": 32, "intermediate_size": 352,
                          "max_position_embeddings": 256}})).to(device).eval()

    results: dict = {}

    task_names = args.tasks or list(cfg.get("tasks", {}).keys())
    for name in task_names:
        spec = cfg.get("tasks", {}).get(name, {})
        limit = 10 if args.smoke else spec.get("limit")
        examples = load_mc_task(name, split=spec.get("split", "validation"), limit=limit, smoke=args.smoke)
        acc = evaluate_mc(model, tokenizer, examples, device=device, limit=limit)
        results[name] = {"metric": "acc", "value": acc, "n": len(examples)}
        print(f"{name:12s} acc  = {acc:.4f}  (n={len(examples)})")

    if not args.skip_ppl and "perplexity" in cfg:
        spec = cfg["perplexity"]
        ppl = evaluate_perplexity(
            model, tokenizer, device=device, split=spec.get("split", "test"),
            limit=20 if args.smoke else spec.get("limit"), smoke=args.smoke,
        )
        results["wikitext103"] = {"metric": "perplexity", "value": ppl}
        print(f"{'wikitext103':12s} ppl  = {ppl:.2f}")

    if not args.skip_code and "code" in cfg:
        for task in cfg["code"].keys():
            spec = cfg["code"][task]
            limit = 1 if args.smoke else spec.get("limit")
            score = evaluate_code(model, tokenizer, task, device=device, limit=limit,
                                  tests_dir=spec.get("tests_dir", "data/custom_code"), smoke=args.smoke,
                                  max_new_tokens=32 if args.smoke else 256)
            results[task] = {"metric": "pass@1", "value": score}
            print(f"{task:12s} pass@1 = {score:.4f}")

    out_dir = Path(cfg.get("run", {}).get("out_dir", "eval_results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[eval] wrote {out_dir / 'results.json'}")


if __name__ == "__main__":  # pragma: no cover
    main()

"""Code generation evaluation: HumanEval, MBPP and a pluggable custom suite.

pass@1 is measured by executing generated code against the task's unit tests in
an isolated subprocess with a timeout.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    match = _FENCE.search(text)
    return (match.group(1) if match else text).strip()


def _run_code(source: str, timeout: int = 10) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "solution.py"
        script.write_text(source, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            return False


@torch.no_grad()
def _generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 256) -> str:
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.2, top_k=50)
    generated = tokenizer.decode(out[0].tolist()[ids.shape[1] :])
    return extract_code(generated)


def _humaneval_cases(limit: int | None, smoke: bool):
    if smoke:
        return [{"prompt": "def add(a, b):\n", "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
                 "entry_point": "add"}]
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    cases = []
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        cases.append({"prompt": ex["prompt"], "test": ex["test"], "entry_point": ex["entry_point"]})
    return cases


def _mbpp_cases(limit: int | None, smoke: bool):
    if smoke:
        return [{"prompt": "Write a function add(a,b) that returns their sum.",
                 "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]}]
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    cases = []
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        tests = ex.get("test_list") or ([ex["test"]] if ex.get("test") else [])
        cases.append({"prompt": ex["prompt"], "test_list": tests})
    return cases


def _custom_cases(tests_dir: str | Path, limit: int | None, smoke: bool):
    if smoke:
        return [{"prompt": "Write a function double(x) returning 2*x.",
                 "test_list": ["assert double(2) == 4"]}]
    tests_dir = Path(tests_dir)
    if not tests_dir.exists():
        return []
    cases = []
    for path in sorted(tests_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cases.append(json.loads(line))
    return cases[: limit or len(cases)]


def evaluate_code(
    model,
    tokenizer,
    task: str,
    device: str = "cpu",
    limit: int | None = None,
    tests_dir: str | Path = "data/custom_code",
    smoke: bool = False,
    max_new_tokens: int = 256,
) -> float:
    model.eval()
    if task == "humaneval":
        cases = _humaneval_cases(limit, smoke)
    elif task == "mbpp":
        cases = _mbpp_cases(limit, smoke)
    elif task == "custom":
        cases = _custom_cases(tests_dir, limit, smoke)
    else:
        raise KeyError(f"Unknown code task '{task}'")

    passed = 0
    for case in cases:
        body = _generate(model, tokenizer, case["prompt"], device, max_new_tokens)
        if task == "humaneval":
            source = f"{case['prompt']}{body}\n\n{case['test']}\ncheck({case['entry_point']})\n"
        elif task == "mbpp":
            tests = "\n".join(case["test_list"])
            source = f"{body}\n\n{tests}\n"
        else:
            tests = "\n".join(case.get("test_list", case.get("tests", [])))
            source = f"{case['prompt']}\n{body}\n\n{tests}\n"
        passed += int(_run_code(source))
    return passed / max(1, len(cases))

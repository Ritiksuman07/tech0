# Tech0 — a 50M-parameter LLM from scratch

Tech0 is a decoder-only transformer (~**49.62M** parameters) implemented in **pure PyTorch**, with a full pipeline for:

1. a custom **32k byte-level BPE** tokenizer,
2. phased **pretraining** with a 25/75 → 70/30 curriculum,
3. **knowledge distillation** from a single teacher (**Llama-3.2-3B**, sequence-level),
4. multi-task **SFT** (ChatML, prompt-masked loss),
5. an **8-benchmark evaluation suite**.

Benchmarks: **HellaSwag, ARC-Easy, PIQA, Winogrande, WikiText-103 (ppl), HumanEval, MBPP, Custom Code**.

---

## Architecture & parameter budget

| Component | Config | Params |
|---|---|---|
| Tied embeddings | 32000 × 512 | 16.384M |
| Per layer | GQA (8 Q / 2 KV, head_dim 64) + SwiGLU (512→1376→512) + 2×RMSNorm | 2,769,920 |
| Backbone | 12 layers | 33.239M |
| Final RMSNorm | 512 | 0.0005M |
| **Total** | | **49,623,552** |

Primitives: **RMSNorm**, **RoPE** (θ=10000, ctx 2048), **Grouped-Query Attention**, **SwiGLU**, tied input/output embeddings, no biases. Gradient checkpointing is supported (and unit-tested for gradient equivalence).

---

## Project layout

```
tech0/
├─ configs/           model_50m, pretrain, distill, sft, eval (YAML)
├─ src/tech0/
│  ├─ model/          rmsnorm, rope, gqa, swiglu, block, transformer
│  ├─ tokenizer/      build_bpe (32k byte-level BPE)
│  ├─ data/           sources, packing, dataset, curriculum, prepare
│  ├─ train/          optim, trainer, train_pretrain, distill, sft
│  ├─ eval/           mc_harness, perplexity, code_eval, run_all
│  ├─ teacher/        hf_teacher (Llama-3.2-3B, device-gated)
│  └─ utils/          config, ckpt, logging, seed
├─ scripts/           setup_env.ps1, smoke_test.ps1, colab/kaggle bootstrap.sh
└─ tests/             model, shapes, eval
```

---

## Quickstart (local dev / smoke, CPU or small GPU)

```powershell
# 1. environment (creates D:\tech0\.venv with a CUDA torch build)
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1

# 2. full offline smoke test: unit tests -> data -> pretrain -> eval
powershell -ExecutionPolicy Bypass -File scripts\smoke_test.ps1
```

Or run stages manually (any Python with torch + deps):

```powershell
$env:PYTHONPATH="D:\tech0\src"
python -m pytest tests -q
python -m tech0.data.prepare     --config configs/pretrain.yaml --smoke
python -m tech0.train.train_pretrain --config configs/pretrain.yaml --smoke
python -m tech0.train.distill    --config configs/distill.yaml  --smoke
python -m tech0.train.sft        --config configs/sft.yaml      --smoke
python -m tech0.eval.run_all     --config configs/eval.yaml     --smoke
```

---

## Real runs on Colab / Kaggle (16 GB T4/P100)

```bash
# Colab
git clone <your-repo> /content/tech0 && cd /content/tech0
bash scripts/colab_bootstrap.sh

# Kaggle
cd /kaggle/working/tech0 && bash scripts/kaggle_bootstrap.sh
```

Then:

```bash
# 1. data: tokenizer + packed token bins (cap with --max-docs / --max-tokens)
python -m tech0.data.prepare --config configs/pretrain.yaml \
    --sources fineweb_edu cosmopedia wikipedia code --max-docs 200000

# 2. pretrain (resumable; checkpoint to mounted Drive to survive 12h session caps)
python -m tech0.train.train_pretrain --config configs/pretrain.yaml
python -m tech0.train.train_pretrain --config configs/pretrain.yaml --resume

# 3. distillation: teacher generates synthetic rationales, student anneals
python -m tech0.train.distill --config configs/distill.yaml

# 4. SFT
python -m tech0.train.sft --config configs/sft.yaml

# 5. evaluation (8 benchmarks)
python -m tech0.eval.run_all --config configs/eval.yaml
```

> **Persistence:** set `run.out_dir` to `/content/drive/MyDrive/tech0/...` (Colab) or `/kaggle/working/...` (Kaggle) so checkpoints and token bins are not lost at session end.

---

## Data curriculum

| Phase | Volume (target) | Target task | General | Sources |
|---|---|---|---|---|
| 1 Foundation | 100B–200B tokens | 25% | 75% | FineWeb-Edu, Cosmopedia v2, Wikipedia |
| 2 Annealing | 20B–40B tokens | 70% | 30% | target corpus + synthetic reasoning traces |
| 3 SFT | 5M–10M tokens | 65% | 35% | instruction pairs with CoT |

Configs ship the full budgets; the default runnable budget is small. At least 25–30% generalist data is retained in Phases 2–3 (sample replay) to avoid catastrophic forgetting.

---

## Distillation (key design note)

- **Default: sequence-level KD.** The teacher generates dense, error-free rationale/response text which the student is annealed on. This is vocabulary-agnostic and fits a 16 GB GPU (Llama-3.2-3B in 4-bit ≈ 2.5 GB).
- **Optional: logit-level KD.** `L = (1−α)·CE + α·τ²·KL(student‖teacher)` with τ∈[1.5,2.5], top-k=100. This **requires a shared vocabulary**; because our 32k BPE differs from Llama's 128k vocab, `distill.py` refuses logit mode unless a vocab-aligned teacher/projection is supplied.
- The teacher is **auto-disabled** below `teacher.min_vram_gb` (the local 2 GB MX330 never loads it), so local dev always uses the offline `--smoke` path.

---

## Evaluation

| Benchmark | Metric | Harness |
|---|---|---|
| HellaSwag / ARC-Easy / PIQA / Winogrande | accuracy (length-normalized loglikelihood) | `eval/mc_harness.py` |
| WikiText-103 | perplexity (coherence health check) | `eval/perplexity.py` |
| HumanEval / MBPP / Custom | pass@1 (sandboxed subprocess, timeout) | `eval/code_eval.py` |

Custom suite: drop `*.jsonl` records `{"prompt": ..., "test_list": [...]}` into `data/custom_code/`.

---

## Precision / hardware notes

- **T4 / P100 (16 GB):** fp16 + `GradScaler` (Pascal/Turing have no bf16). This is the auto default.
- **A100 / L4 / Ampere+:** bf16 is auto-selected.
- **Local MX330 (2 GB, Pascal):** dev + smoke only; real training goes to Colab/Kaggle.

---

## Testing

```powershell
$env:PYTHONPATH="D:\tech0\src"; python -m pytest tests -q
```

`tests/test_model.py` asserts the exact 49,623,552-parameter budget; `tests/test_shapes.py` verifies save/load and that gradient checkpointing yields identical gradients to the plain path.

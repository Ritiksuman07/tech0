# Tech0 end-to-end smoke test (works on CPU or the 2GB local GPU).
# Trains a tiny random-data run for a handful of steps to verify the pipeline.
param(
    [string]$Python = "D:\tech0\.venv\Scripts\python.exe",
    [string]$WorkDir = "D:\tech0"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "$WorkDir\src"

Write-Host "[tech0] 1/4 model unit tests"
& $Python -m pytest "$WorkDir\tests" -q

Write-Host "[tech0] 2/4 build smoke corpus + tokenizer"
& $Python -m tech0.data.prepare --config "$WorkDir\configs\pretrain.yaml" --smoke

Write-Host "[tech0] 3/4 pretrain smoke run (CPU/tiny)"
& $Python -m tech0.train.train_pretrain --config "$WorkDir\configs\pretrain.yaml" --smoke

Write-Host "[tech0] 4/4 eval smoke (ppl + one MC task, limited)"
& $Python -m tech0.eval.run_all --config "$WorkDir\configs\eval.yaml" --smoke

Write-Host "[tech0] Smoke test complete."

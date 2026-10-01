# Tech0 local dev environment (Windows).
# Creates an isolated venv at D:\tech0\.venv and installs a CUDA build of PyTorch.
# The local MX330 (2GB) is for smoke tests only; real runs go to Colab/Kaggle.
param(
    [string]$Venv = "D:\tech0\.venv",
    [string]$Cuda = "cu121"
)

$ErrorActionPreference = "Stop"
Write-Host "[tech0] Creating venv at $Venv"
python -m venv $Venv
$py = Join-Path $Venv "Scripts\python.exe"

Write-Host "[tech0] Upgrading pip"
& $py -m pip install --upgrade pip wheel setuptools

Write-Host "[tech0] Installing torch from $Cuda"
& $py -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$Cuda"

Write-Host "[tech0] Installing project deps"
& "D:\tech0\.venv\Scripts\pip.exe" install -e "D:\tech0[dev]"

Write-Host "[tech0] Done. Activate with:  & '$Venv\Scripts\Activate.ps1'"
& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

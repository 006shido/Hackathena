param([switch]$AcceptResearchTerms)
$ErrorActionPreference = 'Stop'
if (-not $AcceptResearchTerms) { throw 'Read MODEL_NOTICES.md, then run with -AcceptResearchTerms for noncommercial research use.' }
$setupRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
Push-Location $setupRoot
try {
    & py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 with the Windows Python launcher.' }
    $setupPython = Join-Path $setupRoot '.venv/Scripts/python.exe'
    & $setupPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'pip setup failed.' }
    & $setupPython -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
    & $setupPython -m pip install -r ml/requirements-runtime.txt -r ml/requirements-webrtc-lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
    & $setupPython ml/scripts/download_models.py --accept-research-terms
    if ($LASTEXITCODE -ne 0) { throw 'Model download/verification failed.' }
    foreach ($setupDirectory in @('.', 'client', 'server')) {
        & npm ci --prefix $setupDirectory
        if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed: $setupDirectory" }
    }
    & npm run build
    if ($LASTEXITCODE -ne 0) { throw 'Application build failed.' }
    & $setupPython -c 'import torch; assert torch.cuda.is_available(), "An NVIDIA GPU and compatible CUDA driver are required"'
    if ($LASTEXITCODE -ne 0) { throw 'CUDA validation failed.' }
    Write-Host 'Setup complete. Run ./ml/scripts/Start-ResearchPreview.ps1 -EnableMediaDetection -EnableWebRTCVideo'
} finally { Pop-Location }

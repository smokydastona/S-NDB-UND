Param(
  [string]$OutDir = "dist",
  [string]$WorkDir = "build",
  [switch]$Clean
)

$ErrorActionPreference = "Stop"

if ($Clean) {
  if (Test-Path $OutDir) { Remove-Item -Recurse -Force $OutDir }
  if (Test-Path $WorkDir) { Remove-Item -Recurse -Force $WorkDir }
  if (Test-Path "soundgen.spec") { Remove-Item -Force "soundgen.spec" }
}

# Install build-time tooling only (kept out of requirements.txt)
python -m pip install --upgrade pip | Out-Null
python -m pip install pyinstaller | Out-Null

# Ensure runtime deps are present (uses requirements.txt)
python -m pip install -r requirements.txt | Out-Null

# Build two executables (folder-based /onedir for reliability)
# Note: AI engines (torch/diffusers/transformers) make these builds large.
pyinstaller --noconfirm --clean --onedir --name soundgen-generate \
  --collect-all soundgen \
  --collect-all numpy \
  --collect-all scipy \
  --collect-all soundfile \
  --collect-all torch \
  --collect-all diffusers \
  --collect-all transformers \
  --collect-all accelerate \
  --collect-all safetensors \
  -m soundgen.generate \
  --distpath $OutDir --workpath $WorkDir

pyinstaller --noconfirm --clean --onedir --name soundgen-web \
  --collect-all soundgen \
  --collect-all gradio \
  --collect-all numpy \
  --collect-all scipy \
  --collect-all soundfile \
  --collect-all torch \
  --collect-all diffusers \
  --collect-all transformers \
  --collect-all accelerate \
  --collect-all safetensors \
  -m soundgen.web \
  --distpath $OutDir --workpath $WorkDir

Write-Host "Built executables into $OutDir\\soundgen-generate and $OutDir\\soundgen-web"
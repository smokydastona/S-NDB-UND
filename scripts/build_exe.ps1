Param(
  [string]$OutDir = "dist",
  [string]$WorkDir = "build",
  [string]$Version = "",
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
$genName = "soundgen-generate"
$webName = "soundgen-web"
if ($Version -and $Version.Trim().Length -gt 0) {
  $ver = $Version.Trim()
  $genName = "$genName-$ver"
  $webName = "$webName-$ver"
}

pyinstaller --noconfirm --clean --onedir --name $genName \
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

pyinstaller --noconfirm --clean --onedir --name $webName \
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

Write-Host "Built executables into $OutDir\\$genName and $OutDir\\$webName"
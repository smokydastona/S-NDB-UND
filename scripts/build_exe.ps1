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
  Get-ChildItem -Path . -Filter "*.spec" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
}

# Install build-time tooling only (kept out of requirements.txt)
python -m pip install --upgrade pip | Out-Null
python -m pip install pyinstaller | Out-Null

# Ensure runtime deps are present (uses requirements.txt)
python -m pip install -r requirements.txt | Out-Null

# Build the executable (folder-based /onedir for reliability)
# Note: AI engines (torch/diffusers/transformers) make these builds large.
$baseAppName = "S-NDB-UND"
$appName = $baseAppName
if ($Version -and $Version.Trim().Length -gt 0) {
  $ver = $Version.Trim()
  $appName = "$baseAppName-$ver"
}

$commonArgs = @(
  "--noconfirm",
  "--clean",
  "--onedir"
)

$commonCollect = @(
  "--collect-all", "soundgen",
  "--collect-all", "numpy",
  "--collect-all", "scipy",
  "--collect-all", "soundfile",
  "--collect-all", "gradio",
  "--collect-all", "webview",
  "--collect-all", "torch",
  "--collect-all", "diffusers",
  "--collect-all", "transformers",
  "--collect-all", "accelerate",
  "--collect-all", "safetensors"
)

function Invoke-PyInstaller {
  param(
    [Parameter(Mandatory=$true)][string[]]$Args
  )

  python -m PyInstaller @Args
  if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
  }
}

$appArgs = @()
$appArgs += $commonArgs
$appArgs += @("--name", $appName)
$appArgs += $commonCollect
$appArgs += @(
  "--distpath", $OutDir,
  "--workpath", $WorkDir,
  "src/soundgen/app.py"
)

Invoke-PyInstaller -Args $appArgs

Write-Host "Built executable into $OutDir\\$appName"
Param(
  [string]$VenvDir = ".venv",
  [switch]$WithRfxgen,
  [switch]$CheckStableAudio,
  [string]$HfToken = ""
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
  Write-Host "==> $Message"
}

Write-Step "Soundgen setup (Windows)"

if (-not (Test-Path $VenvDir)) {
  Write-Step "Creating virtual environment at $VenvDir"
  python -m venv $VenvDir
}

$py = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $py)) {
  throw "Python interpreter not found at $py. Did venv creation fail?"
}

Write-Step "Upgrading pip"
& $py -m pip install --upgrade pip

Write-Step "Installing dependencies (requirements.txt)"
& $py -m pip install -r requirements.txt

if ($WithRfxgen) {
  $rfx = "tools/rfxgen/rfxgen.exe"
  if (-not (Test-Path $rfx)) {
    Write-Step "Downloading rfxgen.exe via scripts/get_rfxgen.ps1"
    & pwsh -NoProfile -ExecutionPolicy Bypass -File "./scripts/get_rfxgen.ps1"
  } else {
    Write-Step "rfxgen already present at $rfx"
  }
}

Write-Step "Running environment checks (soundgen.doctor)"
if ($CheckStableAudio) {
  if ($HfToken -and $HfToken.Trim().Length -gt 0) {
    & $py -m soundgen.doctor --check-stable-audio --hf-token $HfToken
  } else {
    & $py -m soundgen.doctor --check-stable-audio
  }
} else {
  & $py -m soundgen.doctor
}

Write-Host ""
Write-Step "Done"
Write-Host "Next:"
Write-Host "  1) Activate venv: .\\$VenvDir\\Scripts\\Activate.ps1"
Write-Host "  2) Run Web UI:    python -m soundgen.web"
Write-Host "  3) Run CLI:       python -m soundgen.generate --prompt \"laser zap\" --out outputs\\laser.wav"

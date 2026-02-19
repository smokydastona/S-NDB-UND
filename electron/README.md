# SÖNDBÖUND — Electron shell

This folder contains an Electron wrapper that launches the existing Python/Gradio UI backend and embeds it in a native desktop window.

## Why

- Native window frame (no browser chrome)
- App menu hooks (reload / devtools / quit)
- A straightforward place to add tray + auto-update later

## Dev run (repo checkout)

Prereqs:
- Node.js (LTS recommended)
- A working Python environment for this repo (`.venv` recommended)

From repo root:

```powershell
cd electron
npm install
npm run dev
```

By default, it uses the repo venv at `../.venv/Scripts/python.exe` on Windows.

### Override Python

```powershell
$env:SOUNDGEN_PYTHON = "C:\Path\To\python.exe"
npm run dev
```

## How it works

- Electron starts: `python -m soundgen.app serve --host 127.0.0.1 --port 0`
- The backend prints a line like: `SOUNDGEN_URL=http://127.0.0.1:7860`
- Electron reads that URL from stdout and loads it in a `BrowserWindow`.

Backend logs are written under the Electron user data folder (see `electron-backend.log`).

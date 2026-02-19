# Contributing to SÖNDBÖUND

Thanks for contributing — this repo is intentionally built as a platform (engines + presets + FX + exports).

## Dev setup

### Windows (recommended)

```powershell
./scripts/setup.ps1
```

### Minimal

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Project layout (quick map)

- `src/soundgen/generate.py` — CLI entry point for single sound generation.
- `src/soundgen/web_control_panel.py` — Gradio Control Panel UI (tables + buttons).
- `src/soundgen/web.py` — legacy Gradio UI.
- `src/soundgen/engine_registry.py` — engine registry + plugin loading + `generate_wav()`.
- `src/soundgen/postprocess.py`, `src/soundgen/qa.py` — post chain + QA scoring.
- `src/soundgen/minecraft.py` — `.ogg` conversion + `sounds.json` + optional subtitles.
- `src/soundgen/io_utils.py` — WAV I/O and ffmpeg discovery.
- `configs/` — example preset libraries + FX chain examples.
- `library/` — local catalogs/overrides (gitignored by design).

## Code style / conventions

- Keep audio arrays **mono 1D float32** in **[-1, 1]** between stages.
- Prefer small, focused changes; avoid drive-by formatting.
- Keep heavy deps **lazily imported** (so `--help` and lightweight commands stay fast).
- Preserve existing UX and CLI flags unless the change explicitly targets them.

## Validation (required before PR)

Run these from the repo root:

```powershell
python -m compileall -q src/soundgen
python -m soundgen.doctor
```

There isn’t a full unit-test suite yet. If your change affects core workflows, also run a quick smoke check:

```powershell
python -m soundgen.generate --engine rfxgen --prompt "coin pickup" --post --out outputs\test.wav
python -m soundgen.generate --engine rfxgen --minecraft --namespace mymod --event ui.coin --prompt "coin pickup" --post
```

## How to add a new engine

There are two paths:

### A) Built-in engine (in-repo)

1. Implement the engine backend (usually as a new module under `src/soundgen/`).
2. Wire it into `src/soundgen/engine_registry.py`:
   - Add the engine name to `BUILTIN_ENGINES`.
   - Add a branch in the engine dispatch (search for the existing engine blocks).
3. Ensure it returns **mono float32** audio and a sample rate.
4. Update docs:
   - README engine list (if user-facing)
   - Any relevant spec docs under `docs/`

### B) Plugin engine (recommended for experiments)

SÖNDBÖUND supports best-effort engine plugin discovery.

- See `docs/plugins.md` for the plugin discovery mechanism and expected callable shape.
- Your plugin registers with `register_engine(engine_name=..., engine_fn=...)`.

## How to add a preset

Presets are JSON libraries (v2 supports inheritance + template variables).

- Example libraries:
  - `configs/sfx_presets_v2.example.json`
  - `configs/sfx_presets_v1.example.json`
- Loader + schema logic lives in `src/soundgen/sfx_presets.py`.

Guidelines:

- Prefer adding to the example v2 library first.
- Keep preset names stable; they become part of users’ workflows/manifests.
- If you add new fields, update `docs/sfx_presets_v2_schema.md`.

## How to add a polish profile

Polish profiles are curated “safe defaults” that apply conservatively.

- Implemented in `src/soundgen/polish_profiles.py` (`POLISH_PROFILES`).

Guidelines:

- Keep profiles conservative: avoid extreme loudness and heavy reverb by default.
- Prefer small, explainable parameter sets.
- If the profile is domain-specific (e.g. creature/foley/ambience), name it accordingly.

## How to add or edit an FX chain

FX chains are JSON-driven.

- Docs: `docs/fx_chains_v2.md`
- Example config: `configs/fx_chain_v2.example.json`

When changing FX behavior, check:

- No NaNs/inf are introduced
- Output remains mono float32 in [-1, 1]
- Trimming never returns empty audio

## Docs contributions

Docs are a feature. If you add a capability, please add:

- At least one runnable CLI example in the README (or a relevant doc in `docs/`)
- Any schema updates (manifest/presets) alongside code changes

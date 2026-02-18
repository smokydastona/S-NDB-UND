# S‑NDB‑UND “one‑stop shop” definition + buildable scope

This doc defines what “one‑stop shop” means for S‑NDB‑UND in a way that’s **tight, testable, and shippable**.

## What “one‑stop shop” means (in this project)

A modder can do the full loop **inside S‑NDB‑UND**:

1) Generate sounds (AI + procedural)
2) Shape/polish them (FX chains + post)
3) Do quick, destructive edits (trim/fade/normalize/loop audition)
4) Organize + batch (presets, variations, metadata)
5) Export directly to a game/mod target (Minecraft first)

Non-goal: becoming a general DAW. The editor is deliberately scoped to “edit generated SFX quickly”.

## Minimum feature set that earns the title

If these are true, S‑NDB‑UND earns “one‑stop shop for creature + game SFX”:

### 1) Engines
- Stable Audio Open engine available
- Procedural engine available
- Multi-engine support through the registry (consistent CLI/UI surface)

### 2) Post-processing
- Named FX chains:
  - `tight_game_ready`
  - `creature_grit`
  - `distant_ambient`
  - `ui_polish`
  - `clean_normalized`
- Core blocks exist and are usable in chains: normalization, filters, compressor, limiter, saturation/drive, reverb

### 3) Editor (v1)
- Single-file destructive editor (working copy) with multi-step undo/redo
- Waveform view with: zoom, pan/scroll, selection, time cursor
- Edits: trim, cut/delete selection, fade in/out
- Loudness/gain: ±dB gain and normalize to a peak target (default −1 dBFS)
- Loop audition: play selection looped to seam-check ambience
- Export:
  - overwrite
  - save as new variation (auto-suffix naming)

### 4) Presets + batch
- Presets can define: engine + prompt + params + FX chain
- Batch generation supports: count/variants, auto-naming, metadata sidecars
- Each export retains metadata: seed, engine, preset, FX chain, post params, timestamps

### 5) Game integration (Minecraft v1)
- Minecraft mob soundset generator exists:
  - emits hurt/death/ambient/step
  - updates or outputs `sounds.json` snippets
  - supports variants

## How it fits S‑NDB‑UND’s architecture

Treat the editor as an additional layer on the existing pipeline:

**Engine → Post‑FX → Editor → Export**

Key workflow requirements:
- “Open last render in editor” should be a first-class path from CLI + Web UI.
- Editor can also open any WAV from disk.
- Edits are destructive on a working copy, with undo history.
- Final export writes:
  - processed WAV
  - updated metadata (trim points, gain, edits performed; plus the existing credits/repro fields)

## Phasing (so it doesn’t explode)

### Phase A — One‑stop for generation + polish (MVP)
Goal: never leave S‑NDB‑UND for the common loop.

Deliverables:
- Stable Audio Open engine
- FX chains (including JSON-defined chains)
- Batch generation with presets
- Minimal waveform editor:
  - open last generated file
  - trim, fade, normalize, loop audition
  - export overwrite / export variation

Exit criteria:
- From a prompt, you can generate → polish → trim → export to a Minecraft-ready asset without opening another app.

### Phase B — One‑stop for full mob soundsets
Goal: end-to-end “mob audio pack” workflow.

Deliverables:
- Mob soundset command that:
  - generates all needed sounds
  - applies presets + FX
  - writes/updates `sounds.json` entries
- Editor can open any generated file from that set for quick fixes

Exit criteria:
- A modder can go from nothing → full mob audio → in-game without leaving the tool.

### Phase C — One‑stop where you “live” (productization)
Goal: cohesive GUI experience for the entire workflow.

Deliverables:
- GUI for:
  - preset selection
  - generate button
  - FX chain picker
  - waveform editor
  - a lightweight project view for a mob or resource pack

Exit criteria:
- The GUI becomes the primary way to work end-to-end, with CLI remaining fully capable.

## Editor v1: definition of complete (SFX-focused)

A v1 editor is “complete” when it can replace the 80% use-case of opening Audacity just to do quick fixes:

Must-have:
- Waveform view: zoom, scroll, selection
- Basic edits: cut/copy/paste/delete, trim to selection
- Fades: fade in/out, crossfade selection
- Gain: adjust volume, normalize to target peak
- Loop tools: snap selection to loop, audition loop
- Undo/redo: multi-step history
- Markers: transients, loop points, good takes

Nice-to-have (v2):
- Spectrogram view (click/noise hunting)
- Multi-region per file (slice into many exports)
- Simple layering (overlay 2–3 sounds; not multitrack)

## Suggested implementation stack (Python)

- GUI: PySide6 or PyQt6
- Waveform rendering: pyqtgraph (fast) or custom QPainter using downsampled peaks
- Playback: sounddevice
- Processing: numpy/scipy (and reuse existing post chain code)

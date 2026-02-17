# Creature family LoRA training (Windows)

This repo supports **inference-time LoRA loading** for `stable_audio_open`. Training is intentionally *external* (you can use any LoRA trainer that outputs a diffusers-compatible LoRA).

On Windows, the most reliable path is **WSL2 + Ubuntu** (still “Windows”, but with a real Linux CUDA stack). Native Windows training can work, but it’s more fragile.

## 0) Prepare the dataset (inside this repo)

Export a training-ready folder from your existing generations:

```powershell
# From the repo root
python -m soundgen.creature_finetune prepare --in outputs --out datasets\ghoul_family --family ghoul --copy-audio --convert-to-wav
```

Output:
- `datasets/ghoul_family/audio/…`
- `datasets/ghoul_family/metadata.jsonl`

Each line includes `file_name` + `text` (the prompt), plus embedded credits.

## 1) Decide how “Windows” you mean

### Option A — WSL2 (recommended)

Use WSL2 if you have an NVIDIA GPU and want the least pain.

High level steps:
1) Install WSL2 + Ubuntu
2) Install NVIDIA’s WSL driver + CUDA toolkit support
3) Create a Python venv in WSL
4) Install your chosen LoRA trainer (diffusers+accelerate recommended)
5) Train a LoRA using `audio/` + `metadata.jsonl`

Why this is best:
- CUDA + PyTorch + accelerate are most stable on Linux.
- Fewer edge cases with dependencies.

### Option B — Native Windows (fallback)

High level steps:
1) Install Python 3.12 (recommended) or use conda
2) Install a matching PyTorch CUDA build for your GPU/driver
3) Install your LoRA trainer stack
4) Train from `datasets\...`

Native Windows usually fails due to one of:
- CUDA/torch build mismatch
- wheels missing for one dependency
- path/long-path issues

## 2) Training stack recommendation

### Best default: diffusers + accelerate

- Trainer stack: `diffusers`, `accelerate`, `transformers`, `peft`, `datasets`
- Output goal: a LoRA file that can be loaded by diffusers’ `load_lora_weights()` (often `.safetensors`).

Notes:
- Training scripts for *text-to-audio* diffusion models vary by model and diffusers version.
- Use a trainer that explicitly supports your target pipeline/model.

## 3) Bring the LoRA back into S‑NDB‑UND

1) Put the LoRA file somewhere (example): `lora/ghoul.safetensors`
2) Add a creature family entry:

Create `configs/creature_families.json` (or use `library/creature_families.json` as a local override):

```json
{
  "ghoul": {
    "lora_path": "lora/ghoul.safetensors",
    "trigger": "ghoul",
    "scale": 0.8,
    "negative_prompt": "music, singing, speech, vocals"
  }
}
```

3) Generate using the family:

```powershell
python -m soundgen.generate --engine stable_audio_open --creature-family ghoul --prompt "creature screech" --seconds 1.6 --seed 123 --post --out outputs\ghoul_screech.wav
```

## 4) Practical tips for creature families

- Start with ~50–200 clean examples per family.
- Keep prompts consistent (same style words) to teach a tight “family identity”.
- Prefer short clips (0.6–2.5s) for vocalizations; longer clips for ambience.
- Keep post-processing conservative during dataset creation; heavy polish can bake in artifacts.

## 5) What I need from you to make this guide exact

Reply with:
- GPU model + VRAM (or “CPU only”)
- Whether you can use WSL2

Then I’ll add a concrete command set for the best-fitting trainer stack.

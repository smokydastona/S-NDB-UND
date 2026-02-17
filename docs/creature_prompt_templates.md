# Creature vocalization prompt templates

These templates are designed to work across:

- `stable_audio_open` (Stable Audio Open 1.0 via diffusers)
- `diffusers` (AudioLDM2)
- `layered` (procedural hybrid: sample transient/tail + synth body)

They aim for **Minecraft-friendly SFX**: short, mono, non-musical, no dialogue.

## Core structure (copy/paste template)

Use this as your “prompt skeleton”:

`<creature> <vocalization>, <size>, <emotion>, <texture>, <delivery>, <recording>, <tail>`

Where:

- **creature**: species/archetype (“wolf”, “orc brute”, “slime”, “insect swarm”, “undead”)
- **vocalization**: “growl / roar / snarl / hiss / chitter / screech / moan / bark / yelp / howl”
- **size**: “small / medium / large / massive” + “tight / chesty / heavy”
- **emotion**: “aggressive / scared / calm / neutral”
- **texture**: “gritty / breathy / wet / raspy / clicky / buzzing / airy”
- **delivery**: “short / staccato / burst / sustained / rising / falling / up-down”
- **recording**: “close mic, dry, no reverb, no music”
- **tail**: “short tail / clean cutoff / slight tail / long tail (ambience)”

## Engine-specific guidance

### Stable Audio Open (`stable_audio_open`)

- Use a **negative prompt** to keep it non-musical.
- Good default negative prompt:
  - `music, melody, singing, speech, words, lyrics, laughter, drums`
- If the model is gated on Hugging Face, users must accept terms + authenticate.

Example:

`monster growl, medium size, aggressive, gritty rasp, short burst, close mic, dry, clean cutoff`

### AudioLDM2 (`diffusers`)

- Be explicit about **“sound effect”** and **no music/no speech**.
- For large roars, try `--diffusers-multiband` to reduce muddiness (slower).

### Layered (`layered`)

- Prompts work best when they include **texture words** (“rasp/chitter/buzz/screech”) because the engine auto-selects granular and sample layers.
- For creature vocals, start with:
  - `--engine layered --layered-preset creature --post --polish`

## Templates by vocalization type

Each entry includes a “base prompt” plus optional modifiers.

### 1) Growl (tight, usable in combat)

Base:

`<creature> growl, medium size, aggressive, gritty, short burst, close mic, dry, clean cutoff`

Modifiers:

- Larger: `large, chest resonance, heavy low end`
- More threat: `snarl, breath noise, harsh texture`
- Cleaner: `clear midrange, controlled lows`

### 2) Roar (big, cinematic)

Base:

`<creature> roar, massive size, aggressive, powerful low end, chest resonance, sustained then release, close mic, minimal reverb`

Modifiers:

- More animal: `organic, throat resonance`
- More demonic: `infernal, distorted, gritty`
- Shorter gameplay version: `short roar, no long tail, clean cutoff`

### 3) Snarl (bite / warning)

Base:

`<creature> snarl, small-medium size, aggressive, teeth, breathy rasp, staccato, close mic, dry`

### 4) Hiss (air + threat)

Base:

`<creature> hiss, medium size, aggressive, breathy, airy noise, short, close mic, dry, clean cutoff`

Modifiers:

- Snake-like: `thin, high, sharp`
- Wet: `wet air, saliva, sizzle`

### 5) Chitter (insect / skitter)

Base:

`small creature chitter, insectoid, fast clicks, sharp transients, staccato bursts, close mic, dry`

Modifiers:

- More swarm: `many tiny clicks, jittery`
- More harsh: `scratchy, brittle`

### 6) Screech / shriek (high terror)

Base:

`<creature> screech, medium size, scared, harsh, piercing, rising, short, close mic, dry`

Modifiers:

- Less painful: `controlled highs, not too loud`
- More monster: `distorted, gritty`

### 7) Moan (undead / hollow)

Base:

`undead moan, large, hollow, airy, low and slow, sustained, dry, minimal reverb`

Modifiers:

- More ghost: `ethereal, whispery, spectral tail`
- More zombie: `raspy breath, gritty`

### 8) Bark / yelp (wolf-like reactions)

Base:

`wolf bark, medium, alert, sharp transient, short, close mic, dry, clean cutoff`

Variant:

`wolf yelp, small-medium, scared, short burst, higher pitch, dry`

### 9) Howl (long tone)

Base:

`wolf howl, medium-large, calm, sustained tone, smooth, minimal noise, slight natural tail`

Gameplay-friendly variant:

`short howl, no long tail, clean cutoff`

## “Do / Don’t” (quick rules)

Do:

- Say **“sound effect”**, **“close mic”**, **“dry”**, **“no music”**, **“no speech”**
- Specify **attack shape**: “short burst / staccato / sustained then release”
- Specify **tail intent**: “clean cutoff” vs “slight tail”

Don’t:

- Ask for “song”, “chant”, “lyrics”, “voice line”, “dialogue”
- Leave the tail ambiguous if you need Minecraft snappy UX

## Handy CLI starting points

Layered (fast, consistent):

```powershell
python -m soundgen.generate --engine layered --layered-preset creature --polish --post --seconds 2.2 --seed 42 --prompt "creature growl, gritty, short burst, close mic, dry" --out outputs\creature_growl.wav
```

Stable Audio Open (best quality when available; can be gated):

```powershell
python -m soundgen.generate --engine stable_audio_open --stable-audio-negative-prompt "music, melody, singing, speech, words" --stable-audio-steps 100 --stable-audio-guidance-scale 7 --post --seconds 2.2 --seed 42 --prompt "monster growl, medium size, aggressive, gritty rasp, short burst, close mic, dry" --out outputs\creature_growl_ai.wav
```

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class GenerationParams:
    prompt: str
    seconds: float = 3.0
    seed: Optional[int] = None
    device: str = "cpu"  # "cpu" or "cuda"
    model: str = "cvssp/audioldm2"  # diffusers AudioLDM2 checkpoint
    sample_rate: int = 16000

    # Optional: multi-band generation strategy (runs the model multiple times and recombines bands).
    # This is a pragmatic approximation of "multi-band diffusion" without retraining.
    multiband: bool = False
    multiband_mode: str = "auto"  # auto|2band|3band
    multiband_low_hz: float = 250.0
    multiband_high_hz: float = 3000.0


def _to_mono_float32(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim == 2:
        a = a.mean(axis=1)
    return np.clip(a, -1.0, 1.0).astype(np.float32, copy=False)


def _butter_filter(x: np.ndarray, sr: int, *, kind: str, cutoff_hz: float, order: int = 4) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32, copy=False)

    from scipy import signal

    nyq = 0.5 * float(sr)
    c = float(cutoff_hz)
    if c <= 0.0:
        return x.astype(np.float32, copy=False)
    wn = c / nyq
    if wn <= 0.0 or wn >= 1.0:
        return x.astype(np.float32, copy=False)
    b, a = signal.butter(int(order), wn, btype=str(kind))
    return signal.filtfilt(b, a, x.astype(np.float32, copy=False)).astype(np.float32, copy=False)


def _bandpass(x: np.ndarray, sr: int, *, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32, copy=False)

    # Simple cascade: highpass then lowpass.
    y = _butter_filter(x, sr, kind="highpass", cutoff_hz=float(low_hz), order=order)
    y = _butter_filter(y, sr, kind="lowpass", cutoff_hz=float(high_hz), order=order)
    return y.astype(np.float32, copy=False)


def _parse_multiband_mode(mode: str, *, seconds: float) -> str:
    m = str(mode or "auto").strip().lower()
    if m in {"2", "2band", "two", "lowhigh"}:
        return "2band"
    if m in {"3", "3band", "three", "lowmidhigh"}:
        return "3band"
    # auto
    return "2band" if float(seconds) <= 1.4 else "3band"


def _seed_for_band(seed: int | None, band: str) -> int | None:
    if seed is None:
        return None
    s = int(seed)
    if band == "low":
        return s ^ 0x1357_2468
    if band == "mid":
        return s ^ 0xBEEF_1021
    if band == "high":
        return s ^ 0xC0DE_7731
    return s


def generate_audio(params: GenerationParams) -> tuple[np.ndarray, int]:
    """Generate mono audio from a text prompt using Diffusers.

    Returns:
        (audio, sample_rate) where audio is float32 in [-1, 1] shaped (num_samples,).
    """

    # Lazy import keeps `--help` fast even if deps are missing.
    from diffusers import AudioLDM2Pipeline
    from transformers import GPT2LMHeadModel

    device = params.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")

    torch_device = torch.device(device)

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = AudioLDM2Pipeline.from_pretrained(params.model, torch_dtype=dtype)
    pipe = pipe.to(torch_device)

    # Some environments end up with a plain GPT2Model loaded, but diffusers' AudioLDM2
    # generation path requires GPT2LMHeadModel generation helpers.
    if pipe.language_model.__class__.__name__ != "GPT2LMHeadModel":
        language_model = GPT2LMHeadModel.from_pretrained(params.model, subfolder="language_model", torch_dtype=dtype)
        pipe.language_model = language_model.to(torch_device)

    def _run_one(*, prompt: str, seed: int | None) -> tuple[np.ndarray, int]:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=torch_device)
            generator.manual_seed(int(seed))

        # Diffusers AudioLDM2 expects length in seconds via `audio_length_in_s`.
        result = pipe(
            prompt=str(prompt),
            audio_length_in_s=float(params.seconds),
            generator=generator,
        )

        audio = _to_mono_float32(result.audios[0])
        sample_rate = getattr(pipe, "sample_rate", params.sample_rate)
        return audio, int(sample_rate)

    if not bool(getattr(params, "multiband", False)):
        return _run_one(prompt=params.prompt, seed=(int(params.seed) if params.seed is not None else None))

    mode = _parse_multiband_mode(str(getattr(params, "multiband_mode", "auto")), seconds=float(params.seconds))
    low_hz = float(getattr(params, "multiband_low_hz", 250.0))
    high_hz = float(getattr(params, "multiband_high_hz", 3000.0))

    # Keep crossovers sane for typical 16k/48k models.
    # (We clamp again after we know the actual sample_rate.)
    base_prompt = str(params.prompt)

    if mode == "2band":
        # Use geometric mean as a reasonable single split point.
        split = float(np.sqrt(max(40.0, low_hz) * max(200.0, high_hz)))

        low_prompt = f"{base_prompt}, low-frequency rumble and weight, sub and thump, avoid hiss"
        high_prompt = f"{base_prompt}, crisp transient detail, high-frequency texture and air, avoid rumble"

        low_raw, sr = _run_one(prompt=low_prompt, seed=_seed_for_band(params.seed, "low"))
        high_raw, sr2 = _run_one(prompt=high_prompt, seed=_seed_for_band(params.seed, "high"))
        sr = int(sr)
        if int(sr2) != sr:
            raise RuntimeError(f"Diffusers multiband: sample_rate mismatch ({sr} vs {sr2})")

        split = float(np.clip(split, 80.0, 0.45 * float(sr)))
        low = _butter_filter(low_raw, sr, kind="lowpass", cutoff_hz=split, order=4)
        high = _butter_filter(high_raw, sr, kind="highpass", cutoff_hz=split, order=4)

        y = (1.00 * low + 0.95 * high).astype(np.float32, copy=False)
        return np.clip(y, -1.0, 1.0).astype(np.float32, copy=False), sr

    # 3-band
    low_prompt = f"{base_prompt}, low-frequency rumble and weight, sub and thump, avoid hiss"
    mid_prompt = f"{base_prompt}, midrange body and character, clear and present, avoid rumble and hiss"
    high_prompt = f"{base_prompt}, crisp transient detail, high-frequency texture and air, avoid rumble"

    low_raw, sr = _run_one(prompt=low_prompt, seed=_seed_for_band(params.seed, "low"))
    mid_raw, sr2 = _run_one(prompt=mid_prompt, seed=_seed_for_band(params.seed, "mid"))
    high_raw, sr3 = _run_one(prompt=high_prompt, seed=_seed_for_band(params.seed, "high"))
    sr = int(sr)
    if int(sr2) != sr or int(sr3) != sr:
        raise RuntimeError(f"Diffusers multiband: sample_rate mismatch ({sr}, {sr2}, {sr3})")

    lo = float(np.clip(low_hz, 40.0, 0.35 * float(sr)))
    hi = float(np.clip(high_hz, lo * 1.2, 0.45 * float(sr)))

    low = _butter_filter(low_raw, sr, kind="lowpass", cutoff_hz=lo, order=4)
    mid = _bandpass(mid_raw, sr, low_hz=lo, high_hz=hi, order=4)
    high = _butter_filter(high_raw, sr, kind="highpass", cutoff_hz=hi, order=4)

    y = (1.00 * low + 1.00 * mid + 0.90 * high).astype(np.float32, copy=False)
    return np.clip(y, -1.0, 1.0).astype(np.float32, copy=False), sr

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import butter, lfilter


@dataclass(frozen=True)
class SynthParams:
    prompt: str
    seconds: float = 3.0
    seed: Optional[int] = None

    sample_rate: int = 44100

    # Oscillator
    waveform: str = "sine"  # sine|square|saw|triangle|noise
    freq_hz: float = 440.0
    pitch_min: float = 0.90
    pitch_max: float = 1.10

    # Envelope (ms)
    attack_ms: float = 5.0
    decay_ms: float = 80.0
    sustain_level: float = 0.35
    release_ms: float = 120.0

    # Tone shaping
    noise_mix: float = 0.05  # 0..1
    lowpass_hz: Optional[float] = 16000.0
    highpass_hz: Optional[float] = 30.0

    # FX
    drive: float = 0.0  # 0..1 (tanh drive)


def _oscillator(waveform: str, phase: np.ndarray) -> np.ndarray:
    w = waveform.strip().lower()
    if w == "sine":
        return np.sin(phase)
    if w == "square":
        return np.sign(np.sin(phase))
    if w == "saw":
        # Sawtooth in [-1,1]
        return 2.0 * (phase / (2.0 * math.pi) - np.floor(0.5 + phase / (2.0 * math.pi)))
    if w == "triangle":
        saw = 2.0 * (phase / (2.0 * math.pi) - np.floor(0.5 + phase / (2.0 * math.pi)))
        return 2.0 * np.abs(saw) - 1.0
    if w == "noise":
        # phase ignored
        return np.zeros_like(phase)
    raise ValueError(f"Unsupported waveform: {waveform}")


def _adsr_envelope(n: int, sr: int, *, a_ms: float, d_ms: float, s: float, r_ms: float) -> np.ndarray:
    a = max(0, int(round(float(a_ms) * sr / 1000.0)))
    d = max(0, int(round(float(d_ms) * sr / 1000.0)))
    r = max(0, int(round(float(r_ms) * sr / 1000.0)))
    s = float(np.clip(s, 0.0, 1.0))

    # Sustain lasts for the remaining samples.
    sustain_n = max(0, n - (a + d + r))

    parts: list[np.ndarray] = []
    if a > 0:
        parts.append(np.linspace(0.0, 1.0, a, endpoint=False, dtype=np.float32))
    if d > 0:
        parts.append(np.linspace(1.0, s, d, endpoint=False, dtype=np.float32))
    if sustain_n > 0:
        parts.append(np.full(sustain_n, s, dtype=np.float32))
    if r > 0:
        parts.append(np.linspace(s, 0.0, r, endpoint=True, dtype=np.float32))

    env = np.concatenate(parts) if parts else np.zeros(n, dtype=np.float32)
    if env.shape[0] < n:
        env = np.pad(env, (0, n - env.shape[0]), mode="constant")
    return env[:n].astype(np.float32, copy=False)


def _butter_filter(audio: np.ndarray, sr: int, *, kind: str, cutoff_hz: float) -> np.ndarray:
    cutoff = float(cutoff_hz)
    if cutoff <= 0:
        return audio
    nyq = 0.5 * sr
    norm = min(0.999, max(1e-5, cutoff / nyq))
    b, a = butter(2, norm, btype=kind)
    return lfilter(b, a, audio).astype(np.float32, copy=False)


def generate_with_synth(params: SynthParams) -> tuple[np.ndarray, int]:
    sr = int(params.sample_rate)
    n = max(1, int(round(float(params.seconds) * sr)))

    rng = random.Random(params.seed)

    pmin = float(params.pitch_min)
    pmax = float(params.pitch_max)
    if pmax < pmin:
        pmin, pmax = pmax, pmin
    pitch = rng.uniform(pmin, pmax)

    freq = max(10.0, float(params.freq_hz) * float(pitch))
    t = np.arange(n, dtype=np.float32) / float(sr)
    phase = 2.0 * math.pi * freq * t

    base = _oscillator(params.waveform, phase).astype(np.float32, copy=False)
    if params.waveform.strip().lower() == "noise":
        base = rng.uniform(-1.0, 1.0) * np.ones_like(base, dtype=np.float32)

    noise_mix = float(np.clip(params.noise_mix, 0.0, 1.0))
    if noise_mix > 0.0:
        noise = np.random.default_rng(int(params.seed) if params.seed is not None else None).standard_normal(n).astype(
            np.float32
        )
        noise = noise / (float(np.max(np.abs(noise))) + 1e-9)
        base = (1.0 - noise_mix) * base + noise_mix * noise

    env = _adsr_envelope(
        n,
        sr,
        a_ms=float(params.attack_ms),
        d_ms=float(params.decay_ms),
        s=float(params.sustain_level),
        r_ms=float(params.release_ms),
    )

    audio = (base * env).astype(np.float32, copy=False)

    if params.highpass_hz is not None and float(params.highpass_hz) > 0:
        audio = _butter_filter(audio, sr, kind="highpass", cutoff_hz=float(params.highpass_hz))
    if params.lowpass_hz is not None and float(params.lowpass_hz) > 0:
        audio = _butter_filter(audio, sr, kind="lowpass", cutoff_hz=float(params.lowpass_hz))

    drive = float(np.clip(params.drive, 0.0, 1.0))
    if drive > 0.0:
        k = 1.0 + 12.0 * drive
        audio = np.tanh(k * audio).astype(np.float32, copy=False)

    # Normalize safely
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = (audio / peak).astype(np.float32, copy=False)

    audio = np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)
    return audio, sr

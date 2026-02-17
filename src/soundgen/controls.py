from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ControlHints:
    """Lightweight prompt-to-control hints.

    This is intentionally simple: it nudges defaults for envelopes/filters
    and post-processing without trying to be a full semantic parser.
    """

    loudness_rms_db: float | None = None
    pitch_min: float | None = None
    pitch_max: float | None = None
    highpass_hz: float | None = None
    lowpass_hz: float | None = None
    attack_ms: float | None = None
    release_ms: float | None = None
    drive: float | None = None


def map_prompt_to_controls(prompt: str) -> ControlHints:
    p = (prompt or "").lower()

    hints = ControlHints()

    # Loudness (post-process RMS target)
    if any(k in p for k in ("loud", "punchy", "slam", "impact")):
        hints.loudness_rms_db = -14.0
    if any(k in p for k in ("soft", "quiet", "gentle", "subtle")):
        hints.loudness_rms_db = -22.0

    # Filter / timbre
    if any(k in p for k in ("muffled", "dull", "dark")):
        hints.lowpass_hz = 6000.0
    if any(k in p for k in ("bright", "sparkle", "crispy")):
        hints.lowpass_hz = 18000.0
        hints.highpass_hz = 80.0

    if any(k in p for k in ("boomy", "bass", "rumble")):
        hints.lowpass_hz = 9000.0
        hints.highpass_hz = 15.0

    # Envelope feel
    if any(k in p for k in ("click", "clicky", "pluck", "plucky", "snap")):
        hints.attack_ms = 1.0
        hints.release_ms = 60.0
    if any(k in p for k in ("pad", "swell", "whoosh")):
        hints.attack_ms = 80.0
        hints.release_ms = 220.0

    # Grit
    if any(k in p for k in ("distort", "distorted", "gritty", "dirty")):
        hints.drive = 0.35

    # Pitch range nudges
    if any(k in p for k in ("high pitch", "high-pitch", "chirp", "bleep")):
        hints.pitch_min = 1.05
        hints.pitch_max = 1.25
    if any(k in p for k in ("low", "deep", "thud")):
        hints.pitch_min = 0.80
        hints.pitch_max = 0.98

    return hints

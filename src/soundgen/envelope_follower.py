from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FollowerCoeffs:
    attack_c: float
    release_c: float


def coeffs_from_ms(*, sr: int, attack_ms: float, release_ms: float) -> FollowerCoeffs:
    """Return one-pole smoothing coefficients for attack/release in milliseconds."""

    sr_f = float(max(1, int(sr)))
    atk_s = max(0.0005, float(attack_ms) * 0.001)
    rel_s = max(0.001, float(release_ms) * 0.001)
    attack_c = float(np.exp(-1.0 / (sr_f * atk_s)))
    release_c = float(np.exp(-1.0 / (sr_f * rel_s)))
    return FollowerCoeffs(attack_c=attack_c, release_c=release_c)


def follow_envelope(
    x: np.ndarray,
    sr: int,
    *,
    attack_ms: float,
    release_ms: float,
    mode: str = "peak",  # peak|rms
) -> np.ndarray:
    """Attack/release envelope follower.

    mode:
      - peak: follows abs(x)
      - rms: follows power (x^2) then sqrt
    """

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("follow_envelope expects mono 1D audio")
    if x.size == 0:
        return x.astype(np.float32, copy=False)

    m = str(mode or "peak").strip().lower()
    if m not in {"peak", "rms"}:
        raise ValueError("mode must be 'peak' or 'rms'")

    c = coeffs_from_ms(sr=int(sr), attack_ms=float(attack_ms), release_ms=float(release_ms))
    atk_c = float(c.attack_c)
    rel_c = float(c.release_c)

    if m == "peak":
        drive = np.abs(x).astype(np.float32, copy=False)
        env = 0.0
        out = np.empty_like(drive, dtype=np.float32)
        for i in range(drive.size):
            s = float(drive[i])
            if s > env:
                env = atk_c * env + (1.0 - atk_c) * s
            else:
                env = rel_c * env + (1.0 - rel_c) * s
            out[i] = float(env)
        return out

    # RMS-ish: follow power.
    drive_p = np.square(x, dtype=np.float32)
    env_p = 0.0
    out_p = np.empty_like(drive_p, dtype=np.float32)
    for i in range(drive_p.size):
        s = float(drive_p[i])
        if s > env_p:
            env_p = atk_c * env_p + (1.0 - atk_c) * s
        else:
            env_p = rel_c * env_p + (1.0 - rel_c) * s
        out_p[i] = float(env_p)

    return np.sqrt(out_p, dtype=np.float32)

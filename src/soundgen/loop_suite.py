from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import read_wav_mono, write_wav


def _clip_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        x = x.reshape(-1).astype(np.float32)
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def _db_to_lin(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def _smooth_abs_envelope(x: np.ndarray, *, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window == 1:
        return np.abs(x).astype(np.float32)

    # Moving average of abs(x)
    absx = np.abs(x).astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(absx, kernel, mode="same").astype(np.float32)


def _nearest_zero_crossing(x: np.ndarray, idx: int, *, radius: int) -> int:
    idx = int(idx)
    radius = max(0, int(radius))
    n = int(x.shape[0])
    if n == 0:
        return 0

    lo = max(0, idx - radius)
    hi = min(n - 1, idx + radius)
    if lo >= hi:
        return max(0, min(n - 1, idx))

    # Prefer exact sign changes; otherwise choose minimum abs sample.
    best = idx
    best_abs = float("inf")
    for i in range(lo + 1, hi + 1):
        a = float(x[i - 1])
        b = float(x[i])
        if (a <= 0.0 <= b) or (b <= 0.0 <= a):
            return i
        ab = abs(b)
        if ab < best_abs:
            best_abs = ab
            best = i
    return best


def _equal_power_crossfade(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Crossfade two equal-length mono segments (equal-power)."""

    if a.shape != b.shape:
        raise ValueError("crossfade segments must have same shape")
    n = int(a.shape[0])
    if n == 0:
        return a.astype(np.float32)

    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
    fade_out = np.cos(0.5 * math.pi * t).astype(np.float32)
    fade_in = np.sin(0.5 * math.pi * t).astype(np.float32)
    return (a.astype(np.float32) * fade_out + b.astype(np.float32) * fade_in).astype(np.float32)


@dataclass(frozen=True)
class AutoLoopResult:
    loop_start: int
    loop_end: int
    score: float


def find_auto_loop_points(
    x: np.ndarray,
    sr: int,
    *,
    min_loop_s: float = 1.0,
    max_loop_s: float = 6.0,
    window_ms: float = 60.0,
    step_ms: float = 25.0,
    zero_cross_radius_ms: float = 5.0,
) -> AutoLoopResult:
    """Find loop points by matching a tail window against earlier audio.

    This is intentionally simple and robust for ambience-like material.
    It searches for a loop start such that the first window and last window of
    the loop segment are highly correlated (low seam error).
    """

    x = np.asarray(x, dtype=np.float32).reshape(-1)
    sr = int(sr)
    if sr <= 0:
        raise ValueError("sample rate must be positive")
    n = int(x.shape[0])
    if n < 4:
        return AutoLoopResult(loop_start=0, loop_end=n, score=0.0)

    min_len = max(1, int(float(min_loop_s) * sr))
    max_len = max(min_len, int(float(max_loop_s) * sr))

    w = max(16, int(float(window_ms) * sr / 1000.0))
    step = max(1, int(float(step_ms) * sr / 1000.0))
    zrad = max(0, int(float(zero_cross_radius_ms) * sr / 1000.0))

    # Fix loop end at end-of-file (typical for generated ambience); we can refine
    # end to a nearby zero crossing.
    loop_end = _nearest_zero_crossing(x, n - 1, radius=zrad)
    loop_end = max(w + 1, loop_end)

    # Tail window to match (ending at loop_end)
    tail_end = loop_end
    tail_start = max(0, tail_end - w)
    tail = x[tail_start:tail_end]
    if tail.shape[0] < w:
        # Pad by shifting earlier.
        tail_start = max(0, tail_end - w)
        tail = x[tail_start:tail_end]

    tail = tail.astype(np.float32)
    tail = tail - float(np.mean(tail))
    tail_norm = float(np.linalg.norm(tail) + 1e-9)

    best_s = max(0, loop_end - min_len)
    best_score = -1.0

    search_lo = max(0, loop_end - max_len)
    search_hi = max(0, loop_end - min_len)

    for s in range(search_lo, search_hi + 1, step):
        head_end = s + w
        if head_end >= loop_end:
            break
        head = x[s:head_end].astype(np.float32)
        if head.shape[0] != w:
            continue
        head = head - float(np.mean(head))
        head_norm = float(np.linalg.norm(head) + 1e-9)
        corr = float(np.dot(head, tail) / (head_norm * tail_norm))

        # A small penalty for extremely short loops (helps avoid 1-beat loops)
        loop_len = float(loop_end - s)
        length_pen = 0.0
        if loop_len < float(2.0 * sr):
            length_pen = (float(2.0 * sr) - loop_len) / float(2.0 * sr)

        score = corr - 0.15 * length_pen
        if score > best_score:
            best_score = score
            best_s = s

    best_s = _nearest_zero_crossing(x, best_s, radius=zrad)
    best_e = _nearest_zero_crossing(x, loop_end, radius=zrad)
    best_e = max(best_s + min_len, best_e)
    best_e = min(n, best_e)

    return AutoLoopResult(loop_start=int(best_s), loop_end=int(best_e), score=float(best_score))


def apply_loop_crossfade(
    x: np.ndarray,
    *,
    loop_start: int,
    loop_end: int,
    crossfade_ms: float,
    sr: int,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    sr = int(sr)
    loop_start = int(loop_start)
    loop_end = int(loop_end)
    if loop_end <= loop_start:
        return x

    seg = x[loop_start:loop_end].copy()
    cf = max(0, int(float(crossfade_ms) * sr / 1000.0))
    if cf <= 0:
        return seg

    cf = min(cf, max(1, seg.shape[0] // 2))
    head = seg[:cf]
    tail = seg[-cf:]
    xfade = _equal_power_crossfade(tail, head)
    seg[:cf] = xfade
    return seg


def trim_tail(
    x: np.ndarray,
    sr: int,
    *,
    threshold_db: float = -45.0,
    hold_ms: float = 200.0,
    lookback_ms: float = 30.0,
    keep_ms: float = 30.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Trim low-energy tail.

    Uses a smoothed abs envelope and finds the last time the envelope exceeds
    the threshold, then keeps a small pad.
    """

    x = np.asarray(x, dtype=np.float32).reshape(-1)
    sr = int(sr)
    if x.size == 0:
        return x, {"trimmed": False, "reason": "empty"}

    thr = _db_to_lin(float(threshold_db))
    env_win = max(1, int(float(lookback_ms) * sr / 1000.0))
    env = _smooth_abs_envelope(x, window=env_win)

    hold = max(1, int(float(hold_ms) * sr / 1000.0))
    keep = max(0, int(float(keep_ms) * sr / 1000.0))

    above = env > thr
    if not bool(np.any(above)):
        # Nothing above threshold: keep a small slice to avoid empty outputs.
        keep_n = min(int(sr * 0.25), int(x.size))
        return x[:keep_n].copy(), {"trimmed": True, "reason": "all_below_threshold", "new_samples": keep_n}

    last = int(np.max(np.nonzero(above)))

    # Require sustained silence for `hold` before trimming.
    cut = int(x.size)
    silent_run = 0
    for i in range(last + 1, int(x.size)):
        if env[i] <= thr:
            silent_run += 1
            if silent_run >= hold:
                cut = i - hold + 1
                break
        else:
            silent_run = 0

    if cut >= int(x.size):
        return x, {"trimmed": False, "reason": "no_sustained_silence"}

    new_len = max(1, min(int(x.size), cut + keep))
    return x[:new_len].copy(), {"trimmed": True, "threshold_db": float(threshold_db), "new_samples": int(new_len)}


def extract_noise_bed(
    x: np.ndarray,
    sr: int,
    *,
    bed_seconds: float,
    window_seconds: float = 1.0,
    step_seconds: float = 0.25,
    crossfade_ms: float = 40.0,
) -> np.ndarray:
    """Extract a low-energy 'bed' window and loop it to a target duration."""

    x = np.asarray(x, dtype=np.float32).reshape(-1)
    sr = int(sr)
    n = int(x.size)
    if n == 0:
        return x

    win = max(16, int(float(window_seconds) * sr))
    step = max(1, int(float(step_seconds) * sr))
    if n <= win:
        bed = x.copy()
    else:
        best_i = 0
        best_rms = float("inf")
        for i in range(0, n - win + 1, step):
            seg = x[i : i + win]
            rms = float(np.sqrt(np.mean(seg.astype(np.float32) ** 2) + 1e-12))
            if rms < best_rms:
                best_rms = rms
                best_i = i
        bed = x[best_i : best_i + win].copy()

    target = max(1, int(float(bed_seconds) * sr))
    if bed.size >= target:
        return bed[:target].copy()

    cf = max(0, int(float(crossfade_ms) * sr / 1000.0))
    if cf <= 0:
        reps = int(math.ceil(float(target) / float(bed.size)))
        out = np.tile(bed, reps)[:target].copy()
        return out

    # Loop with crossfade between repeats.
    out = bed.copy()
    while out.size < target:
        need = target - out.size
        chunk = bed if need >= bed.size else bed[:need]

        if out.size >= cf and chunk.size >= cf:
            xfade = _equal_power_crossfade(out[-cf:], chunk[:cf])
            out[-cf:] = xfade
            out = np.concatenate([out, chunk[cf:]], axis=0)
        else:
            out = np.concatenate([out, chunk], axis=0)

    return out[:target].copy()


def mix_bed_under(
    x: np.ndarray,
    bed: np.ndarray,
    *,
    bed_gain_db: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    bed = np.asarray(bed, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    if bed.size == 0:
        return x

    if bed.size < x.size:
        reps = int(math.ceil(float(x.size) / float(bed.size)))
        bed2 = np.tile(bed, reps)[: x.size]
    else:
        bed2 = bed[: x.size]

    g = float(_db_to_lin(float(bed_gain_db)))
    out = x + (bed2 * g)
    return _clip_mono(out)


def _cmd_auto(args: argparse.Namespace) -> int:
    x, sr = read_wav_mono(Path(args.input))
    res = find_auto_loop_points(
        x,
        sr,
        min_loop_s=float(args.min_loop_s),
        max_loop_s=float(args.max_loop_s),
        window_ms=float(args.window_ms),
        step_ms=float(args.step_ms),
        zero_cross_radius_ms=float(args.zero_cross_radius_ms),
    )

    seg = apply_loop_crossfade(
        x,
        loop_start=res.loop_start,
        loop_end=res.loop_end,
        crossfade_ms=float(args.crossfade_ms),
        sr=sr,
    )

    write_wav(Path(args.output), _clip_mono(seg), sr)
    print(
        f"auto-loop: start={res.loop_start / sr:.3f}s end={res.loop_end / sr:.3f}s "
        f"len={(res.loop_end - res.loop_start) / sr:.3f}s score={res.score:.4f}"
    )
    return 0


def _cmd_trim_tail(args: argparse.Namespace) -> int:
    x, sr = read_wav_mono(Path(args.input))
    y, info = trim_tail(
        x,
        sr,
        threshold_db=float(args.threshold_db),
        hold_ms=float(args.hold_ms),
        lookback_ms=float(args.lookback_ms),
        keep_ms=float(args.keep_ms),
    )
    write_wav(Path(args.output), _clip_mono(y), sr)
    print(f"trim-tail: {info}")
    return 0


def _cmd_extract_bed(args: argparse.Namespace) -> int:
    x, sr = read_wav_mono(Path(args.input))
    bed = extract_noise_bed(
        x,
        sr,
        bed_seconds=float(args.seconds),
        window_seconds=float(args.window_seconds),
        step_seconds=float(args.step_seconds),
        crossfade_ms=float(args.crossfade_ms),
    )
    write_wav(Path(args.output), _clip_mono(bed), sr)
    return 0


def _cmd_mix_bed(args: argparse.Namespace) -> int:
    x, sr = read_wav_mono(Path(args.input))
    bed, bed_sr = read_wav_mono(Path(args.bed))
    if int(bed_sr) != int(sr):
        raise ValueError(f"bed sample rate {bed_sr} does not match input {sr}")
    out = mix_bed_under(x, bed, bed_gain_db=float(args.bed_gain_db))
    write_wav(Path(args.output), _clip_mono(out), sr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Loop suite tools: auto loop points, tail trim heuristics, and noise bed helpers.\n\n"
            "All audio stays mono float32 in [-1, 1]."
        )
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    auto = sub.add_parser("auto", help="Find loop points and export a loopable segment")
    auto.add_argument("--in", dest="input", required=True, help="Input WAV")
    auto.add_argument("--out", dest="output", required=True, help="Output WAV (loop segment)")
    auto.add_argument("--min-loop-s", type=float, default=1.0)
    auto.add_argument("--max-loop-s", type=float, default=6.0)
    auto.add_argument("--window-ms", type=float, default=60.0)
    auto.add_argument("--step-ms", type=float, default=25.0)
    auto.add_argument("--zero-cross-radius-ms", type=float, default=5.0)
    auto.add_argument("--crossfade-ms", type=float, default=40.0)
    auto.set_defaults(fn=_cmd_auto)

    tt = sub.add_parser("trim-tail", help="Trim low-energy tail")
    tt.add_argument("--in", dest="input", required=True, help="Input WAV")
    tt.add_argument("--out", dest="output", required=True, help="Output WAV")
    tt.add_argument("--threshold-db", type=float, default=-45.0)
    tt.add_argument("--hold-ms", type=float, default=200.0)
    tt.add_argument("--lookback-ms", type=float, default=30.0)
    tt.add_argument("--keep-ms", type=float, default=30.0)
    tt.set_defaults(fn=_cmd_trim_tail)

    eb = sub.add_parser("extract-bed", help="Extract a low-energy bed and loop it to a duration")
    eb.add_argument("--in", dest="input", required=True, help="Input WAV")
    eb.add_argument("--out", dest="output", required=True, help="Output WAV (bed)")
    eb.add_argument("--seconds", type=float, required=True, help="Output duration seconds")
    eb.add_argument("--window-seconds", type=float, default=1.0)
    eb.add_argument("--step-seconds", type=float, default=0.25)
    eb.add_argument("--crossfade-ms", type=float, default=40.0)
    eb.set_defaults(fn=_cmd_extract_bed)

    mb = sub.add_parser("mix-bed", help="Mix a bed under a sound at a given gain")
    mb.add_argument("--in", dest="input", required=True, help="Input WAV")
    mb.add_argument("--bed", required=True, help="Bed WAV (same sample rate)")
    mb.add_argument("--out", dest="output", required=True, help="Output WAV")
    mb.add_argument("--bed-gain-db", type=float, default=-28.0)
    mb.set_defaults(fn=_cmd_mix_bed)

    return p


def run_loop_suite(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    fn = getattr(args, "fn", None)
    if not callable(fn):
        raise SystemExit(2)
    return int(fn(args))


def main() -> None:
    raise SystemExit(run_loop_suite())


if __name__ == "__main__":
    main()

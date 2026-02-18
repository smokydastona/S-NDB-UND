from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly


@dataclass(frozen=True)
class AnalysisReport:
    sample_rate: int
    seconds: float
    peak: float
    true_peak: float
    rms: float
    rms_dbfs: float
    crest_db: float
    # "Normalization factor" in the Ardour sense: multiplier to reach a target.
    normalize_to_peak_factor: float
    normalize_to_rms_factor: float


def _safe_log10(x: float) -> float:
    return float(np.log10(max(float(x), 1e-12)))


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))


def _peak(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(x)))


def compute_analysis(
    x: np.ndarray,
    sr: int,
    *,
    normalize_peak_db: float = -1.0,
    normalize_rms_db: float = -18.0,
    true_peak_oversample: int = 4,
) -> AnalysisReport:
    """Compute lightweight offline analysis metrics.

    This intentionally avoids heavy dependencies; it provides a stable "editor-grade" summary
    that can be stored in credits for QA/best-of-N and export reporting.
    """

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("compute_analysis expects mono 1D audio")

    sr_i = int(sr)
    seconds = float(x.size / sr_i) if sr_i > 0 else 0.0

    peak = _peak(x)
    rms = _rms(x)
    rms_dbfs = 20.0 * _safe_log10(rms)
    crest_db = 20.0 * _safe_log10(peak / max(rms, 1e-12))

    # True-peak approximation via oversampling + peak measurement.
    os = int(true_peak_oversample)
    if os <= 1 or x.size < 8 or sr_i <= 0:
        true_peak = float(peak)
    else:
        # resample_poly is efficient and stable for this purpose.
        xo = resample_poly(x, up=os, down=1).astype(np.float32, copy=False)
        true_peak = _peak(xo)

    peak_target = float(10.0 ** (float(normalize_peak_db) / 20.0))
    rms_target = float(10.0 ** (float(normalize_rms_db) / 20.0))

    normalize_to_peak_factor = float(peak_target / peak) if peak > 0 else 0.0
    normalize_to_rms_factor = float(rms_target / rms) if rms > 0 else 0.0

    return AnalysisReport(
        sample_rate=sr_i,
        seconds=seconds,
        peak=float(peak),
        true_peak=float(true_peak),
        rms=float(rms),
        rms_dbfs=float(rms_dbfs),
        crest_db=float(crest_db),
        normalize_to_peak_factor=float(normalize_to_peak_factor),
        normalize_to_rms_factor=float(normalize_to_rms_factor),
    )

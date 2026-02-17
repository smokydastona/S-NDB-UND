from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int, *, subtype: str = "PCM_16") -> None:
    """Write a mono WAV file.

    audio: float32 array in [-1, 1]
    """
    ensure_parent_dir(path)
    sf.write(str(path), audio, sample_rate, subtype=str(subtype))


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read audio as mono float32 in [-1, 1]."""

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        mono = data[:, 0]
    else:
        mono = np.mean(data, axis=1, dtype=np.float32)
    return mono.astype(np.float32, copy=False), int(sr)


def find_ffmpeg() -> str:
    """Return the ffmpeg executable path.

    Used for format conversions (e.g. .wav->.ogg for Minecraft export) and decoding
    sample libraries.
    """

    found = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if found:
        return found

    # WinGet installs (e.g. Gyan.FFmpeg) may not be visible until a new shell.
    local_appdata = os.getenv("LOCALAPPDATA")
    if local_appdata:
        base = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
        if base.exists():
            for p in base.glob("Gyan.FFmpeg_*/*/bin/ffmpeg.exe"):
                return str(p)

    raise FileNotFoundError(
        "ffmpeg not found on PATH. Install ffmpeg (or add it to PATH) to convert audio formats."
    )


def convert_audio_with_ffmpeg(
    in_path: Path,
    out_path: Path,
    *,
    sample_rate: Optional[int] = None,
    channels: Optional[int] = None,
    out_format: Optional[str] = None,
    ogg_quality: int = 5,
    mp3_bitrate: str = "192k",
) -> None:
    """Convert audio via ffmpeg.

    Intended for non-Minecraft exports (mp3/ogg/flac) and optional resampling.
    """

    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = (out_format or out_path.suffix.lstrip(".")).lower()
    codec_args: list[str] = []
    if fmt == "ogg":
        codec_args = ["-c:a", "libvorbis", "-qscale:a", str(int(ogg_quality))]
    elif fmt == "mp3":
        codec_args = ["-c:a", "libmp3lame", "-b:a", str(mp3_bitrate)]
    elif fmt == "flac":
        codec_args = ["-c:a", "flac"]
    elif fmt == "wav":
        # Leave codec default (pcm_s16le) unless user rewrites with soundfile.
        codec_args = []
    else:
        raise ValueError(f"Unsupported out_format for ffmpeg conversion: {fmt}")

    cmd: list[str] = [
        find_ffmpeg(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(in_path),
    ]
    if sample_rate is not None:
        cmd += ["-ar", str(int(sample_rate))]
    if channels is not None:
        cmd += ["-ac", str(int(channels))]
    cmd += codec_args
    cmd += [str(out_path)]

    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed converting audio:\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout: {completed.stdout}\n"
            f"stderr: {completed.stderr}"
        )

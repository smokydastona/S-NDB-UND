from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import sys

import numpy as np


@dataclass(frozen=True)
class StableAudioOpenParams:
    prompt: str
    seconds: float
    seed: int | None
    device: str
    model: str
    negative_prompt: str | None = None
    num_inference_steps: int = 100
    guidance_scale: float = 7.0
    hf_token: str | None = None


@lru_cache(maxsize=4)
def _load_pipeline(model: str, device: str, use_fp16: bool):
    try:
        import torch
        from diffusers import StableAudioPipeline
    except Exception as e:  # pragma: no cover
        py = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise RuntimeError(
            "Stable Audio Open requires the AI dependencies (torch + diffusers). "
            "Install with `pip install -r requirements.txt`. "
            "If installs fail, try Python 3.12 (the CI/EXE build uses 3.12). "
            f"Current Python: {py}."
        ) from e

    dtype = torch.float16 if use_fp16 else torch.float32

    pipe = StableAudioPipeline.from_pretrained(model, torch_dtype=dtype)
    pipe = pipe.to(device)

    # Small memory win; safe for inference.
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    return pipe


def _load_pipeline_with_token(model: str, device: str, use_fp16: bool, token: str):
    """Load a pipeline using an explicit HF token.

    Not cached (avoid keeping tokens in cache keys).
    """
    try:
        import torch
        from diffusers import StableAudioPipeline
    except Exception as e:  # pragma: no cover
        py = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise RuntimeError(
            "Stable Audio Open requires the AI dependencies (torch + diffusers). "
            "Install with `pip install -r requirements.txt`. "
            "If installs fail, try Python 3.12 (the CI/EXE build uses 3.12). "
            f"Current Python: {py}."
        ) from e

    dtype = torch.float16 if use_fp16 else torch.float32

    # diffusers/huggingface_hub moved from use_auth_token -> token.
    try:
        pipe = StableAudioPipeline.from_pretrained(model, torch_dtype=dtype, token=str(token))
    except TypeError:
        pipe = StableAudioPipeline.from_pretrained(model, torch_dtype=dtype, use_auth_token=str(token))
    pipe = pipe.to(device)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    return pipe


def generate_audio(p: StableAudioOpenParams) -> tuple[np.ndarray, int]:
    """Generate audio using Stability AI Stable Audio Open 1.0 via diffusers.

    Returns:
      (audio_mono_float32, sample_rate)

    Notes:
    - Model is typically gated on Hugging Face; users must accept terms and have auth configured.
    - The pipeline can return stereo @ 44.1kHz; this function downmixes to mono.
    """

    prompt = str(p.prompt)
    seconds = float(p.seconds)
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be > 0")

    # Stable Audio Open 1.0 supports up to ~47s.
    seconds = min(seconds, 47.0)

    device = str(p.device or "cpu")
    model = str(p.model)
    if not model:
        raise ValueError("Stable Audio model id must be provided")

    use_fp16 = device == "cuda"

    import torch

    hf_token = str(p.hf_token).strip() if p.hf_token else ""
    if hf_token:
        pipe = _load_pipeline_with_token(model, device, use_fp16, hf_token)
    else:
        pipe = _load_pipeline(model, device, use_fp16)

    generator = None
    if p.seed is not None:
        # Some pipelines want CPU generator even if on GPU; be permissive.
        try:
            generator = torch.Generator(device=device).manual_seed(int(p.seed))
        except Exception:
            generator = torch.Generator().manual_seed(int(p.seed))

    kwargs = {
        "prompt": prompt,
        "audio_end_in_s": float(seconds),
        "num_inference_steps": int(p.num_inference_steps),
        "guidance_scale": float(p.guidance_scale),
    }
    if p.negative_prompt:
        kwargs["negative_prompt"] = str(p.negative_prompt)
    if generator is not None:
        kwargs["generator"] = generator

    try:
        out = pipe(**kwargs)
    except Exception as e:
        msg = str(e)
        if "gated" in msg.lower() or "accept" in msg.lower() or "401" in msg or "unauthorized" in msg.lower():
            raise RuntimeError(
                "Stable Audio Open model is gated on Hugging Face. "
                "Accept the model terms on the model page and ensure your Hugging Face token is available "
                "(e.g., set HUGGINGFACE_HUB_TOKEN or run `huggingface-cli login`)."
            ) from e
        raise

    # diffusers returns a list/np array; be robust.
    audio = getattr(out, "audios", None)
    if audio is None:
        raise RuntimeError("StableAudioPipeline returned no audio")

    audio_np = np.asarray(audio)
    # Common shapes:
    # - (batch, channels, samples)
    # - (batch, samples)
    # - (channels, samples)
    # - (samples,)
    if audio_np.ndim == 3:
        audio_np = audio_np[0]
    if audio_np.ndim == 2:
        # (channels, samples) or (samples, channels)
        if audio_np.shape[0] in {1, 2}:
            audio_np = audio_np.mean(axis=0)
        else:
            audio_np = audio_np.mean(axis=1)

    audio_np = np.asarray(audio_np, dtype=np.float32).reshape(-1)
    audio_np = np.clip(audio_np, -1.0, 1.0)

    sr = getattr(pipe, "sample_rate", None)
    if sr is None:
        # Stable Audio Open is 44.1kHz.
        sr = 44100

    return audio_np, int(sr)

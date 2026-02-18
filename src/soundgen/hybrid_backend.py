from __future__ import annotations

import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .io_utils import read_wav_mono
from .samplelib_backend import SampleLibParams, generate_with_samplelib


@dataclass(frozen=True)
class HybridParams:
    prompt: str
    seconds: float
    seed: Optional[int]
    device: str

    # Which AI engine provides the base layer.
    # Supported: stable_audio_open | diffusers
    base_engine: str = "stable_audio_open"

    # Stable Audio Open
    stable_audio_model: str = "stabilityai/stable-audio-open-1.0"
    stable_audio_negative_prompt: str | None = None
    stable_audio_steps: int = 100
    stable_audio_guidance_scale: float = 7.0
    stable_audio_sampler: str | None = None
    stable_audio_hf_token: str | None = None
    stable_audio_lora_path: str | None = None
    stable_audio_lora_scale: float = 1.0
    stable_audio_lora_trigger: str | None = None

    # Diffusers (AudioLDM2)
    diffusers_model: str = "cvssp/audioldm2"
    diffusers_multiband: bool = False
    diffusers_multiband_mode: str = "auto"
    diffusers_multiband_low_hz: float = 250.0
    diffusers_multiband_high_hz: float = 3000.0

    # samplelib
    library_zips: tuple[Path, ...] = ()
    library_pitch_min: float = 0.85
    library_pitch_max: float = 1.20
    library_mix_count: int = 1
    library_index_path: Optional[Path] = None

    # layered (procedural) controls
    layered_preset: str = "auto"
    layered_preset_lock: bool = True
    layered_variant_index: int = 0
    layered_micro_variation: float = 0.0
    layered_env_curve_shape: str = "linear"

    layered_transient_tilt: float = 0.0
    layered_body_tilt: float = 0.0
    layered_tail_tilt: float = 0.0

    layered_xfade_transient_to_body_ms: float = 0.0
    layered_xfade_body_to_tail_ms: float = 0.0
    layered_xfade_curve_shape: str = "linear"

    layered_transient_hp_hz: float = 0.0
    layered_transient_lp_hz: float = 0.0
    layered_transient_drive: float = 0.0
    layered_transient_gain_db: float = 0.0

    layered_body_hp_hz: float = 0.0
    layered_body_lp_hz: float = 0.0
    layered_body_drive: float = 0.0
    layered_body_gain_db: float = 0.0

    layered_tail_hp_hz: float = 0.0
    layered_tail_lp_hz: float = 0.0
    layered_tail_drive: float = 0.0
    layered_tail_gain_db: float = 0.0

    layered_source_lock: bool = False
    layered_source_seed: int | None = None

    layered_granular_preset: str = "off"
    layered_granular_amount: float = 0.0
    layered_granular_grain_ms: float = 28.0
    layered_granular_spray: float = 0.35

    layered_transient_ms: int = 110
    layered_tail_ms: int = 350
    layered_transient_attack_ms: float = 1.0
    layered_transient_hold_ms: float = 10.0
    layered_transient_decay_ms: float = 90.0
    layered_body_attack_ms: float = 5.0
    layered_body_hold_ms: float = 0.0
    layered_body_decay_ms: float = 80.0
    layered_tail_attack_ms: float = 15.0
    layered_tail_hold_ms: float = 0.0
    layered_tail_decay_ms: float = 320.0
    layered_duck_amount: float = 0.35
    layered_duck_release_ms: float = 90.0


@dataclass(frozen=True)
class HybridResult:
    audio: np.ndarray
    sample_rate: int
    sources: tuple[dict[str, Any], ...]
    credits_extra: dict[str, Any]


def _normalize_base_engine_name(name: str) -> str:
    n = str(name or "").strip().lower()
    if n in {"stable", "stable_audio", "stable_audio_open", "stableaudioopen"}:
        return "stable_audio_open"
    if n in {"diff", "diffusers", "audioldm2"}:
        return "diffusers"
    return n


def _pad_or_trim(x: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32).reshape(-1)
    if a.size == n:
        return a
    if a.size > n:
        return a[:n].astype(np.float32, copy=False)
    out = np.zeros((n,), dtype=np.float32)
    out[: a.size] = a
    return out


def _procedural_noise_layer(*, n: int, seed: int | None, peak: float = 0.6) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    rng = np.random.default_rng(int(seed) if seed is not None else None)
    x = rng.standard_normal(int(n)).astype(np.float32, copy=False)
    # Soft clip and control level.
    x = np.tanh(1.6 * x).astype(np.float32, copy=False)
    p = float(np.max(np.abs(x))) if x.size else 0.0
    if p > 1e-12:
        x = (x * (float(peak) / p)).astype(np.float32, copy=False)
    return np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)


def generate_with_hybrid(params: HybridParams) -> HybridResult:
    """Hybrid engine: AI base layer + procedural transient/tail + procedural granular texture.

    The procedural layers reuse the same parameterization as the `layered` engine so
    existing CLI/UI controls apply.
    """

    import soundgen.layered_backend as lb

    base_engine = _normalize_base_engine_name(params.base_engine)
    prompt = str(params.prompt)
    seconds = float(params.seconds)
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be > 0")

    base_credits: dict[str, Any]
    if base_engine == "stable_audio_open":
        from .stable_audio_backend import StableAudioOpenParams, generate_audio

        sp = StableAudioOpenParams(
            prompt=prompt,
            seconds=seconds,
            seed=params.seed,
            device=str(params.device or "cpu"),
            model=str(params.stable_audio_model or "stabilityai/stable-audio-open-1.0"),
            negative_prompt=(
                str(params.stable_audio_negative_prompt)
                if params.stable_audio_negative_prompt
                else None
            ),
            num_inference_steps=int(params.stable_audio_steps),
            guidance_scale=float(params.stable_audio_guidance_scale),
            sampler=(str(params.stable_audio_sampler) if params.stable_audio_sampler else None),
            hf_token=(str(params.stable_audio_hf_token).strip() if params.stable_audio_hf_token else None),
            lora_path=(str(params.stable_audio_lora_path).strip() if params.stable_audio_lora_path else None),
            lora_scale=float(params.stable_audio_lora_scale),
            lora_trigger=(str(params.stable_audio_lora_trigger).strip() if params.stable_audio_lora_trigger else None),
        )
        base_audio, sr = generate_audio(sp)
        base_credits = {
            "model": sp.model,
            "device": sp.device,
            "seed": params.seed,
            "stable_audio_steps": int(sp.num_inference_steps),
            "stable_audio_guidance_scale": float(sp.guidance_scale),
            "stable_audio_sampler": (str(sp.sampler) if sp.sampler else None),
            "stable_audio_negative_prompt": (str(sp.negative_prompt) if sp.negative_prompt else None),
            "stable_audio_lora_path": (str(sp.lora_path) if sp.lora_path else None),
            "stable_audio_lora_scale": (float(sp.lora_scale) if sp.lora_path else None),
            "stable_audio_lora_trigger": (str(sp.lora_trigger) if sp.lora_trigger else None),
        }
    elif base_engine == "diffusers":
        # Avoid importing torch-heavy modules unless used.
        from .audiogen_backend import GenerationParams, generate_audio

        gp = GenerationParams(
            prompt=prompt,
            seconds=seconds,
            seed=params.seed,
            device=str(params.device or "cpu"),
            model=str(params.diffusers_model or "cvssp/audioldm2"),
            multiband=bool(params.diffusers_multiband),
            multiband_mode=str(params.diffusers_multiband_mode or "auto"),
            multiband_low_hz=float(params.diffusers_multiband_low_hz),
            multiband_high_hz=float(params.diffusers_multiband_high_hz),
        )
        base_audio, sr = generate_audio(gp)
        base_credits = {
            "model": gp.model,
            "device": gp.device,
            "seed": params.seed,
            "diffusers_multiband": bool(gp.multiband),
            "diffusers_multiband_mode": (str(gp.multiband_mode) if bool(gp.multiband) else None),
            "diffusers_multiband_low_hz": (float(gp.multiband_low_hz) if bool(gp.multiband) else None),
            "diffusers_multiband_high_hz": (float(gp.multiband_high_hz) if bool(gp.multiband) else None),
        }
    else:
        raise ValueError(f"Unknown hybrid base engine '{params.base_engine}'. Use stable_audio_open or diffusers.")

    sr = int(sr)
    n = max(1, int(round(seconds * float(sr))))
    body = _pad_or_trim(base_audio, n)

    lp = lb.LayeredParams(
        prompt=prompt,
        seconds=seconds,
        seed=params.seed,
        sample_rate=sr,
        preset=str(params.layered_preset),
        preset_lock=bool(params.layered_preset_lock),
        variant_index=int(params.layered_variant_index),
        micro_variation=float(params.layered_micro_variation),
        env_curve_shape=str(params.layered_env_curve_shape),
        transient_tilt=float(params.layered_transient_tilt),
        body_tilt=float(params.layered_body_tilt),
        tail_tilt=float(params.layered_tail_tilt),
        xfade_transient_to_body_ms=float(params.layered_xfade_transient_to_body_ms),
        xfade_body_to_tail_ms=float(params.layered_xfade_body_to_tail_ms),
        xfade_curve_shape=str(params.layered_xfade_curve_shape),
        transient_hp_hz=float(params.layered_transient_hp_hz),
        transient_lp_hz=float(params.layered_transient_lp_hz),
        transient_drive=float(params.layered_transient_drive),
        transient_gain_db=float(params.layered_transient_gain_db),
        body_hp_hz=float(params.layered_body_hp_hz),
        body_lp_hz=float(params.layered_body_lp_hz),
        body_drive=float(params.layered_body_drive),
        body_gain_db=float(params.layered_body_gain_db),
        tail_hp_hz=float(params.layered_tail_hp_hz),
        tail_lp_hz=float(params.layered_tail_lp_hz),
        tail_drive=float(params.layered_tail_drive),
        tail_gain_db=float(params.layered_tail_gain_db),
        source_lock=bool(params.layered_source_lock),
        source_seed=(int(params.layered_source_seed) if params.layered_source_seed is not None else None),
        granular_preset=str(params.layered_granular_preset),
        granular_amount=float(params.layered_granular_amount),
        granular_grain_ms=float(params.layered_granular_grain_ms),
        granular_spray=float(params.layered_granular_spray),
        library_zips=tuple(Path(p) for p in params.library_zips),
        library_pitch_min=float(params.library_pitch_min),
        library_pitch_max=float(params.library_pitch_max),
        library_mix_count=int(params.library_mix_count),
        library_index_path=params.library_index_path,
        transient_ms=int(params.layered_transient_ms),
        tail_ms=int(params.layered_tail_ms),
        transient_attack_ms=float(params.layered_transient_attack_ms),
        transient_hold_ms=float(params.layered_transient_hold_ms),
        transient_decay_ms=float(params.layered_transient_decay_ms),
        body_attack_ms=float(params.layered_body_attack_ms),
        body_hold_ms=float(params.layered_body_hold_ms),
        body_decay_ms=float(params.layered_body_decay_ms),
        tail_attack_ms=float(params.layered_tail_attack_ms),
        tail_hold_ms=float(params.layered_tail_hold_ms),
        tail_decay_ms=float(params.layered_tail_decay_ms),
        duck_amount=float(params.layered_duck_amount),
        duck_release_ms=float(params.layered_duck_release_ms),
    )

    # Apply preset + common curve shape mapping.
    lp = lb._apply_preset(lp)
    lp = lb._apply_curve_shape(lp)

    # Family RNG: seed controls character. Variant index controls micro-variation.
    family_seed = lb._seed_offset(lp.seed, 1)
    variant_seed = lb._seed_offset(lp.seed, 1000 + int(lp.variant_index))
    rng_family = random.Random(family_seed)
    rng_variant = random.Random(variant_seed)

    # Apply micro-variation after preset/curve choices.
    lp = lb._apply_micro_variation(lp, rng_variant)

    # Full-length layer buffers.
    transient_full = np.zeros(n, dtype=np.float32)
    body_full = np.zeros(n, dtype=np.float32)
    tail_full = np.zeros(n, dtype=np.float32)
    sources: list[dict[str, Any]] = []

    transient_n = min(n, max(1, int(sr * (int(lp.transient_ms) / 1000.0))))
    tail_n = min(n, max(1, int(sr * (int(lp.tail_ms) / 1000.0))))

    # Source RNG base: can be pinned independently from body/variant.
    source_seed_base = lp.source_seed if lp.source_seed is not None else lp.seed

    if lp.library_zips:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            t_seed = lb._seed_offset(source_seed_base if bool(lp.source_lock) else lp.seed, 101)
            t_wav = tmp_dir / "transient.wav"
            t_prompt = f"{prompt} impact click transient"
            t_params = SampleLibParams(
                prompt=t_prompt,
                out_path=t_wav,
                seconds=float(transient_n / sr),
                seed=t_seed,
                library_zips=tuple(lp.library_zips),
                pitch_min=float(lp.library_pitch_min),
                pitch_max=float(lp.library_pitch_max),
                mix_count=int(lp.library_mix_count),
                index_path=lp.library_index_path,
                sample_rate=sr,
            )
            t_res = generate_with_samplelib(t_params)
            t_audio, _ = read_wav_mono(t_res.out_path)
            t_audio = t_audio[:transient_n]

            ta = lb._ms_to_n(lp.transient_attack_ms, sr)
            th = lb._ms_to_n(lp.transient_hold_ms, sr)
            td = lb._ms_to_n(lp.transient_decay_ms, sr)
            tenv = lb._ahd_env(
                t_audio.size,
                attack_n=ta,
                hold_n=th,
                decay_n=td,
                attack_curve=float(lp.transient_attack_curve),
                decay_curve=float(lp.transient_decay_curve),
            )
            t_audio = (t_audio * tenv).astype(np.float32, copy=False)
            t_audio = lb._fade_out(t_audio, max(1, int(0.004 * sr)))
            t_audio = lb._apply_spectral_tilt(t_audio, sr=sr, tilt=float(lp.transient_tilt), cutoff_hz=1800.0)
            t_audio = lb._apply_layer_fx(
                t_audio,
                sr=sr,
                hp_hz=float(lp.transient_hp_hz),
                lp_hz=float(lp.transient_lp_hz),
                drive=float(lp.transient_drive),
                gain_db=float(lp.transient_gain_db),
            )
            transient_full[: t_audio.size] = float(lp.transient_gain) * t_audio

            for s in t_res.sources:
                sources.append(
                    {
                        "layer": "transient",
                        "zip": s.zip_path,
                        "member": s.member,
                        "repo": s.repo,
                        "attribution_files": list(s.attribution_files),
                    }
                )

            tail_seed = lb._seed_offset(source_seed_base if bool(lp.source_lock) else lp.seed, 202)
            tail_wav = tmp_dir / "tail.wav"
            tail_prompt = f"{prompt} tail decay whoosh"
            tail_params = SampleLibParams(
                prompt=tail_prompt,
                out_path=tail_wav,
                seconds=float(tail_n / sr),
                seed=tail_seed,
                library_zips=tuple(lp.library_zips),
                pitch_min=float(lp.library_pitch_min),
                pitch_max=float(lp.library_pitch_max),
                mix_count=1,
                index_path=lp.library_index_path,
                sample_rate=sr,
            )
            tail_res = generate_with_samplelib(tail_params)
            tail_audio, _ = read_wav_mono(tail_res.out_path)
            tail_audio = tail_audio[:tail_n]

            ta = lb._ms_to_n(lp.tail_attack_ms, sr)
            th = lb._ms_to_n(lp.tail_hold_ms, sr)
            td = lb._ms_to_n(lp.tail_decay_ms, sr)
            tenv = lb._ahd_env(
                tail_audio.size,
                attack_n=ta,
                hold_n=th,
                decay_n=td,
                attack_curve=float(lp.tail_attack_curve),
                decay_curve=float(lp.tail_decay_curve),
            )
            tail_audio = (tail_audio * tenv).astype(np.float32, copy=False)
            tail_audio = lb._fade_in(tail_audio, max(1, int(0.004 * sr)))
            tail_audio = lb._apply_spectral_tilt(tail_audio, sr=sr, tilt=float(lp.tail_tilt), cutoff_hz=2400.0)
            tail_audio = lb._apply_layer_fx(
                tail_audio,
                sr=sr,
                hp_hz=float(lp.tail_hp_hz),
                lp_hz=float(lp.tail_lp_hz),
                drive=float(lp.tail_drive),
                gain_db=float(lp.tail_gain_db),
            )

            start = max(0, n - tail_audio.size)
            tail_full[start : start + tail_audio.size] = float(lp.tail_gain) * tail_audio

            for s in tail_res.sources:
                sources.append(
                    {
                        "layer": "tail",
                        "zip": s.zip_path,
                        "member": s.member,
                        "repo": s.repo,
                        "attribution_files": list(s.attribution_files),
                    }
                )
    else:
        # Fully procedural fallback (still satisfies “procedural transient/noise layers”).
        t_seed = lb._seed_offset(lp.seed, 9001)
        transient = _procedural_noise_layer(n=transient_n, seed=t_seed, peak=0.7)
        ta = lb._ms_to_n(lp.transient_attack_ms, sr)
        th = lb._ms_to_n(lp.transient_hold_ms, sr)
        td = lb._ms_to_n(lp.transient_decay_ms, sr)
        tenv = lb._ahd_env(
            transient.size,
            attack_n=ta,
            hold_n=th,
            decay_n=td,
            attack_curve=float(lp.transient_attack_curve),
            decay_curve=float(lp.transient_decay_curve),
        )
        transient = (transient * tenv).astype(np.float32, copy=False)
        transient = lb._apply_spectral_tilt(transient, sr=sr, tilt=float(lp.transient_tilt), cutoff_hz=1800.0)
        transient = lb._apply_layer_fx(
            transient,
            sr=sr,
            hp_hz=float(lp.transient_hp_hz),
            lp_hz=float(lp.transient_lp_hz),
            drive=float(lp.transient_drive),
            gain_db=float(lp.transient_gain_db),
        )
        transient_full[: transient.size] = float(lp.transient_gain) * transient

        tail_seed = lb._seed_offset(lp.seed, 9002)
        tail = _procedural_noise_layer(n=tail_n, seed=tail_seed, peak=0.55)
        # Make tail smoother and place at end.
        tail = lb._one_pole_lowpass(tail, sr=sr, cutoff_hz=3500.0)
        ta = lb._ms_to_n(lp.tail_attack_ms, sr)
        th = lb._ms_to_n(lp.tail_hold_ms, sr)
        td = lb._ms_to_n(lp.tail_decay_ms, sr)
        tenv = lb._ahd_env(
            tail.size,
            attack_n=ta,
            hold_n=th,
            decay_n=td,
            attack_curve=float(lp.tail_attack_curve),
            decay_curve=float(lp.tail_decay_curve),
        )
        tail = (tail * tenv).astype(np.float32, copy=False)
        tail = lb._apply_spectral_tilt(tail, sr=sr, tilt=float(lp.tail_tilt), cutoff_hz=2400.0)
        tail = lb._apply_layer_fx(
            tail,
            sr=sr,
            hp_hz=float(lp.tail_hp_hz),
            lp_hz=float(lp.tail_lp_hz),
            drive=float(lp.tail_drive),
            gain_db=float(lp.tail_gain_db),
        )
        start = max(0, n - tail.size)
        tail_full[start : start + tail.size] = float(lp.tail_gain) * tail

    # Body from AI base across full duration.
    ba = lb._ms_to_n(lp.body_attack_ms, sr)
    bd = lb._ms_to_n(lp.body_decay_ms, sr)
    if float(lp.body_hold_ms) <= 0.0:
        bh = max(0, body.size - ba - bd)
    else:
        bh = lb._ms_to_n(lp.body_hold_ms, sr)
    benv = lb._ahd_env(
        body.size,
        attack_n=ba,
        hold_n=bh,
        decay_n=bd,
        attack_curve=float(lp.body_attack_curve),
        decay_curve=float(lp.body_decay_curve),
    )
    body = (body * benv).astype(np.float32, copy=False)

    # Optional granular texture mixed into body (procedural, deterministic).
    gran_amount = float(np.clip(float(lp.granular_amount), 0.0, 1.0))
    gran_preset = (lp.granular_preset or "off").strip().lower()
    if gran_preset == "auto":
        gran_preset = lb._auto_granular_preset(prompt, str(lp.preset))
    if gran_amount > 1e-4 and gran_preset != "off":
        gran_seed = lb._seed_offset(lp.seed, 7000 + int(lp.variant_index) * 37)
        texture = lb._synthesize_granular_texture(
            seconds=float(seconds),
            sr=sr,
            seed=gran_seed,
            preset=gran_preset,
            grain_ms=float(lp.granular_grain_ms),
            spray=float(lp.granular_spray),
        )
        texture = texture[: body.size]
        texture = (texture * benv[: texture.size]).astype(np.float32, copy=False)
        body = (body + gran_amount * texture).astype(np.float32, copy=False)

    body = lb._apply_spectral_tilt(body, sr=sr, tilt=float(lp.body_tilt), cutoff_hz=1600.0)
    body = lb._apply_layer_fx(
        body,
        sr=sr,
        hp_hz=float(lp.body_hp_hz),
        lp_hz=float(lp.body_lp_hz),
        drive=float(lp.body_drive),
        gain_db=float(lp.body_gain_db),
    )

    duck_gain = lb._sidechain_duck_gain(
        transient_full,
        sr=sr,
        amount=float(lp.duck_amount),
        release_ms=float(lp.duck_release_ms),
    )
    body = (body * duck_gain[: body.size]).astype(np.float32, copy=False)
    body_full[: body.size] = float(lp.body_gain) * body

    tb_n = lb._ms_to_n(float(lp.xfade_transient_to_body_ms), sr)
    if tb_n > 1:
        tb_n = min(tb_n, n)
        tb_out, tb_in = lb._xfade_windows(tb_n, shape=str(lp.xfade_curve_shape))
        transient_full[:tb_n] *= tb_out
        body_full[:tb_n] *= tb_in

    bt_n = lb._ms_to_n(float(lp.xfade_body_to_tail_ms), sr)
    if bt_n > 1 and tail_full.any():
        start = int(np.argmax(np.abs(tail_full) > 0.0))
        if start <= 0:
            start = max(0, n - tail_n)
        bt_n = min(bt_n, n - start)
        if bt_n > 1:
            bt_out, bt_in = lb._xfade_windows(bt_n, shape=str(lp.xfade_curve_shape))
            body_full[start : start + bt_n] *= bt_out
            body_full[start + bt_n :] *= 0.0
            tail_full[start : start + bt_n] *= bt_in

    x = (transient_full + body_full + tail_full).astype(np.float32, copy=False)
    x = lb._soft_clip(x, drive=0.10)
    x = lb._normalize_peak(x, peak=0.98)
    x = np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)

    credits_extra: dict[str, Any] = {
        "hybrid": {
            "base_engine": str(base_engine),
            "base": dict(base_credits),
        },
        "layered": {
            "preset": str(lp.preset),
            "preset_lock": bool(lp.preset_lock),
            "variant_index": int(lp.variant_index),
            "micro_variation": float(lp.micro_variation),
            "source_lock": bool(lp.source_lock),
            "source_seed": (int(source_seed_base) if source_seed_base is not None else None),
            "granular": {
                "preset": str(gran_preset),
                "amount": float(gran_amount),
                "grain_ms": float(lp.granular_grain_ms),
                "spray": float(lp.granular_spray),
            },
            "transient_ms": int(lp.transient_ms),
            "tail_ms": int(lp.tail_ms),
            "xfade_ms": {
                "transient_to_body": float(lp.xfade_transient_to_body_ms),
                "body_to_tail": float(lp.xfade_body_to_tail_ms),
            },
            "xfade_curve_shape": str(lp.xfade_curve_shape),
            "gains": {
                "transient": float(lp.transient_gain),
                "body": float(lp.body_gain),
                "tail": float(lp.tail_gain),
            },
            "envelopes_ms": {
                "transient": {
                    "attack": float(lp.transient_attack_ms),
                    "hold": float(lp.transient_hold_ms),
                    "decay": float(lp.transient_decay_ms),
                },
                "body": {
                    "attack": float(lp.body_attack_ms),
                    "hold": float(lp.body_hold_ms),
                    "decay": float(lp.body_decay_ms),
                },
                "tail": {
                    "attack": float(lp.tail_attack_ms),
                    "hold": float(lp.tail_hold_ms),
                    "decay": float(lp.tail_decay_ms),
                },
            },
            "env_curve_shape": str(lp.env_curve_shape),
            "tilt": {
                "transient": float(lp.transient_tilt),
                "body": float(lp.body_tilt),
                "tail": float(lp.tail_tilt),
            },
            "layer_fx": {
                "transient": {
                    "hp_hz": float(lp.transient_hp_hz),
                    "lp_hz": float(lp.transient_lp_hz),
                    "drive": float(lp.transient_drive),
                    "gain_db": float(lp.transient_gain_db),
                },
                "body": {
                    "hp_hz": float(lp.body_hp_hz),
                    "lp_hz": float(lp.body_lp_hz),
                    "drive": float(lp.body_drive),
                    "gain_db": float(lp.body_gain_db),
                },
                "tail": {
                    "hp_hz": float(lp.tail_hp_hz),
                    "lp_hz": float(lp.tail_lp_hz),
                    "drive": float(lp.tail_drive),
                    "gain_db": float(lp.tail_gain_db),
                },
            },
            "interaction": {
                "duck_amount": float(lp.duck_amount),
                "duck_release_ms": float(lp.duck_release_ms),
            },
            "body_source": str(base_engine),
        },
    }

    return HybridResult(audio=x, sample_rate=sr, sources=tuple(sources), credits_extra=credits_extra)

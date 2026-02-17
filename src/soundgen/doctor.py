from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    name: str
    details: str
    fix: str | None = None


def _check_ffmpeg() -> CheckResult:
    from .io_utils import find_ffmpeg

    try:
        p = find_ffmpeg()
        return CheckResult(True, "ffmpeg", f"OK: {p}")
    except Exception as e:
        return CheckResult(
            False,
            "ffmpeg",
            f"Missing: {e}",
            fix="Install ffmpeg and put it on PATH (Windows: `winget install Gyan.FFmpeg`).",
        )


def _check_rfxgen() -> CheckResult:
    found = shutil.which("rfxgen") or shutil.which("rfxgen.exe")
    local = Path("tools") / "rfxgen" / "rfxgen.exe"
    if found:
        return CheckResult(True, "rfxgen", f"OK: {found}")
    if local.exists():
        return CheckResult(True, "rfxgen", f"OK: {local}")
    return CheckResult(
        False,
        "rfxgen",
        "Missing: rfxgen.exe not found on PATH or at tools/rfxgen/rfxgen.exe",
        fix="Run `./scripts/get_rfxgen.ps1` or download rfxgen.exe and place it in tools/rfxgen/.",
    )


def _check_torch() -> CheckResult:
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        return CheckResult(True, "torch", f"OK: torch={torch.__version__} cuda_available={cuda}")
    except Exception as e:
        return CheckResult(
            False,
            "torch",
            f"Missing/broken: {e}",
            fix="Install dependencies with `pip install -r requirements.txt`.",
        )


def _get_hf_token(explicit: str | None) -> str | None:
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    # HF uses HF_TOKEN, and many users set HUGGINGFACE_HUB_TOKEN.
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        v = os.getenv(k)
        if v and v.strip():
            return v.strip()
    return None


def _check_hf_token(explicit: str | None) -> CheckResult:
    tok = _get_hf_token(explicit)
    if tok:
        return CheckResult(True, "huggingface token", "OK: token is set (not printed)")
    return CheckResult(
        False,
        "huggingface token",
        "Missing: no token found in --hf-token, HF_TOKEN, or HUGGINGFACE_HUB_TOKEN",
        fix="If you need gated models: set HF_TOKEN/HUGGINGFACE_HUB_TOKEN or run `huggingface-cli login`.",
    )


def _check_hf_model_access(model_id: str, token: str | None) -> CheckResult:
    try:
        from huggingface_hub import model_info
        from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

        try:
            info = model_info(model_id, token=token)  # type: ignore[call-arg]
        except TypeError:
            info = model_info(model_id, use_auth_token=token)  # type: ignore[call-arg]

        sha = getattr(info, "sha", None)
        return CheckResult(True, f"hf model access ({model_id})", f"OK: reachable sha={sha}")
    except Exception as e:
        msg = str(e)
        low = msg.lower()
        if "gated" in low or "accept" in low or "401" in low or "403" in low:
            return CheckResult(
                False,
                f"hf model access ({model_id})",
                f"Blocked (gated): {msg}",
                fix=(
                    "Accept the model terms on the Hugging Face model page, then authenticate "
                    "(huggingface-cli login or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN)."
                ),
            )
        return CheckResult(
            False,
            f"hf model access ({model_id})",
            f"Failed: {msg}",
            fix="Check network access, model id, and (if gated) your HF token/terms acceptance.",
        )


def run_doctor(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Soundgen environment doctor (plug-and-play checks).")
    p.add_argument("--hf-token", default=None, help="Optional HF token to test gated model access.")
    p.add_argument(
        "--check-stable-audio",
        action="store_true",
        help="Also checks Hugging Face access for the Stable Audio Open model (requires network).",
    )
    args = p.parse_args(argv)

    print(f"Python: {sys.version.split()[0]}")

    results: list[CheckResult] = []
    results.append(_check_ffmpeg())
    results.append(_check_rfxgen())
    results.append(_check_torch())

    tok_res = _check_hf_token(args.hf_token)
    results.append(tok_res)

    if args.check_stable_audio:
        token = _get_hf_token(args.hf_token)
        results.append(_check_hf_model_access("stabilityai/stable-audio-open-1.0", token))

    ok = True
    for r in results:
        status = "OK" if r.ok else "FAIL"
        print(f"[{status}] {r.name}: {r.details}")
        if not r.ok and r.fix:
            print(f"       Fix: {r.fix}")
            ok = False

    return 0 if ok else 2


def main() -> None:
    raise SystemExit(run_doctor())


if __name__ == "__main__":
    main()

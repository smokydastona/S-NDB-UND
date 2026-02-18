from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


class VariantDict(TypedDict, total=False):
    id: str
    seed: int | None
    seconds: float
    rms_dbfs: float
    locked: bool
    select: bool
    audio_key: str
    edited_audio_key: str | None
    wav_path: str
    edited_wav_path: str | None
    meta: dict[str, Any]


@dataclass
class Variant:
    id: str
    seed: int | None
    seconds: float
    rms_dbfs: float
    locked: bool = False
    select: bool = False
    audio_key: str = ""
    edited_audio_key: str | None = None
    wav_path: str = ""
    edited_wav_path: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Variant":
        return cls(
            id=str(d.get("id") or "").strip(),
            seed=(int(d["seed"]) if d.get("seed") is not None else None),
            seconds=float(d.get("seconds") or 0.0),
            rms_dbfs=float(d.get("rms_dbfs") if d.get("rms_dbfs") is not None else float("-inf")),
            locked=bool(d.get("locked", False)),
            select=bool(d.get("select", False)),
            audio_key=str(d.get("audio_key") or ""),
            edited_audio_key=(str(d.get("edited_audio_key")) if d.get("edited_audio_key") else None),
            wav_path=str(d.get("wav_path") or ""),
            edited_wav_path=(str(d.get("edited_wav_path")) if d.get("edited_wav_path") else None),
            meta=(dict(d.get("meta")) if isinstance(d.get("meta"), dict) else {}),
        )

    def to_dict(self) -> VariantDict:
        return {
            "id": self.id,
            "seed": self.seed,
            "seconds": float(self.seconds),
            "rms_dbfs": float(self.rms_dbfs),
            "locked": bool(self.locked),
            "select": bool(self.select),
            "audio_key": str(self.audio_key or ""),
            "edited_audio_key": (str(self.edited_audio_key) if self.edited_audio_key else None),
            "wav_path": str(self.wav_path),
            "edited_wav_path": (str(self.edited_wav_path) if self.edited_wav_path else None),
            "meta": dict(self.meta),
        }


def normalize_variants_state(state: list[dict[str, Any]] | None) -> list[VariantDict]:
    out: list[VariantDict] = []
    for v in state or []:
        if not isinstance(v, dict):
            continue
        vv = Variant.from_dict(v)
        if not vv.id:
            continue
        out.append(vv.to_dict())
    return out

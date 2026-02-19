from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .ai_index import build_or_update_index, top_matching_files


@dataclass(frozen=True)
class ContextDoc:
    rel_path: str
    kind: str  # e.g. 'md', 'json', 'py'
    weight: float = 1.0


_CURATED_DOCS: list[ContextDoc] = [
    ContextDoc("README.md", "md", 2.0),
    ContextDoc("example_manifest.json", "json", 2.0),
    ContextDoc("docs/one_stop_shop.md", "md", 1.4),
    ContextDoc("docs/project_system.md", "md", 1.6),
    ContextDoc("docs/v4_spec.md", "md", 1.6),
    ContextDoc("docs/pro_mode_spec.md", "md", 1.2),
    ContextDoc("docs/fx_chains_v2.md", "md", 1.2),
    ContextDoc("docs/sfx_presets_v2_schema.md", "md", 1.2),
]


_RAG_DOC_GLOBS: list[str] = [
    "docs/**/*.md",
]


_RAG_CODE_RELS: list[str] = [
    # Core entrypoints / pipelines
    "src/soundgen/generate.py",
    "src/soundgen/batch.py",
    "src/soundgen/web_control_panel.py",
    "src/soundgen/web.py",
    "src/soundgen/desktop.py",
    "src/soundgen/app.py",

    # Export, formats, QA/post
    "src/soundgen/minecraft.py",
    "src/soundgen/io_utils.py",
    "src/soundgen/postprocess.py",
    "src/soundgen/qa.py",

    # Project + manifest schemas
    "src/soundgen/project.py",
    "src/soundgen/manifest.py",

    # Engines registry
    "src/soundgen/engine_registry.py",
]


def _is_reasonable_context_file(rel_path: str) -> bool:
    rp = str(rel_path).replace("\\", "/").lstrip("/")
    # Avoid including generated / huge / irrelevant directories.
    bad_prefixes = (
        "outputs/",
        "resourcepack/",
        "pre_gen_sound/",
        "tools/",
        "library/",
        "soundpack_v1/",
        ".git/",
        "electron/",
    )
    if rp.startswith(bad_prefixes):
        return False
    return True


def _iter_rag_candidates(root: Path) -> list[tuple[str, Path, float]]:
    """Return list of (rel_path, abs_path, weight)"""

    items: list[tuple[str, Path, float]] = []

    # Docs (broad scan).
    for g in _RAG_DOC_GLOBS:
        for p in root.glob(g):
            try:
                rel = str(p.relative_to(root)).replace("\\", "/")
            except Exception:
                continue
            if not _is_reasonable_context_file(rel):
                continue
            items.append((rel, p, 1.0))

    # Key code modules (explicit list).
    for rel in _RAG_CODE_RELS:
        p = root / rel
        if p.exists() and p.is_file():
            rp = str(rel).replace("\\", "/")
            if _is_reasonable_context_file(rp):
                items.append((rp, p, 1.15))

    # Deduplicate by rel path.
    out: list[tuple[str, Path, float]] = []
    seen: set[str] = set()
    for rel, p, w in items:
        if rel in seen:
            continue
        seen.add(rel)
        out.append((rel, p, w))
    return out


_APP_FACTS = (
    "SÖNDBÖUND app facts (authoritative):\n"
    "- Project type: Python (src-layout) prompt→SFX generator for Minecraft-ready audio packs.\n"
    "- Audio internal format between stages: mono 1D float32 in [-1, 1].\n"
    "- Core flow: prompt/manifest/docs → engine generates mono WAV → optional post-process/polish/QA → export (WAV/OGG/MP3/FLAC) → optional Minecraft pack (sounds.json + lang).\n"
    "- CLI entry: python -m soundgen.generate\n"
    "- UI entry: python -m soundgen.web (Control Panel UI is default; legacy UI via SOUNDGEN_WEB_UI=legacy).\n"
    "- Desktop window: python -m soundgen.desktop (prefers WebView2 / edgechromium).\n"
    "- Batch: python -m soundgen.batch --manifest <json/csv>\n"
    "- Minecraft export conventions: sounds.json event key is <event> (full id <namespace>:<event>); variants are object entries with name/weight/volume/pitch; subtitles go to assets/<ns>/lang/en_us.json.\n"
    "- OGG export requires ffmpeg.\n"
)


def _repo_roots() -> list[Path]:
    roots: list[Path] = []

    # Current working directory (dev runs).
    try:
        roots.append(Path.cwd())
    except Exception:
        pass

    # src/soundgen/..../repo
    try:
        roots.append(Path(__file__).resolve().parents[2])
    except Exception:
        pass

    # Packaged exe dir (PyInstaller)
    try:
        exe_dir = Path(sys.executable).resolve().parent
        roots.append(exe_dir)
    except Exception:
        pass

    # Deduplicate while preserving order.
    out: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        key = str(r.resolve()) if r else ""
        if key and key not in seen:
            out.append(r)
            seen.add(key)
    return out


def _read_text_best_effort(p: Path, max_chars: int) -> str | None:
    try:
        if not p.exists() or not p.is_file():
            return None
        b = p.read_bytes()
        # Try utf-8 first, fall back.
        try:
            t = b.decode("utf-8")
        except Exception:
            t = b.decode("utf-8", errors="replace")
        t = t.replace("\r\n", "\n")
        if len(t) > max_chars:
            return t[:max_chars] + "\n…(truncated)…\n"
        return t
    except Exception:
        return None


_WORD_RE = re.compile(r"[a-z0-9_\-\.]{3,}")


def _keywords(text: str, max_terms: int = 12) -> list[str]:
    raw = str(text or "").strip().lower()
    if not raw:
        return []
    terms = _WORD_RE.findall(raw)
    # De-dupe while keeping order.
    out: list[str] = []
    seen: set[str] = set()
    for t in terms:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_terms:
            break
    return out


def _score_text(text: str, terms: Iterable[str]) -> float:
    t = text.lower()
    score = 0.0
    for term in terms:
        if not term:
            continue
        score += float(t.count(term))
    return score


def _excerpt_around_terms(text: str, terms: list[str], max_chars: int) -> str:
    if not text:
        return ""
    if not terms:
        return text[:max_chars] + ("\n…(truncated)…\n" if len(text) > max_chars else "")

    lo = text.lower()
    hits: list[int] = []
    for term in terms:
        i = lo.find(term)
        if i >= 0:
            hits.append(i)
    if not hits:
        return text[:max_chars] + ("\n…(truncated)…\n" if len(text) > max_chars else "")

    center = sorted(hits)[0]
    start = max(0, center - int(max_chars * 0.35))
    end = min(len(text), start + max_chars)
    excerpt = text[start:end]
    if start > 0:
        excerpt = "…" + excerpt
    if end < len(text):
        excerpt = excerpt + "…"
    return excerpt


def build_app_context(
    *,
    query: str,
    max_chars: int = 14000,
    per_doc_max_chars: int = 3500,
) -> str:
    """Build a compact, query-aware context pack from local docs.

    This is designed for chat assistants: it favors actionable specs, schemas, and examples.
    It never raises on missing files.
    """

    # Allow disabling context injection entirely.
    if str(os.environ.get("SOUNDGEN_AI_CONTEXT", "1")).strip().lower() in {"0", "false", "off", "no"}:
        return ""

    terms = _keywords(query)

    roots = _repo_roots()

    # Resolve curated docs.
    candidates: list[tuple[float, str, str]] = []  # (score, rel_path, excerpt)
    for doc in _CURATED_DOCS:
        best_text = None
        best_root = None
        for root in roots:
            p = (root / doc.rel_path)
            txt = _read_text_best_effort(p, max_chars=per_doc_max_chars * 3)
            if txt is None:
                continue
            best_text = txt
            best_root = root
            break
        if best_text is None:
            continue

        s = _score_text(best_text, terms) * float(doc.weight)
        # Always include README + manifest even if query doesn't hit.
        if doc.rel_path in {"README.md", "example_manifest.json"}:
            s += 0.25
        excerpt = _excerpt_around_terms(best_text, terms, max_chars=per_doc_max_chars)
        candidates.append((s, doc.rel_path, excerpt))

    # Local RAG: prefer using the on-disk index (fast) to pick top files to excerpt.
    scan_enabled = str(os.environ.get("SOUNDGEN_AI_CONTEXT_SCAN", "1")).strip().lower() not in {"0", "false", "off", "no"}
    if scan_enabled and terms:
        scan_root = None
        for r in roots:
            try:
                if (r / "src" / "soundgen").exists() or (r / "docs").exists():
                    scan_root = r
                    break
            except Exception:
                continue

        if scan_root is not None:
            idx_enabled = str(os.environ.get("SOUNDGEN_AI_INDEX", "1")).strip().lower() not in {"0", "false", "off", "no"}
            ranked_rels: list[str] = []
            if idx_enabled:
                try:
                    _ = build_or_update_index(force=False)
                    ranked_rels = top_matching_files(query=query, k=14)
                except Exception:
                    ranked_rels = []

            # Fallback: bounded scan order (less accurate, but no index needed).
            if not ranked_rels:
                rag_items = _iter_rag_candidates(scan_root)
                ranked_rels = [rel for rel, _, _ in rag_items[:40]]

            per_file_read = max(9000, int(per_doc_max_chars * 3))
            for rel in ranked_rels:
                p = scan_root / rel
                txt = _read_text_best_effort(p, max_chars=per_file_read)
                if not txt:
                    continue
                s = _score_text(txt, terms)
                if s <= 0.0:
                    continue
                excerpt = _excerpt_around_terms(txt, terms, max_chars=per_doc_max_chars)
                candidates.append((float(s), rel, excerpt))

    # Pick top docs by score.
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Always include at least README + manifest if available.
    picked: list[tuple[float, str, str]] = []
    seen_paths: set[str] = set()
    for must in ("README.md", "example_manifest.json"):
        for s, rel, ex in candidates:
            if rel == must and rel not in seen_paths:
                picked.append((s, rel, ex))
                seen_paths.add(rel)
                break

    for s, rel, ex in candidates:
        if rel in seen_paths:
            continue
        picked.append((s, rel, ex))
        seen_paths.add(rel)
        if len(picked) >= 10:
            break

    parts: list[str] = []
    parts.append(_APP_FACTS.strip())

    parts.append("\nKey modules (orientation):\n" + "\n".join(
        [
            "- src/soundgen/generate.py: CLI generation pipeline and flags.",
            "- src/soundgen/web_control_panel.py: main Gradio Control Panel UI.",
            "- src/soundgen/minecraft.py: pack export (.ogg + sounds.json + lang).",
            "- src/soundgen/postprocess.py and src/soundgen/qa.py: post-processing + QA metrics.",
            "- src/soundgen/batch.py: manifest-driven batch generation and pack zipping.",
        ]
    ))

    if picked:
        parts.append("\nSelected docs/spec excerpts (query-aware):")
        for _, rel, ex in picked:
            parts.append(f"\n--- FILE: {rel} ---\n{ex.strip()}\n")

    out = "\n".join(parts).strip() + "\n"

    if len(out) > max_chars:
        out = out[:max_chars] + "\n…(context truncated)…\n"
    return out

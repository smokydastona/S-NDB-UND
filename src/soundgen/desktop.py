from __future__ import annotations

import argparse
import datetime as _dt
import os
import platform
import socket
import sys
import traceback
from pathlib import Path

from .web import build_demo


def _desktop_log_path() -> Path:
    if sys.platform == "win32":
        la = str(os.environ.get("LOCALAPPDATA") or "").strip()
        base = (Path(la) if la else (Path.home() / "AppData" / "Local")) / "SÖNDBÖUND"
    else:
        base = Path.home() / ".söndböund"
    return base / "desktop.log"


def _write_desktop_log(text: str) -> None:
    try:
        p = _desktop_log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now().isoformat(timespec="seconds")
        with p.open("a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {text}\n")
    except Exception:
        return


def _hide_console_window_windows_best_effort() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE
    except Exception:
        return


def _safe_dist_version(dist_name: str) -> str | None:
    try:
        from importlib.metadata import version  # pyright: ignore[reportMissingImports]

        v = str(version(dist_name)).strip()
        return v or None
    except Exception:
        return None


def _safe_find_spec_origin(module_name: str) -> str | None:
    try:
        import importlib.util

        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return None
        origin = getattr(spec, "origin", None)
        if origin:
            return str(origin)
        return None
    except Exception:
        return None


def _sha256_file(path: Path, limit_bytes: int | None = None) -> str | None:
    try:
        import hashlib

        h = hashlib.sha256()
        read_total = 0
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
                read_total += len(chunk)
                if limit_bytes is not None and read_total >= int(limit_bytes):
                    break
        return h.hexdigest()
    except Exception:
        return None


def _log_desktop_diagnostics(local_url: str | None) -> None:
    try:
        _write_desktop_log("desktop_diag_begin")
        _write_desktop_log(f"python={sys.version.replace(chr(10), ' ').strip()}")
        _write_desktop_log(f"platform={platform.platform()}")
        _write_desktop_log(f"exe={sys.executable}")
        _write_desktop_log(f"cwd={os.getcwd()}")
        _write_desktop_log(f"frozen={bool(getattr(sys, 'frozen', False))}")
        _write_desktop_log(f"_MEIPASS={getattr(sys, '_MEIPASS', None)!r}")
        if local_url:
            _write_desktop_log(f"local_url={local_url}")

        for env_key in ("SOUNDGEN_DESKTOP_GUI", "SOUNDGEN_HIDE_CONSOLE_ON_READY"):
            _write_desktop_log(f"env.{env_key}={os.environ.get(env_key)!r}")

        # Distribution (pip) versions (doesn't import the modules).
        for dist in ("pywebview", "pythonnet", "clr-loader", "gradio"):
            _write_desktop_log(f"dist.{dist}={_safe_dist_version(dist)!r}")

        # Module locations.
        for mod in ("webview", "pythonnet", "clr_loader", "clr"):
            _write_desktop_log(f"mod.{mod}.origin={_safe_find_spec_origin(mod)!r}")

        # Candidate runtime DLL paths.
        candidates: list[Path] = []

        try:
            import importlib.util

            spec = importlib.util.find_spec("pythonnet")
            if spec is not None and getattr(spec, "origin", None):
                pkg_dir = Path(str(spec.origin)).parent
                candidates.append(pkg_dir / "runtime" / "Python.Runtime.dll")
        except Exception:
            pass

        try:
            exe_dir = Path(sys.executable).resolve().parent
            candidates.append(exe_dir / "_internal" / "pythonnet" / "runtime" / "Python.Runtime.dll")
            candidates.append(exe_dir / "pythonnet" / "runtime" / "Python.Runtime.dll")
        except Exception:
            pass

        try:
            meipass = getattr(sys, "_MEIPASS", None)
            if meipass:
                base = Path(str(meipass))
                candidates.append(base / "pythonnet" / "runtime" / "Python.Runtime.dll")
        except Exception:
            pass

        seen: set[str] = set()
        for c in candidates:
            key = str(c)
            if key in seen:
                continue
            seen.add(key)
            exists = False
            size = None
            sha = None
            try:
                exists = c.exists()
                if exists:
                    size = int(c.stat().st_size)
                    sha = _sha256_file(c)
            except Exception:
                pass
            _write_desktop_log(f"pythonnet_dll path={key!s} exists={exists} size={size!r} sha256={sha!r}")

        _write_desktop_log("desktop_diag_end")
    except Exception:
        return


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def run_desktop(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the SÖNDBÖUND UI in a native desktop window (embedded webview).")
    p.add_argument("--host", default="127.0.0.1", help="Bind host for the local Gradio server.")
    p.add_argument(
        "--port",
        type=int,
        default=0,
        help="Bind port for the local Gradio server (0 = auto).",
    )
    args = p.parse_args(argv)

    try:
        import webview  # pyright: ignore[reportMissingImports]
    except Exception as e:
        msg = str(e).strip() or e.__class__.__name__
        raise SystemExit(
            "pywebview is required for the desktop window. "
            "Install it with: pip install pywebview\n\n"
            f"Import error: {msg}"
        )

    host = str(args.host).strip() or "127.0.0.1"
    port = int(args.port)
    if port <= 0:
        port = _pick_free_port(host)

    demo = build_demo()

    launch_kwargs = {
        "server_name": host,
        "server_port": port,
        "inbrowser": False,
        "prevent_thread_lock": True,
    }

    local_url: str | None = None
    try:
        res = demo.launch(**launch_kwargs, quiet=True)
    except TypeError:
        # Older Gradio versions might not support `quiet`.
        res = demo.launch(**launch_kwargs)

    if isinstance(res, tuple) and len(res) >= 2:
        # Gradio commonly returns (app, local_url, share_url)
        try:
            local_url = str(res[1])
        except Exception:
            local_url = None
    elif isinstance(res, str):
        local_url = res

    if not local_url:
        local_url = f"http://{host}:{port}"

    _log_desktop_diagnostics(local_url)

    webview.create_window("SÖNDBÖUND", local_url, width=1200, height=800)

    # On Windows, pywebview may default to the WinForms backend, which depends on
    # pythonnet/.NET. Packaged builds can break if the runtime/DLLs are mismatched.
    # Prefer Edge Chromium (WebView2) first, then fall back to mshtml.
    preferred_gui = str(os.environ.get("SOUNDGEN_DESKTOP_GUI", "")).strip().lower() or None
    raw_candidates: list[str | None] = [preferred_gui, "edgechromium", "mshtml", None]
    gui_candidates: list[str | None] = []
    for gui in raw_candidates:
        if gui in {"", "none"}:
            gui = None
        if gui not in gui_candidates:
            gui_candidates.append(gui)
    last_error: Exception | None = None

    hide_on_ready = str(os.environ.get("SOUNDGEN_HIDE_CONSOLE_ON_READY", "")).strip() == "1"
    if hide_on_ready:
        _write_desktop_log("console_hide=on_ready")

    for gui in gui_candidates:
        try:
            _write_desktop_log(f"webview_start gui={gui!r} url={local_url}")

            # Hide console only after the GUI backend is initialized.
            func = _hide_console_window_windows_best_effort if hide_on_ready else None

            if gui is None:
                if func is None:
                    webview.start()
                else:
                    try:
                        webview.start(func=func)
                    except TypeError:
                        # Older pywebview versions may not support func=.
                        webview.start()
            else:
                if func is None:
                    webview.start(gui=gui)
                else:
                    try:
                        webview.start(gui=gui, func=func)
                    except TypeError:
                        webview.start(gui=gui)
            last_error = None
            break
        except Exception as e:
            _write_desktop_log(
                f"webview_failed gui={gui!r}: {str(e).strip() or e.__class__.__name__}\n{traceback.format_exc()}"
            )
            last_error = e
            continue

    if last_error is not None:
        raise last_error
    return 0


def main() -> None:
    raise SystemExit(run_desktop())


if __name__ == "__main__":
    main()

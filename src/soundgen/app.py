from __future__ import annotations

import argparse
import sys


def _hide_console_window_windows() -> None:
    """Hide the console window when launching GUI modes on Windows.

    We keep the EXE as a console subsystem build so CLI usage works.
    When the user double-clicks the EXE (defaulting to desktop mode),
    this hides the spawned console for a more app-like experience.
    """

    if sys.platform != "win32":
        return

    try:
        import ctypes

        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE
    except Exception:
        # Best-effort only; never fail app startup because of this.
        return


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="soundgen",
        description=(
            "Sound Generator (single app).\n"
            "- No args: opens the desktop UI window.\n"
            "- 'generate': CLI generation (same flags as soundgen.generate).\n"
            "- 'web': Gradio UI in your browser.\n"
            "- 'desktop': Gradio UI in an embedded desktop window."
        ),
    )

    sub = p.add_subparsers(dest="command")

    sub.add_parser("generate", help="Generate sounds via CLI (prompt -> wav/ogg).")
    sub.add_parser("web", help="Run the Gradio UI in your browser.")
    sub.add_parser("desktop", help="Run the UI in an embedded desktop window.")

    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    # Default: desktop UI.
    if not argv:
        _hide_console_window_windows()
        from .desktop import run_desktop

        return int(run_desktop([]))

    ns, rest = parser.parse_known_args(argv)
    cmd = (ns.command or "").strip().lower()

    if cmd == "generate":
        from .generate import main as generate_main

        return int(generate_main(rest))

    if cmd == "web":
        _hide_console_window_windows()
        from .web import main as web_main

        web_main()
        return 0

    if cmd == "desktop":
        _hide_console_window_windows()
        from .desktop import run_desktop

        return int(run_desktop(rest))

    # Unknown or missing subcommand.
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

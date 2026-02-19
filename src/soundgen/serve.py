from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _local_url_from_launch_result(res: object, *, host: str, port: int) -> str:
    if isinstance(res, tuple) and len(res) >= 2:
        try:
            u = str(res[1])
            if u:
                return u
        except Exception:
            pass
    if isinstance(res, str) and res.strip():
        return str(res).strip()
    return f"http://{host}:{port}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Start the SÖNDBÖUND Gradio UI as a local web server without opening a browser. "
            "Useful for Electron shells and other wrappers."
        )
    )
    p.add_argument("--host", default="127.0.0.1", help="Bind host for the local Gradio server.")
    p.add_argument(
        "--port",
        type=int,
        default=0,
        help="Bind port for the local Gradio server (0 = auto).",
    )
    p.add_argument(
        "--ui",
        choices=["control", "legacy"],
        default=None,
        help="Select UI variant (default uses SOUNDGEN_WEB_UI env or control-panel).",
    )
    p.add_argument(
        "--print-json",
        action="store_true",
        help="Print a single JSON line with {url, host, port} for machine parsing.",
    )
    args = p.parse_args(list(argv or []))

    host = str(args.host).strip() or "127.0.0.1"
    port = int(args.port)
    if port <= 0:
        port = _pick_free_port(host)

    if args.ui == "legacy":
        os.environ["SOUNDGEN_WEB_UI"] = "legacy"
    elif args.ui == "control":
        os.environ.pop("SOUNDGEN_WEB_UI", None)

    from .web import build_demo

    demo = build_demo()

    launch_kwargs = {
        "server_name": host,
        "server_port": port,
        "inbrowser": False,
        "prevent_thread_lock": True,
    }

    try:
        res = demo.launch(**launch_kwargs, quiet=True)
    except TypeError:
        # Older Gradio versions might not support `quiet`.
        res = demo.launch(**launch_kwargs)

    url = _local_url_from_launch_result(res, host=host, port=port)

    if bool(args.print_json):
        print(json.dumps({"url": url, "host": host, "port": port}, ensure_ascii=False), flush=True)
    else:
        # Easy to parse from Electron stdout.
        print(f"SOUNDGEN_URL={url}", flush=True)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

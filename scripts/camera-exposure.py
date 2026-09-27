#!/usr/bin/env python3
"""Sway shortcut client for Buttercup camera exposure.

    camera-exposure.py up|down [--step LINES]   # manual exposure step (viewer . and / keys)
    camera-exposure.py toggle                   # auto <-> manual (manual holds current exposure)
    camera-exposure.py auto|manual|status

Each change claims a short camera-control lease on the viewer's control socket,
queues one exposure action and releases the lease. Feedback appears on the
attention-style eye OSD (scripts/eye-focus-osd.py).
"""
import argparse
import json
import os
import socket
import subprocess
import sys

SOCKET = "/tmp/buttercup-eye-control.sock"
OSD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eye-focus-osd.py")
OWNER = "sway-camera-exposure"
LEASE_MS = 3000
RANGE = (32, 8192)


def request(path, line):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(3)
        try:
            connection.connect(path)
        except (FileNotFoundError, ConnectionRefusedError) as error:
            raise RuntimeError(f"Buttercup viewer is not running at {path}") from error
        connection.sendall(f"{line}\n".encode())
        connection.shutdown(socket.SHUT_WR)
        data = bytearray()
        while len(data) <= 1 << 20:
            part = connection.recv(65536)
            if not part:
                break
            data.extend(part)
    reply = json.loads(data)
    if not reply.get("ok"):
        raise RuntimeError(reply.get("error") or f"viewer rejected: {line}")
    return reply


def osd(**message):
    try:
        import importlib.util
        sys.dont_write_bytecode = True  # keep generated files out of the source tree
        spec = importlib.util.spec_from_file_location("eye_focus_osd", OSD)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.show(dict(kind="exposure", eyebrow="EYE EXPOSURE", range=RANGE,
                                log_scale=True, **message))
    except (OSError, ImportError, ValueError):
        return False


def notify(title, body, error=False):
    try:
        subprocess.run(["notify-send", "--app-name=Buttercup", "--expire-time=1500",
                        *(["--urgency=critical"] if error else []), title, body],
                       timeout=2, check=False)
    except (OSError, subprocess.TimeoutExpired):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("up", "down", "toggle", "auto", "manual", "status"))
    parser.add_argument("--step", type=int, default=64, help="exposure lines per press")
    parser.add_argument("--socket", default=os.environ.get("BUTTERCUP_CONTROL_SOCKET", SOCKET))
    parser.add_argument("--quiet", action="store_true", help="suppress fallback notifications")
    args = parser.parse_args()
    try:
        exposure = request(args.socket, "EXPOSURE STATUS")["exposure"]
        if args.action == "status":
            print(json.dumps(exposure))
            return 0
        action = args.action
        if action == "toggle":
            action = "auto" if exposure.get("manual") else "manual"
        token = request(args.socket, f"LEASE CLAIM {OWNER} {LEASE_MS}")["lease"]["token"]
        try:
            current = exposure.get("actual_lines")
            if action == "auto":
                request(args.socket, f"WITH {token} EXPOSURE AUTO")
                message = "Automatic exposure"
                shown = osd(detail="auto  target p90", tone="waiting", position=current, sticky_tone=True)
            elif action == "manual":
                request(args.socket, f"WITH {token} EXPOSURE MANUAL")
                message = f"Manual exposure held at {current} lines"
                shown = osd(detail="manual  held", tone="success", position=current)
            else:
                step = abs(args.step) * (1 if action == "up" else -1)
                request(args.socket, f"WITH {token} EXPOSURE STEP {step}")
                message = f"Exposure {action} {step:+d} lines"
                shown = osd(detail=f"manual  {'brighter ▸' if step > 0 else '◂ darker'}  {step:+d}",
                            tone="working", position=current)
        finally:
            try:
                request(args.socket, f"LEASE RELEASE {token}")
            except (OSError, RuntimeError, ValueError):
                pass
        if not args.quiet and not shown:
            notify("Buttercup exposure", message)
        return 0
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        if not osd(title="Exposure unavailable", detail=str(error), tone="error", sticky_tone=True):
            notify("Buttercup exposure warning", str(error), error=True)
        print(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

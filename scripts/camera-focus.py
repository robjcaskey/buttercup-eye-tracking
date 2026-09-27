#!/usr/bin/env python3
"""Sway shortcut client for Buttercup camera focus: step in/out or return to autofocus.

Each call claims a short camera-control lease on the viewer's control socket,
issues one focus command and releases the lease. The viewer bounds the lens
position; this client never talks to the camera directly.
"""
import argparse
import json
import os
import socket
import subprocess

SOCKET = "/tmp/buttercup-eye-control.sock"
OSD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eye-focus-osd.py")
OWNER = "sway-camera-focus"
LEASE_MS = 3000


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
    """Show the attention-style eye-focus OSD; False if it could not start."""
    try:
        import importlib.util
        import sys
        sys.dont_write_bytecode = True  # keep generated files out of the source tree
        spec = importlib.util.spec_from_file_location("eye_focus_osd", OSD)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.show(message)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("in", "out", "auto", "status"))
    parser.add_argument("--step", type=int, default=8, help="lens position units per press")
    parser.add_argument("--socket", default=os.environ.get("BUTTERCUP_CONTROL_SOCKET", SOCKET))
    parser.add_argument("--quiet", action="store_true", help="suppress success notifications")
    args = parser.parse_args()
    try:
        focus = request(args.socket, "STATUS").get("focus") or {}
        if args.action == "status":
            print(json.dumps(focus))
            return 0
        token = request(args.socket, f"LEASE CLAIM {OWNER} {LEASE_MS}")["lease"]["token"]
        try:
            if args.action == "auto":
                request(args.socket, f"WITH {token} FOCUS AUTO")
                message = "Autofocus re-armed; it runs once the eye crop is locked and still"
                shown = osd(title="Autofocus", detail="Runs once the eye is locked and still", tone="waiting",
                            position=focus.get("position"), sticky_tone=True)
            else:
                current = focus.get("position") or focus.get("target")
                if current is None:
                    raise RuntimeError("focus position is not known yet")
                step = abs(args.step) * (1 if args.action == "in" else -1)
                target = int(current) + step
                reply = request(args.socket, f"WITH {token} FOCUS SET {target}")
                target = reply.get("target", target)
                message = f"Focus {args.action}: {current} -> {target}"
                shown = osd(detail=f"Manual  {'in ▸' if step > 0 else '◂ out'}  {step:+d}", tone="working",
                            position=target)
        finally:
            try:
                request(args.socket, f"LEASE RELEASE {token}")
            except (OSError, RuntimeError, ValueError):
                pass
        if not args.quiet and not shown:
            notify("Buttercup focus", message)
        return 0
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        if not osd(title="Focus unavailable", detail=str(error), tone="error", sticky_tone=True):
            notify("Buttercup focus warning", str(error), error=True)
        print(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

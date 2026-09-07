#!/usr/bin/env python3
"""Sway shortcut client for Buttercup's opt-in absolute uinput pointer."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

IDENTIFIER = "0:0:Buttercup_Gaze_Pointer"
SOCKET = "/tmp/buttercup-eye-control.sock"


def command(path, action, *, focus=False):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(2)
        try:
            connection.connect(path)
        except (FileNotFoundError, ConnectionRefusedError) as error:
            raise RuntimeError(f"Buttercup viewer is not running at {path}. Start the viewer before enabling gaze mouse movement.") from error
        prefix = "GAZE FOCUS" if focus else "MOUSE OUTPUT"
        connection.sendall(f"{prefix} {action}\n".encode())
        connection.shutdown(socket.SHUT_WR)
        data = bytearray()
        while len(data) <= 65536:
            part = connection.recv(4096)
            if not part:
                break
            data.extend(part)
        else:
            raise RuntimeError("Viewer returned an oversized mouse status")
    reply = json.loads(data)
    if not reply.get("ok"):
        raise RuntimeError(reply.get("error") or "Viewer rejected mouse command")
    status = reply.get("gaze_focus" if focus else "mouse_output")
    if not isinstance(status, dict) or not isinstance(status.get("enabled"), bool):
        raise RuntimeError("Viewer does not support desktop mouse output; restart the updated viewer")
    return status


def focus_toggle(args):
    """Independent window focus; never open uinput or map/move a pointer."""
    status = command(args.socket, "STATUS", focus=True)
    if args.action == "status":
        print(json.dumps(status))
        return
    enable = args.action == "on" or (args.action == "toggle" and not status["enabled"])
    try:
        status = command(args.socket, "ON" if enable else "OFF", focus=True)
    except Exception:
        if enable:
            try:
                command(args.socket, "OFF", focus=True)
            except Exception:
                pass
        raise
    if status["enabled"] != enable:
        raise RuntimeError("Viewer did not confirm the requested gaze-focus state")
    if not args.quiet:
        notify("Buttercup gaze focus " + ("ON" if enable else "OFF"),
               "Look at a window briefly to focus it. Pointer movement is OFF." if enable
               else "Focus follows eyes disabled. Pointer position unchanged.")


def sway(*args):
    result = subprocess.run(["swaymsg", "-r", *args], capture_output=True, text=True, timeout=2)
    if result.returncode:
        raise RuntimeError("Could not configure the Buttercup pointer in Sway: " + result.stderr.strip())
    return json.loads(result.stdout)


def map_output(requested):
    outputs = [o for o in sway("-t", "get_outputs") if o.get("active")]
    if requested:
        outputs = [o for o in outputs if o["name"] == requested]
    if len(outputs) != 1:
        raise RuntimeError("Mouse remains OFF: choose one active monitor with --output NAME when using multiple monitors")
    output = outputs[0]["name"]
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", output):
        raise RuntimeError("Mouse remains OFF: unsupported Sway output name")
    # This applies to a future device too, before ON can emit any positions.
    # Never change physical pointers or the attention manager's virtual keyboard.
    reply = sway("input", IDENTIFIER, "map_to_output", output)
    if not reply or not all(item.get("success") for item in reply):
        raise RuntimeError("Mouse remains OFF: Sway rejected pointer-to-monitor mapping")
    return output


def notify(title, body, error=False):
    print(f"{title}: {body}", file=sys.stderr if error else sys.stdout)
    try:
        subprocess.run(["notify-send", "--app-name=Buttercup", "--urgency=" + ("critical" if error else "normal"),
                        "--expire-time=" + ("10000" if error else "2500"), title, body],
                       check=False, timeout=2)
    except (OSError, subprocess.TimeoutExpired):
        pass  # stderr still explains the failure if notifications are absent.


def toggle(args):
    status = command(args.socket, "STATUS")
    if args.action == "status":
        print(json.dumps(status))
        return
    enable = args.action == "on" or (args.action == "toggle" and not status["enabled"])
    output = None
    if enable and not status["enabled"]:
        # Check access before mapping. The viewer independently opens the device
        # and handles ACL/permission changes between this check and ON.
        try:
            fd = os.open("/dev/uinput", os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC)
            os.close(fd)
        except PermissionError as error:
            raise RuntimeError("Mouse remains OFF: no write permission to /dev/uinput. Grant your user write access before enabling; no permissions were changed.") from error
        except OSError as error:
            raise RuntimeError(f"Mouse remains OFF: cannot open /dev/uinput: {error}") from error
        output = map_output(args.output)
    try:
        status = command(args.socket, "ON" if enable else "OFF")
    except Exception:
        # ON may have succeeded before a timeout. Best-effort explicit OFF,
        # never retry TOGGLE (which could accidentally turn it back on).
        if enable:
            try:
                command(args.socket, "OFF")
            except Exception:
                pass
        raise
    if status["enabled"] != enable:
        raise RuntimeError("Viewer did not confirm the requested mouse state")
    if not args.quiet:
        notify("Buttercup mouse " + ("ON" if enable else "OFF"),
               (f"Absolute gaze movement{f' on {output}' if output else ''}. Press the same shortcut to stop."
                if enable else "Gaze mouse movement disabled. Physical mouse unchanged."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("toggle", "on", "off", "status"), default="toggle", nargs="?")
    parser.add_argument("--socket", default=os.environ.get("BUTTERCUP_CONTROL_SOCKET", SOCKET))
    parser.add_argument("--output", default=os.environ.get("BUTTERCUP_MOUSE_OUTPUT"))
    parser.add_argument("--focus", action="store_true", help="toggle window focus instead of pointer movement")
    parser.add_argument("--quiet", action="store_true", help="suppress success notifications, never errors")
    args = parser.parse_args()
    try:
        runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
        with (runtime / "buttercup-mouse-toggle.lock").open("a") as lock:
            # Serialize rapid presses of the two equivalent shortcuts. Do not
            # retain a stale pending toggle indefinitely if the viewer is stuck.
            deadline = time.monotonic() + 2
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Another Buttercup mouse toggle is still pending")
                    time.sleep(0.02)
            if args.focus:
                focus_toggle(args)
            else:
                toggle(args)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        notify("Buttercup gaze focus warning" if args.focus else "Buttercup mouse warning", str(error), error=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

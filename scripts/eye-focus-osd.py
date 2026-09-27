#!/usr/bin/env python3
"""On-screen display for Buttercup camera (eye) focus, styled like the
attention-manager surface: a dark monospace card with a mint eyebrow, shown on
a click-through layer-shell overlay that never takes keyboard focus.

    eye-focus-osd.py serve            # resident overlay (started on demand)
    eye-focus-osd.py show JSON        # send one update, starting the overlay if needed
    eye-focus-osd.py status           # toggle the pinned overview (press again to hide)

JSON fields: title, detail, tone (working|success|waiting|error), position,
kind (focus|exposure|status), eyebrow, range [min, max], log_scale, hold_s.
"""
import json
import os
import socket
import subprocess
import sys
import time

RUNTIME = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
OSD_SOCKET = os.path.join(RUNTIME, "buttercup-eye-focus-osd.sock")
VIEWER_SOCKET = os.environ.get("BUTTERCUP_CONTROL_SOCKET", "/tmp/buttercup-eye-control.sock")
LAYER_SHELL = "/usr/lib/x86_64-linux-gnu/libgtk4-layer-shell.so"
VCM_MIN, VCM_MAX = 32, 992
HIDE_AFTER_S = 1.6

# Palette and geometry copied from attention-manager's attention-surface.
CSS = """
window.eye-focus-osd { background-color: rgba(0,0,0,0); background-image: none; box-shadow: none; }
.attention-card {
  background-color: #141418; background-image: none; color: #e0e0e0;
  border: 2px solid #79798f; border-radius: 10px; padding: 36px 44px 32px 44px;
  box-shadow: 0 12px 42px rgba(0,0,0,0.45); font-family: monospace;
}
.attention-card.working { border-color: rgba(116,185,255,0.65); }
.attention-card.success { border-color: rgba(113,224,184,0.62); }
.attention-card.waiting { border-color: rgba(255,203,107,0.62); }
.attention-card.error { border-color: rgba(255,118,135,0.72); }
.eyebrow { color: #71e0b8; font-size: 15px; font-weight: 750; letter-spacing: 1.6px; margin-bottom: 16px; }
.dialog-title { color: #ffffff; font-size: 34px; font-weight: 750; }
.dialog-body { color: #aaaab8; font-size: 20px; font-weight: 550; margin-top: 10px; }
.gauge { min-height: 10px; margin-top: 22px; border-radius: 5px; background-color: #2a2a33; }
.gauge-fill { min-height: 10px; border-radius: 5px; background-color: #71e0b8; }
.gauge-fill.working { background-color: rgb(116,185,255); }
.gauge-fill.waiting { background-color: rgb(255,203,107); }
.gauge-fill.error { background-color: rgb(255,118,135); }
.section { color: #71e0b8; font-size: 13px; font-weight: 750; letter-spacing: 1.2px; margin-top: 26px; margin-bottom: 10px; }
.pill {
  font-size: 17px; font-weight: 650; padding: 6px 14px; border-radius: 9px;
  border: 1px solid #4a4a58; color: #8a8a98; background-color: #1b1b21;
}
.pill.on { color: #71e0b8; border-color: rgba(113,224,184,0.62); background-color: rgba(113,224,184,0.08); }
.pill.warn { color: rgb(255,203,107); border-color: rgba(255,203,107,0.62); background-color: rgba(255,203,107,0.08); }
.pill.info { color: #e0e0e0; border-color: #626273; }
.kv-grid { margin-top: 14px; }
.kv-key { color: #8a8a98; font-size: 16px; font-weight: 600; min-width: 190px; }
.kv-value { color: #f4f4f6; font-size: 18px; font-weight: 700; min-width: 230px; }
.dialog-footer { color: #a6a6b5; font-size: 15px; margin-top: 26px; }
"""
TONES = ("working", "success", "waiting", "error")


def show(message):
    """Send one update; start the resident overlay on first use."""
    payload = json.dumps(message).encode()
    for attempt in range(2):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
                client.sendto(payload, OSD_SOCKET)
            return True
        except (FileNotFoundError, ConnectionRefusedError):
            if attempt:
                return False
            subprocess.Popen([sys.executable, os.path.abspath(__file__), "serve"],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not os.path.exists(OSD_SOCKET):
                time.sleep(0.03)
            time.sleep(0.05)
    return False


def viewer_status(command, key):
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(0.3)
            connection.connect(VIEWER_SOCKET)
            connection.sendall(command.encode() + b"\n")
            connection.shutdown(socket.SHUT_WR)
            data = bytearray()
            while True:
                part = connection.recv(65536)
                if not part:
                    break
                data.extend(part)
        reply = json.loads(data)
        return reply if key is None else (reply.get(key) or {})
    except (OSError, ValueError):
        return None



REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS = os.path.join(REPO, "outputs", "settings")
CARD_WIDTH = 900
# Fine eye mode: 14704 pixel clocks per line (camera FINE_LINE_LENGTH); the
# line period measured from frame length vs delivered frame rate is ~34.5 us.
LINE_TIME_US = 34.5
MODEL_NAMES = {"eye-student": "Butter Obelisk", "sam31": "SAM 3.1", "native": "Native"}


def read_setting(name):
    try:
        with open(os.path.join(SETTINGS, name)) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def camera_snapshot():
    """One consistent read of viewer and saved state for the overview."""
    status = viewer_status("STATUS", None) or {}
    return {
        "status": status,
        "focus": status.get("focus") or {},
        "exposure": viewer_status("EXPOSURE STATUS", "exposure") or {},
        "stereo": viewer_status("STEREO STATUS", None) or {},
        "cursor": viewer_status("GAZE CURSOR STATUS", "gaze_cursor") or {},
        "mouse": viewer_status("MOUSE OUTPUT STATUS", "mouse_output") or {},
        "mount": read_setting("camera-mount.json") or {},
        "calibration": read_setting("gaze-calibration.json"),
        "lens": read_setting("joint-camera-intrinsics.json"),
    }


def imperial_units():
    """True when the locale's LC_MEASUREMENT is US/imperial (glibc measurement=2)."""
    if not hasattr(imperial_units, "cached"):
        imperial = None
        try:
            output = subprocess.run(["locale", "-k", "LC_MEASUREMENT"], capture_output=True,
                                    text=True, timeout=2).stdout
            for line in output.splitlines():
                if line.startswith("measurement="):
                    imperial = line.split("=", 1)[1].strip() == "2"
        except (OSError, subprocess.TimeoutExpired):
            pass
        if imperial is None:
            name = next((os.environ[key] for key in ("LC_ALL", "LC_MEASUREMENT", "LANG")
                         if os.environ.get(key)), "")
            territory = name.split(".")[0].split("_")[-1].upper() if "_" in name else ""
            imperial = territory in ("US", "LR", "MM")
        imperial_units.cached = imperial
    return imperial_units.cached


def length_text(mm):
    if not mm:
        return "—"
    return f"{mm / 25.4:.1f} in" if imperial_units() else f"{mm / 10:.1f} cm"


def exposure_text(lines):
    if not isinstance(lines, (int, float)):
        return "—"
    return f"{lines * LINE_TIME_US / 1000:.1f} ms ({lines} lines)"


def overview_pills(snapshot):
    """Per section: {"pills": [(text, style)], "values": [(key, value)]}.

    Pills are on/off states (style 'on', 'warn' or '' for off); facts are
    key/value readouts rendered as a label and a value, never as pills.
    """
    status, focus, exposure = snapshot["status"], snapshot["focus"], snapshot["exposure"]
    if not status:
        return {"camera": {"pills": [("viewer not running", "warn")], "values": []},
                "tracking": {"pills": [], "values": []}, "outputs": {"pills": [], "values": []}}
    autofocus = not status.get("manual_focus", False)
    auto_exposure = not exposure.get("manual", True) if exposure else False
    # Index 0 is the subject's right eye, 1 the left (viewer subject_eye_label).
    distances = (list(snapshot["stereo"].get("eye_distance_mm") or []) + [None, None])[:2]

    distance_text = length_text
    camera = {
        "pills": [
            ("● AF auto" if autofocus else "○ AF manual", "on" if autofocus else ""),
            ("● AE auto" if auto_exposure else "○ AE manual", "on" if auto_exposure else ""),
        ],
        "values": [
            ("Lens position", f"{focus.get('position', '—')}" + ("" if focus.get("settled", True) else "  moving")),
            ("Exposure", exposure_text(exposure.get("actual_lines"))),
            ("Right eye distance", distance_text(distances[0])),
            ("Left eye distance", distance_text(distances[1])),
        ],
    }
    mode = (status.get("segmentation") or {}).get("mode", "?")
    stereo = snapshot["stereo"].get("active")
    lens = snapshot["lens"]
    tracking = {
        "pills": [
            ("● stereo" if stereo else "○ single eye", "on" if stereo else ""),
            ("● calibrated" if snapshot["calibration"] else "○ not calibrated",
             "on" if snapshot["calibration"] else "warn"),
        ],
        "values": [
            ("Segmentation model", MODEL_NAMES.get(mode, mode)),
            ("Camera mount", snapshot["mount"].get("mode", "flexible").replace("-", " ")),
            ("Lens model", f"measured {lens['fx_fy_cx_cy_px'][0]:.0f} px" if lens else "nominal 4000 px"),
            ("Calibration eye", status.get("focus_eye", "?")),
        ],
    }
    cursor, mouse = snapshot["cursor"], snapshot["mouse"]
    outputs = {
        "pills": [
            ("● gaze ring" if cursor.get("enabled") else "○ gaze ring", "on" if cursor.get("enabled") else ""),
            ("● mouse control" if mouse.get("enabled") else "○ mouse control", "on" if mouse.get("enabled") else ""),
        ],
        "values": [],
    }
    if cursor.get("enabled") and cursor.get("status") not in (None, "", "tracking"):
        outputs["values"].append(("Gaze ring", cursor["status"]))
    return {"camera": camera, "tracking": tracking, "outputs": outputs}


def serve():
    # gtk4-layer-shell must load before libwayland-client; re-exec once if needed.
    if LAYER_SHELL not in os.environ.get("LD_PRELOAD", "") and os.path.exists(LAYER_SHELL):
        env = dict(os.environ, LD_PRELOAD=(LAYER_SHELL + " " + os.environ.get("LD_PRELOAD", "")).strip())
        os.execve(sys.executable, [sys.executable, os.path.abspath(__file__), "serve"], env)
    # Single instance: a second helper would steal the socket and orphan the
    # first overlay, so later messages could toggle an invisible copy.
    import fcntl
    lock = open(os.path.join(RUNTIME, "buttercup-eye-focus-osd.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    import gi
    gi.require_version("Gtk", "4.0")
    gi.require_version("Gdk", "4.0")
    gi.require_version("Gtk4LayerShell", "1.0")
    from gi.repository import Gdk, GLib, Gtk, Gtk4LayerShell as LayerShell

    try:
        os.unlink(OSD_SOCKET)
    except FileNotFoundError:
        pass
    inbox = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    inbox.bind(OSD_SOCKET)
    inbox.setblocking(False)

    app = Gtk.Application(application_id="org.buttercup.EyeFocusOsd")
    state = {"hide_at": 0.0, "tone": "success", "message": {}, "pinned": False, "pinned_message": {}}

    def activate(app):
        provider = Gtk.CssProvider()
        provider.load_from_data(CSS.encode())
        Gtk.StyleContext.add_provider_for_display(Gdk.Display.get_default(), provider,
                                                  Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        window = Gtk.ApplicationWindow(application=app, title="Eye focus", decorated=False, resizable=False)
        window.add_css_class("eye-focus-osd")
        LayerShell.init_for_window(window)
        LayerShell.set_namespace(window, "buttercup-eye-focus-osd")
        LayerShell.set_layer(window, LayerShell.Layer.OVERLAY)
        LayerShell.set_keyboard_mode(window, LayerShell.KeyboardMode.NONE)
        # No anchors: the compositor centers the surface, like attention
        # manager's centered (foveal/dialog) cards.
        LayerShell.set_exclusive_zone(window, 0)
        window.set_focusable(False)
        window.set_can_target(False)

        def click_through(widget):
            surface = widget.get_surface()
            if surface is not None:
                import cairo
                surface.set_input_region(cairo.Region())
        window.connect("realize", click_through)

        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        card.add_css_class("attention-card")
        card.set_size_request(CARD_WIDTH, -1)
        eyebrow = Gtk.Label(label="EYE CAMERA", xalign=0)
        eyebrow.add_css_class("eyebrow")
        title = Gtk.Label(xalign=0)
        title.add_css_class("dialog-title")
        detail = Gtk.Label(xalign=0, wrap=True)
        detail.add_css_class("dialog-body")
        gauge = Gtk.Box()
        gauge.add_css_class("gauge")
        fill = Gtk.Box()
        fill.add_css_class("gauge-fill")
        gauge.append(fill)
        for widget in (eyebrow, title, detail, gauge):
            card.append(widget)
        rows = {}
        for key, heading in (("camera", "CAMERA"), ("tracking", "EYE TRACKING"), ("outputs", "OUTPUTS")):
            label = Gtk.Label(label=heading, xalign=0)
            label.add_css_class("section")
            flow = Gtk.FlowBox(selection_mode=Gtk.SelectionMode.NONE, max_children_per_line=8,
                               column_spacing=10, row_spacing=10, homogeneous=False,
                               halign=Gtk.Align.START)
            grid = Gtk.Grid(column_spacing=18, row_spacing=6, halign=Gtk.Align.START)
            grid.add_css_class("kv-grid")
            card.append(label)
            card.append(flow)
            card.append(grid)
            rows[key] = (flow, grid)
        footer = Gtk.Label(xalign=0, wrap=True,
                           label="Super+] [ exposure   Shift+] [ focus   \\ autofocus   Shift+\\ auto exposure\n"
                                 "Super+/ status   Shift+M gaze mouse   Shift+O gaze ring   Y wleyes   (Magic works too)")
        footer.add_css_class("dialog-footer")
        card.append(footer)
        window.set_child(card)

        def set_pills(flow, pills):
            while (child := flow.get_first_child()) is not None:
                flow.remove(child)
            for text, style in pills:
                pill = Gtk.Label(label=text, halign=Gtk.Align.START)
                pill.add_css_class("pill")
                if style:
                    pill.add_css_class(style)
                flow.insert(pill, -1)

        def set_values(grid, values):
            while (child := grid.get_first_child()) is not None:
                grid.remove(child)
            # Two key/value pairs per row: dim label, bright value.
            for index, (name, value) in enumerate(values):
                row, column = divmod(index, 2)
                key_label = Gtk.Label(label=name, xalign=0)
                key_label.add_css_class("kv-key")
                value_label = Gtk.Label(label=value, xalign=0)
                value_label.add_css_class("kv-value")
                grid.attach(key_label, column * 2, row, 1, 1)
                grid.attach(value_label, column * 2 + 1, row, 1, 1)
            grid.set_visible(bool(values))

        def render(position, tone, message, overview):
            for name in TONES:
                card.remove_css_class(name)
                fill.remove_css_class(name)
            card.add_css_class(tone)
            fill.add_css_class(tone)
            eyebrow.set_label(message.get("eyebrow", "EYE CAMERA"))
            title.set_label(message.get("title") or (f"Lens {position}" if position is not None else "Lens —"))
            detail.set_label(message.get("detail", ""))
            low, high = message.get("range", (VCM_MIN, VCM_MAX))
            if position is None or high <= low:
                fraction = 0.0
            elif message.get("log_scale"):
                import math
                fraction = math.log(max(position, low) / low) / math.log(high / low)
            else:
                fraction = (position - low) / (high - low)
            fill.set_size_request(max(10, int((CARD_WIDTH - 88) * min(1.0, max(0.0, fraction)))), 10)
            gauge.set_visible(message.get("kind") != "status")
            for key, section in overview.items():
                flow, grid = rows[key]
                set_pills(flow, section["pills"])
                flow.set_visible(bool(section["pills"]))
                set_values(grid, section["values"])

        def poll():
            changed = False
            while True:
                try:
                    data = inbox.recv(65536)
                except BlockingIOError:
                    break
                try:
                    incoming = json.loads(data)
                except ValueError:
                    continue
                changed = True
                if incoming.get("toggle"):
                    # The status card toggles: a press pins it until the next press.
                    visible = state["pinned"] or time.monotonic() < state["hide_at"]
                    state["pinned"] = not (state["pinned"] and visible)
                    state["hide_at"] = 0.0
                    state["pinned_message"] = incoming
                    state["message"] = incoming
                    state["tone"] = incoming.get("tone", "success")
                    continue
                state["message"] = incoming
                state["tone"] = incoming.get("tone") if incoming.get("tone") in TONES else "working"
                state["hide_at"] = time.monotonic() + float(incoming.get("hold_s", HIDE_AFTER_S))
            now = time.monotonic()
            if state["pinned"] and now >= state["hide_at"] and state["message"] is not state["pinned_message"]:
                # A transient change has expired; fall back to the pinned status.
                state["message"] = state["pinned_message"]
                state["tone"] = state["pinned_message"].get("tone", "success")
            if now < state["hide_at"] or state["pinned"]:
                message = dict(state["message"])
                position = message.get("position")
                snapshot = camera_snapshot()
                if message.get("kind") == "status":
                    pass
                elif message.get("kind") == "exposure":
                    exposure = snapshot.get("exposure") or {}
                    if exposure:
                        position = exposure.get("actual_lines", position)
                        if state["tone"] in ("working", "success") and not message.get("sticky_tone"):
                            settled = exposure.get("actual_lines") == exposure.get("target_lines")
                            state["tone"] = "success" if settled else "working"
                        if position is not None:
                            message["title"] = f"Exposure {position * LINE_TIME_US / 1000:.1f} ms"
                            message["detail"] = f"{message.get('detail', '')}   {position} lines".strip()
                else:
                    focus = snapshot.get("focus") or {}
                    if focus:
                        position = focus.get("position", position)
                        # Live lens state: blue while moving, mint once settled.
                        if state["tone"] in ("working", "success") and not message.get("sticky_tone"):
                            state["tone"] = "success" if focus.get("settled") else "working"
                        if not message.get("title") and position is not None:
                            message["title"] = f"Lens {position}"
                render(position, state["tone"], message, overview_pills(snapshot))
                if not window.get_visible():
                    window.present()
            elif window.get_visible() and not changed:
                window.set_visible(False)
            return True

        GLib.timeout_add(120, poll)

    app.connect("activate", activate)
    app.hold()
    try:
        app.run(None)
    finally:
        inbox.close()
        try:
            os.unlink(OSD_SOCKET)
        except FileNotFoundError:
            pass


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "serve":
        serve()
        return 0
    if len(sys.argv) == 3 and sys.argv[1] == "show":
        return 0 if show(json.loads(sys.argv[2])) else 1
    if len(sys.argv) == 2 and sys.argv[1] == "status":
        return 0 if show({"kind": "status", "eyebrow": "EYE TRACKING", "title": "Status",
                          "detail": "Camera, eye tracking and gaze outputs",
                          "tone": "success", "sticky_tone": True, "toggle": True}) else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Shortcut tests without creating devices, moving the cursor or notifying."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("toggle_mouse", Path(__file__).with_name("toggle-mouse-output.py"))
mouse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mouse)


class ShortcutTests(unittest.TestCase):
    def args(self, action="toggle"):
        return SimpleNamespace(action=action, socket="unused", output=None, quiet=True)

    def test_off_needs_no_device_permissions_or_compositor_mapping(self):
        with patch.object(mouse, "command", side_effect=[{"enabled": True}, {"enabled": False}]) as command, \
             patch.object(mouse.os, "open", side_effect=AssertionError("OFF must not open uinput")), \
             patch.object(mouse, "map_output", side_effect=AssertionError("OFF must not need Sway")):
            mouse.toggle(self.args())
        self.assertEqual([c.args[1] for c in command.call_args_list], ["STATUS", "OFF"])

    def test_permission_error_never_sends_on(self):
        with patch.object(mouse, "command", return_value={"enabled": False}) as command, \
             patch.object(mouse.os, "open", side_effect=PermissionError("denied")), \
             patch.object(mouse, "map_output") as mapping:
            with self.assertRaisesRegex(RuntimeError, "no write permission to /dev/uinput"):
                mouse.toggle(self.args())
        command.assert_called_once_with("unused", "STATUS")
        mapping.assert_not_called()

    def test_map_exact_pointer_before_enabling(self):
        calls = []
        def command(_, action):
            calls.append(action)
            return {"enabled": action == "ON"}
        with patch.object(mouse, "command", side_effect=command), \
             patch.object(mouse.os, "open", return_value=123), patch.object(mouse.os, "close"), \
             patch.object(mouse, "map_output", side_effect=lambda _: calls.append("MAP")):
            mouse.toggle(self.args())
        self.assertEqual(calls, ["STATUS", "MAP", "ON"])

    def test_enable_timeout_attempts_off_not_another_toggle(self):
        with patch.object(mouse, "command", side_effect=[{"enabled": False}, TimeoutError("late"), {"enabled": False}]) as command, \
             patch.object(mouse.os, "open", return_value=123), patch.object(mouse.os, "close"), \
             patch.object(mouse, "map_output"):
            with self.assertRaises(TimeoutError):
                mouse.toggle(self.args())
        self.assertEqual([c.args[1] for c in command.call_args_list], ["STATUS", "ON", "OFF"])

    def test_single_monitor_maps_only_buttercup_device(self):
        with patch.object(mouse, "sway", side_effect=[[{"active": True, "name": "DP-3"}], [{"success": True}]]) as sway:
            self.assertEqual(mouse.map_output(None), "DP-3")
        self.assertEqual(sway.call_args.args, ("input", mouse.IDENTIFIER, "map_to_output", "DP-3"))

    def test_multiple_outputs_require_explicit_selection(self):
        outputs = [{"active": True, "name": "DP-3"}, {"active": True, "name": "DP-4"}]
        with patch.object(mouse, "sway", return_value=outputs):
            with self.assertRaisesRegex(RuntimeError, "choose one active monitor"):
                mouse.map_output(None)
        with patch.object(mouse, "sway", side_effect=[outputs, [{"success": True}]]):
            self.assertEqual(mouse.map_output("DP-4"), "DP-4")

    def test_rejected_mapping_does_not_enable(self):
        with patch.object(mouse, "command", return_value={"enabled": False}) as command, \
             patch.object(mouse.os, "open", return_value=123), patch.object(mouse.os, "close"), \
             patch.object(mouse, "map_output", side_effect=RuntimeError("map rejected")):
            with self.assertRaisesRegex(RuntimeError, "map rejected"):
                mouse.toggle(self.args())
        command.assert_called_once_with("unused", "STATUS")

    def test_focus_toggle_needs_no_uinput_access_and_uses_separate_control(self):
        with patch.object(mouse, "command", side_effect=[{"enabled": False}, {"enabled": True}]) as command, \
             patch.object(mouse.os, "open", side_effect=AssertionError("focus must not open uinput")), \
             patch.object(mouse, "map_output", side_effect=AssertionError("focus must not map the pointer")):
            mouse.focus_toggle(self.args())
        self.assertEqual([c.args[1] for c in command.call_args_list], ["STATUS", "ON"])
        self.assertTrue(all(c.kwargs == {"focus": True} for c in command.call_args_list))

    def test_focus_off_is_independent_of_mouse_output(self):
        with patch.object(mouse, "command", side_effect=[{"enabled": True}, {"enabled": False}]) as command:
            mouse.focus_toggle(self.args())
        self.assertEqual([c.args[1] for c in command.call_args_list], ["STATUS", "OFF"])
        self.assertTrue(all(c.kwargs == {"focus": True} for c in command.call_args_list))

    def test_focus_enable_timeout_rolls_back_focus_only(self):
        with patch.object(mouse, "command", side_effect=[{"enabled": False}, TimeoutError("late"), {"enabled": False}]) as command:
            with self.assertRaises(TimeoutError):
                mouse.focus_toggle(self.args())
        self.assertEqual([c.args[1] for c in command.call_args_list], ["STATUS", "ON", "OFF"])
        self.assertTrue(all(c.kwargs == {"focus": True} for c in command.call_args_list))


if __name__ == "__main__":
    unittest.main()

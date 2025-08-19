"""
Keyboard typing helper.

Backends
- macOS (Darwin): AppleScript via `osascript` (allows prompting for Accessibility).
- Windows/Linux: `pynput` Controller for cross-platform keyboard events.

Notes
- macOS: Python host must be allowed under Privacy & Security → Accessibility.
- Linux: On Wayland, global key injection may be restricted; X11 is recommended.
"""
from __future__ import annotations
import platform
import subprocess
from typing import Optional
import os

# Lazy-load pynput to avoid hard dependency and static import issues
_PYNPUT_TRIED = False
_PYNPUT_AVAILABLE = False
_KBController = None  # set to class when available


def _get_pynput_controller():
	"""Return pynput Controller class if available (lazy), else None."""
	global _PYNPUT_TRIED, _PYNPUT_AVAILABLE, _KBController
	if not _PYNPUT_TRIED:
		_PYNPUT_TRIED = True
		try:  # dynamic import
			from pynput.keyboard import Controller  # type: ignore
			_KBController = Controller
			_PYNPUT_AVAILABLE = True
		except Exception:
			_KBController = None
			_PYNPUT_AVAILABLE = False
	return _KBController


_prompted_once = False


def _escape_osascript_text(s: str) -> str:
	# AppleScript string quotes are ", escape them and backslashes
	return s.replace("\\", "\\\\").replace("\"", "\\\"")


def _maybe_prompt_access() -> None:
	"""Open Accessibility settings and show a small dialog once to guide the user."""
	global _prompted_once
	if _prompted_once:
		return
	_prompted_once = True
	try:
		# Open the Accessibility settings pane
		subprocess.run([
			"open",
			"x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
		], check=False)
		# Show a guidance dialog (doesn't require Accessibility)
		msg = (
			"To enable typing into other apps, allow access in System Settings →\n"
			"Privacy & Security → Accessibility. Enable Terminal (or your Python host)."
		)
		subprocess.run([
			"osascript",
			"-e",
			f'display dialog "{_escape_osascript_text(msg)}" buttons {{"OK"}} default button 1 with icon caution giving up after 20',
		], check=False)
	except Exception:
		pass


def request_access() -> None:
	"""Public helper to proactively prompt the user to grant Accessibility access."""
	if platform.system() == 'Darwin':
		_maybe_prompt_access()


def has_access() -> bool:
	"""Return True if the process appears to have Accessibility permission (macOS).

	We check by asking System Events to perform a no-op keystroke of an empty string,
	which requires Accessibility but does not produce visible text.
	"""
	if platform.system() != 'Darwin':
		# No special permission check on Windows/Linux for pynput
		return True
	try:
		res = subprocess.run([
			'osascript',
			'-e', 'tell application "System Events" to keystroke ""'
		], capture_output=True, text=True)
		return res.returncode == 0
	except Exception:
		return False


def type_text(text: str) -> bool:
	"""Type the given text in the focused application. Returns True on success."""
	if not text:
		return False
	sysname = platform.system()
	if sysname == 'Darwin':
		try:
			esc = _escape_osascript_text(text)
			script = f'tell application "System Events" to keystroke "{esc}"'
			# Run quietly; if Accessibility is not granted, this will fail but stay silent
			res = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
			if res.returncode != 0:
				_maybe_prompt_access()
			return res.returncode == 0
		except Exception:
			# Fallback to pynput if available
			Controller = _get_pynput_controller()
			if Controller is not None:
				try:
					Controller().type(text)
					return True
				except Exception:
					return False
			return False
	# Windows / Linux via pynput
	Controller = _get_pynput_controller()
	if Controller is not None and sysname in ('Windows', 'Linux'):
		try:
			Controller().type(text)
			return True
		except Exception:
			return False
	return False


def backend_name() -> str:
	"""Return the active backend name for diagnostics."""
	sysname = platform.system()
	if sysname == 'Darwin':
		return 'applescript'
	Controller = _get_pynput_controller()
	if Controller is not None and sysname in ('Windows', 'Linux'):
		return 'pynput'
	return 'noop'


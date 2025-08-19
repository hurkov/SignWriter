# Signboard

Real-time hand sign recognition with OpenCV + MediaPipe and a simple TUI.

## Features
- Train on your own samples and auto-generate a model
- Live recognition window with overlays
- Optional typing into active app (macOS Accessibility required)
- Sensitivity control and 1s cooldown for output
- Sentence-case typing (first letter upper at start/after punctuation)

## Requirements
- Python 3.10+
- Typing backend:
	- macOS: AppleScript (Accessibility permission required)
	- Windows/Linux: `pynput` keyboard controller

Install deps:

```bash
pip install -r requirements.txt
```

## Run
- TUI app:

```bash
python main.py
```

- Recognition directly:

```bash
python input/recognize.py --prefer-fps --display-width 640 --type
```

## Privacy / personal info
- The repo ignores logs and runtime configs via `.gitignore`.
- No personal paths are committed.

## OS setup notes
- macOS
	- Grant Accessibility to your terminal (or Python app) in System Settings → Privacy & Security → Accessibility.
	- The app can prompt and deep-link to the correct settings pane.
- Windows
	- No special permission is typically required. `pynput` is used for typing into the active window.
	- If you see no output, try running the terminal as Administrator.
- Ubuntu / Linux
	- On Wayland sessions, global key injection may be restricted; try an X11 session for best results.
	- Ensure `xinput`/desktop environment allows simulated key events.

## Notes
- Add proper nouns to `conf/settings.json` under `keyboard.proper_nouns` for smarter casing.

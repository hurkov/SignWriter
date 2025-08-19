"""Settings screen for configuring language, camera, window position, and sensitivity."""

from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Button, Static, Select
from textual.screen import Screen
from conf.config import config
from conf.i18n import t
from core.system_checker import SystemChecker
from typing import List, Tuple


class SimpleSettingsScreen(Screen):
    """Single-page settings for language, camera, position."""

    def __init__(self, user_name: str):
        super().__init__()
        self.user_name = user_name
        self.language_options = [("English", "en"), ("Italiano", "it"), ("Русский", "ru"), ("Українська", "uk")]
        # Build camera options from cached verification if available; fallback to probe
        self.camera_options = []
        # Sensitivity options as percentage display mapped to 0-1 thresholds
        self.sensitivity_options = [
            ("25%", 0.25),
            ("50%", 0.50),
            ("75%", 0.75),
            ("85%", 0.85),
            ("95%", 0.95),
        ]
        self.position_options = [
            ("Top left", "top_left"),
            ("Top right", "top_right"),
            ("Bottom left", "bottom_left"),
            ("Bottom right", "bottom_right"),
            ("Center", "center"),
        ]

    def _get_camera_options(self) -> List[Tuple[str, int]]:
        opts: List[Tuple[str, int]] = []
        try:
            # Prefer cameras cached by the app during verification
            cached = getattr(getattr(self, 'app', None), 'detected_cameras', None)
            devices = cached if cached is not None else SystemChecker().check_camera_availability().get("devices", [])
            for cam in devices:
                cid = int(cam.get("id", 0))
                # Show only "[n] Camera"
                label = f"[{cid}] Camera"
                opts.append((label, cid))
        except Exception:
            pass
        if not opts:
            opts = [("[0] Camera", 0), ("[1] Camera", 1), ("[2] Camera", 2)]
        return opts

    def compose(self) -> ComposeResult:
        current_lang = config.get("ui", "language", "en")
        self.current_lang = current_lang
        # Populate camera options now that app is available (to access cache)
        if not self.camera_options:
            self.camera_options = self._get_camera_options()
        device_id = config.get("camera", "device_id", 0)
        # Coerce device_id to int if it's stored as string
        try:
            device_id = int(device_id)
        except Exception:
            device_id = 0
        # Ensure device_id is valid for current options
        options = self.camera_options if self.camera_options else [("Default (0)", 0)]
        valid_values = {v for _, v in options}
        if device_id not in valid_values:
            # Pick first available option value
            device_id = next(iter(valid_values)) if valid_values else 0
            # Also sync the config silently so future opens stay consistent
            try:
                config.set("camera", "device_id", device_id)
                config.save()
            except Exception:
                pass
        position = config.get("ui", "window_position", "top_left")
        # Current sensitivity (default to 0.75 if not set)
        try:
            current_sens = float(config.get("detection", "confidence_threshold", 0.75))
        except Exception:
            current_sens = 0.75
        # Normalize to one of our options
        sens_values = [v for _, v in self.sensitivity_options]
        if current_sens not in sens_values:
            # Pick closest option
            closest = min(sens_values, key=lambda v: abs(v - current_sens))
            current_sens = closest

        yield Container(
            Vertical(
                Static(t(current_lang, "app_title"), id="settings-title"),
                self._row(t(current_lang, "settings_language"), Select(self.language_options, value=current_lang, id="lang")),
                self._row(t(current_lang, "settings_camera"), Select(options, value=device_id, id="cam")),
                self._row(t(current_lang, "settings_position"), Select(self.position_options, value=position, id="pos")),
                self._row(t(current_lang, "settings_sensitivity"), Select(self.sensitivity_options, value=current_sens, id="sensitivity")),
                Horizontal(
                    Button(t(current_lang, "btn_back"), id="settings-exit", variant="default"),
                    Button(t(current_lang, "btn_save"), id="settings-save", variant="success"),
                    id="settings-actions"
                ),
                id="simple-settings-stack"
            ),
            id="simple-settings-wrapper"
        )

    def _row(self, label: str, widget: Select) -> Horizontal:
        return Horizontal(Static(label, classes="setting-label"), widget, classes="setting-row")

    def on_mount(self) -> None:
        # Apply responsive layout immediately
        try:
            self.on_resize(None)
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "settings-exit":
            self.app.pop_screen()
        elif event.button.id == "settings-save":
            lang = self.query_one("#lang", Select).value
            cam = self.query_one("#cam", Select).value
            pos = self.query_one("#pos", Select).value
            sens = self.query_one("#sensitivity", Select).value
            # Do not change width/height here; recognizer will choose max or defaults
            config.set("camera", "device_id", cam)
            config.set("ui", "language", lang)
            config.set("ui", "window_position", pos)
            # Sensitivity stored as 0..1 float
            try:
                config.set("detection", "confidence_threshold", float(sens))
            except Exception:
                pass
            if config.save():
                self.notify("Settings saved", title="Success")
            else:
                self.notify("Failed to save", title="Error", severity="error")
            # Pop settings first
            self.app.pop_screen()
            # Delay refresh so main menu is fully active
            if self.app and hasattr(self.app, "refresh_language"):
                try:
                    self.app.set_timer(0.05, self.app.refresh_language)
                except Exception:
                    self.app.refresh_language()

    def _apply_language(self, lang: str) -> None:
        try:
            self.query_one("#settings-title", Static).update(t(lang, "app_title"))
            labels = [
                ("#lang", "settings_language"),
                ("#cam", "settings_camera"),
                ("#pos", "settings_position"),
                ("#sensitivity", "settings_sensitivity"),
            ]
            for row in self.query(".setting-row"):  # type: ignore
                static = row.children[0]
                if isinstance(static, Static):
                    for wid, key in labels:
                        if wid.strip('#') in [c.id for c in row.children if hasattr(c, 'id')]:
                            static.update(t(lang, key))
            self.query_one("#settings-exit", Button).label = t(lang, "btn_back")
            self.query_one("#settings-save", Button).label = t(lang, "btn_save")
        except Exception:
            pass

    def on_select_changed(self, event: Select.Changed) -> None:
        """Live-update labels when language changes in the Select."""
        try:
            if getattr(event.select, "id", None) == "lang":
                new_lang = getattr(event, "value", None) or event.select.value
                if new_lang:
                    self._apply_language(new_lang)
        except Exception:
            pass

    def on_resize(self, event) -> None:
        """Responsive layout for Settings: keep content within frame without overflow."""
        try:
            size = self.size
            wrapper = self.query_one("#simple-settings-stack")
            # Adjust wrapper width by terminal width
            if size.width < 60:
                wrapper.styles.width = "96%"
            elif size.width < 90:
                wrapper.styles.width = "85%"
            elif size.width < 120:
                wrapper.styles.width = "75%"
            else:
                wrapper.styles.width = "65%"
            # Compact rows and buttons on very small heights
            compact = size.height < 28
            for row in self.query(".setting-row"):
                row.styles.height = 2 if compact else 3
            actions = self.query_one("#settings-actions")
            for btn in actions.query("Button"):
                btn.styles.height = 2 if compact else 3
        except Exception:
            pass

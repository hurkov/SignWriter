"""Configuration package for signboard app."""
from .config import config
from .settings_screen import SimpleSettingsScreen  # re-export

__all__ = ["config", "SimpleSettingsScreen"]

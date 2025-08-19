"""App configuration management: load/save and access settings."""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for the application"""
    
    def __init__(self, config_file: str = "settings.json"):
        self.config_dir = Path(__file__).parent
        self.config_file = self.config_dir / config_file
        self._settings = self._load_default_settings()
        self.load()
    
    def _load_default_settings(self) -> Dict[str, Any]:
        """Load default configuration settings"""
        return {
            "camera": {
                "width": 640,
                "height": 480,
                "fps": 30,
                "device_id": 0
            },
            "ui": {
                "language": "en",
                "theme": "dark",
                "window_title": "Sign Language Keyboard",
                "window_position": "top_left"
            },
            "detection": {
                "confidence_threshold": 0.7,
                "detection_timeout": 2.0,
                "hand_detection_enabled": True,
                "gesture_smoothing": True
            },
            "keyboard": {
                "typing_delay": 0.1,
                "auto_space": True,
                "caps_lock": False,
                "smart_case": True,            # sentence case: first letter upper at start/after punctuation
                "title_case_each_word": False, # if True, capitalize first letter of every word
                "proper_nouns": []             # words to always TitleCase (e.g., ["Rome", "Kyiv", "Alice"]) 
            },
            "advanced": {
                "debug_mode": False,
                "log_level": "INFO",
                "save_detections": False
            }
        }
    
    def load(self) -> None:
        """Load settings from config file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    self._merge_settings(loaded_settings)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Could not load config file: {e}. Using defaults.")
    
    def save(self) -> bool:
        """Save current settings to config file"""
        try:
            self.config_dir.mkdir(exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._settings, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Could not save config file: {e}")
            return False
    
    def _merge_settings(self, loaded_settings: Dict[str, Any]) -> None:
        """Merge loaded settings with defaults"""
        for category, settings in loaded_settings.items():
            if category in self._settings:
                if isinstance(settings, dict):
                    self._settings[category].update(settings)
                else:
                    self._settings[category] = settings
    
    def get(self, category: str, key: str, default: Any = None) -> Any:
        """Get a specific setting value"""
        return self._settings.get(category, {}).get(key, default)
    
    def set(self, category: str, key: str, value: Any) -> None:
        """Set a specific setting value"""
        if category not in self._settings:
            self._settings[category] = {}
        self._settings[category][key] = value
    
    def get_category(self, category: str) -> Dict[str, Any]:
        """Get all settings for a category"""
        return self._settings.get(category, {})
    
    def set_category(self, category: str, settings: Dict[str, Any]) -> None:
        """Set all settings for a category"""
        self._settings[category] = settings
    
    def reset_to_defaults(self) -> None:
        """Reset all settings to default values"""
        self._settings = self._load_default_settings()
    
    def reset_category(self, category: str) -> None:
        """Reset a specific category to default values"""
        defaults = self._load_default_settings()
        if category in defaults:
            self._settings[category] = defaults[category]
    
    @property
    def all_settings(self) -> Dict[str, Any]:
        """Get all settings"""
        return self._settings.copy()


# Global configuration instance
config = Config()

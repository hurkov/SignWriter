"""
Settings validation and constraints for Sign Language Keyboard configuration
"""

from typing import Any, Dict, List, Tuple, Union


class SettingsValidator:
    """Validates configuration settings and provides constraints"""
    
    # Define valid ranges and options for settings
    CONSTRAINTS = {
        "camera": {
            "width": {"type": int, "options": [320, 640, 800, 1024, 1280, 1920]},
            "height": {"type": int, "options": [240, 480, 600, 640, 768, 720, 1080]},
            "fps": {"type": int, "options": [15, 24, 30, 60]},
            "device_id": {"type": int, "options": [0, 1, 2, 3]}
        },
        "ui": {
            "language": {"type": str, "options": ["en", "it", "ru", "uk"]},
            "theme": {"type": str, "options": ["dark", "light", "auto"]},
            "window_title": {"type": str, "max_length": 50},
            "window_position": {"type": str, "options": ["top_left", "top_right", "bottom_left", "bottom_right", "center"]}
        },
        "detection": {
            "confidence_threshold": {"type": float, "options": [0.1, 0.3, 0.5, 0.7, 0.9]},
            "detection_timeout": {"type": float, "options": [0.5, 1.0, 2.0, 3.0, 5.0]},
            "hand_detection_enabled": {"type": bool},
            "gesture_smoothing": {"type": bool}
        },
        "keyboard": {
            "typing_delay": {"type": float, "options": [0.0, 0.1, 0.2, 0.5, 1.0]},
            "auto_space": {"type": bool},
            "caps_lock": {"type": bool}
        },
        "advanced": {
            "debug_mode": {"type": bool},
            "log_level": {"type": str, "options": ["DEBUG", "INFO", "WARNING", "ERROR"]},
            "save_detections": {"type": bool}
        }
    }
    
    # User-friendly labels for settings
    LABELS = {
        "camera": {
            "width": "Camera Width (px)",
            "height": "Camera Height (px)",
            "fps": "Frame Rate (FPS)",
            "device_id": "Camera Device ID"
        },
        "ui": {
            "language": "Language",
            "theme": "Theme",
            "window_title": "Window Title",
            "window_position": "Window Position"
        },
        "detection": {
            "confidence_threshold": "Detection Confidence",
            "detection_timeout": "Detection Timeout (s)",
            "hand_detection_enabled": "Enable Hand Detection",
            "gesture_smoothing": "Gesture Smoothing"
        },
        "keyboard": {
            "typing_delay": "Typing Delay (s)",
            "auto_space": "Auto Space",
            "caps_lock": "Caps Lock"
        },
        "advanced": {
            "debug_mode": "Debug Mode",
            "log_level": "Log Level",
            "save_detections": "Save Detections"
        }
    }
    
    # Category descriptions
    CATEGORY_DESCRIPTIONS = {
        "camera": "Camera and video capture settings",
        "ui": "User interface and display preferences",
        "detection": "Hand and gesture detection parameters",
        "keyboard": "Keyboard simulation settings",
        "advanced": "Advanced and debugging options"
    }
    
    @classmethod
    def validate_value(cls, category: str, key: str, value: Any) -> Tuple[bool, str]:
        """
        Validate a setting value against constraints
        Returns (is_valid, error_message)
        """
        if category not in cls.CONSTRAINTS:
            return False, f"Unknown category: {category}"
        
        if key not in cls.CONSTRAINTS[category]:
            return False, f"Unknown setting: {key} in {category}"
        
        constraint = cls.CONSTRAINTS[category][key]
        
        # Type validation
        expected_type = constraint["type"]
        if not isinstance(value, expected_type):
            return False, f"Expected {expected_type.__name__}, got {type(value).__name__}"
        
        # String constraints
        if expected_type == str:
            if "options" in constraint and value not in constraint["options"]:
                return False, f"Must be one of: {', '.join(constraint['options'])}"
            if "max_length" in constraint and len(value) > constraint["max_length"]:
                return False, f"Maximum length is {constraint['max_length']}"
        
        # Numeric constraints
        elif expected_type in (int, float):
            if "min" in constraint and value < constraint["min"]:
                return False, f"Minimum value is {constraint['min']}"
            if "max" in constraint and value > constraint["max"]:
                return False, f"Maximum value is {constraint['max']}"
        
        return True, ""
    
    @classmethod
    def get_constraint(cls, category: str, key: str) -> Dict[str, Any]:
        """Get constraint information for a setting"""
        return cls.CONSTRAINTS.get(category, {}).get(key, {})
    
    @classmethod
    def get_label(cls, category: str, key: str) -> str:
        """Get user-friendly label for a setting"""
        return cls.LABELS.get(category, {}).get(key, key.replace("_", " ").title())
    
    @classmethod
    def get_category_description(cls, category: str) -> str:
        """Get description for a category"""
        return cls.CATEGORY_DESCRIPTIONS.get(category, category.title())
    
    @classmethod
    def get_all_categories(cls) -> List[str]:
        """Get list of all setting categories"""
        return list(cls.CONSTRAINTS.keys())
    
    @classmethod
    def get_category_keys(cls, category: str) -> List[str]:
        """Get list of all keys in a category"""
        return list(cls.CONSTRAINTS.get(category, {}).keys())
    
    @classmethod
    def clamp_value(cls, category: str, key: str, value: Any) -> Any:
        """Clamp a value to valid range"""
        constraint = cls.get_constraint(category, key)
        
        if constraint.get("type") in (int, float):
            if "min" in constraint:
                value = max(value, constraint["min"])
            if "max" in constraint:
                value = min(value, constraint["max"])
        
        return value

"""
Core functionality for the Sign Language Keyboard project.
Contains system validation, compatibility checking, and template management.
"""

from .compatibility_checker import CompatibilityChecker

# Only import what exists - comment out until these files are properly implemented
# from .system_checker import SystemChecker
# from .template_manager import TemplateManager

__all__ = [
    'CompatibilityChecker',
    # 'SystemChecker', 
    # 'TemplateManager'
]
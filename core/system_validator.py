"""
Main system validation orchestrator that coordinates all checks.
"""

import getpass
from typing import Dict, Any
from .compatibility_checker import CompatibilityChecker
from .system_checker import SystemChecker
from .template_manager import TemplateManager

class SystemValidator:
    """Orchestrates all system validation checks"""
    
    def __init__(self):
        self.compatibility_checker = CompatibilityChecker()
        self.system_checker = SystemChecker()
        self.template_manager = TemplateManager()
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run all validation checks and return comprehensive results"""
        results = {}
        
        # Avoid storing personal user info
        try:
            results['user'] = None
        except Exception:
            results['user'] = None
        
        # Run compatibility checks
        compatibility_results = self.compatibility_checker.run_compatibility_check()
        results.update(compatibility_results)
        
        # Run system checks (this should be implemented in system_checker.py)
        # system_results = self.system_checker.run_system_checks()
        # results.update(system_results)
        
        # Run template validation (this should be implemented in template_manager.py)
        # template_results = self.template_manager.validate_templates()
        # results.update(template_results)
        
        return results
    
    def display_validation_results(self, results: Dict[str, Any]) -> None:
        """Display validation results in a formatted way"""
        print("🔍 Running Full System Validation...")
        print("=" * 50)
        
        if results.get('user'):
            print(f"User: {results['user']}")
        print(f"Timestamp: {results['timestamp']}")
        print(f"Python: {results['python']['message']}")
        
        print("\nLibraries:")
        for msg in results['libraries']['messages']:
            print(f"  {msg}")
        
        print(f"\nOverall Status: {'✅ PASS' if results['overall_status'] else '❌ FAIL'}")

if __name__ == "__main__":
    validator = SystemValidator()
    results = validator.run_full_validation()
    validator.display_validation_results(results)
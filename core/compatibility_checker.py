import sys
import importlib
import importlib.util
import datetime
from typing import Dict, List, Tuple

class CompatibilityChecker:
    def __init__(self):
        self.min_python_version = (3, 8)  # Minimum Python 3.8
        self.required_libraries = [
            'cv2',
            'mediapipe', 
            'textual',
            'rich',
            'numpy'
        ]

    def check_python_version(self) -> tuple[bool, str]:
        """Python version compatibility check"""
        current_version = sys.version_info[:2]

        if current_version >= self.min_python_version:
            return True, f"✅ Python {sys.version.split()[0]} - Compatible"
        else:
            return False, f"❌ Python {sys.version.split()[0]} - Requires {self.min_python_version[0]}.{self.min_python_version[1]}+" 
    
    def check_libraries(self) -> tuple[bool, list[str]]:
        """Check if all required libraries are available"""
        results = []
        all_libraries_ok = True
        for library in self.required_libraries:
            try:
                spec = importlib.util.find_spec(library)
                if spec is not None:
                    results.append(f"✅ {library} - Available")
                else:
                    results.append(f"❌ {library} - Missing")
                    all_libraries_ok = False
                
            except ImportError:
                results.append(f"❌ {library} - Import Error")
                all_libraries_ok = False
    
        return all_libraries_ok, results
    
    def run_compatibility_check(self) -> Dict[str, any]:
        """Run compatibility checks and return results"""
        results = {}
        
        # Check Python version
        python_ok, python_msg = self.check_python_version()
        results['python'] = {
            'status': python_ok,
            'message': python_msg
        }
        
        # Check libraries
        libraries_ok, library_messages = self.check_libraries()
        results['libraries'] = {
            'status': libraries_ok,
            'messages': library_messages
        }
        
        # Overall compatibility status
        results['overall_status'] = python_ok and libraries_ok
        results['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return results

# Add this at the end of your compatibility_checker.py
if __name__ == "__main__":
    print("🔍 Running Compatibility Check...")
    print("=" * 50)
    
    checker = CompatibilityChecker()
    results = checker.run_compatibility_check()
    
    # Display results
    print(f"Timestamp: {results['timestamp']}")
    print(f"Python: {results['python']['message']}")
    
    print("\nLibraries:")
    for msg in results['libraries']['messages']:
        print(f"  {msg}")
    
    print(f"\nOverall Status: {'✅ PASS' if results['overall_status'] else '❌ FAIL'}")
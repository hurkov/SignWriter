import platform
import subprocess
import sys
import json
import os
import datetime
import psutil
import cv2
from typing import Dict, Any, Optional, Tuple
import traceback
import contextlib
import io
import re
import shutil

# Suppress ALL OpenCV output globally
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

# Suppress OpenCV warnings and errors at the C++ level
cv2.setLogLevel(0)  # 0 = SILENT

class SystemChecker:
    """Comprehensive system information and hardware validation"""
    
    def __init__(self):
        self.platform = platform.system()
        self.architecture = platform.machine()
        self.errors = []
        self.logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        
        # Ensure logs directory exists
        os.makedirs(self.logs_dir, exist_ok=True)
    
    def check_camera_availability(self) -> Dict[str, Any]:
        """Check if camera devices are available"""
        camera_info = {
            'available': False,
            'devices': [],
            'primary_device': None,
            'error': None
        }
        
        try:
            # Check for available camera devices with minimal testing
            available_cameras = []
            
            # Complete output suppression
            import warnings
            warnings.filterwarnings('ignore')
            
            # Capture ALL output including system warnings
            original_stderr = os.dup(2)
            original_stdout = os.dup(1)
            
            try:
                # Redirect to null device
                null_fd = os.open(os.devnull, os.O_WRONLY)
                os.dup2(null_fd, 2)  # stderr
                os.dup2(null_fd, 1)  # stdout
                
                # Set OpenCV to completely silent mode
                cv2.setLogLevel(0)
                
                # Probe a small range of indices
                max_indices = 6
                for camera_id in range(max_indices):
                    try:
                        # Create capture object with explicit backend
                        if self.platform == 'Darwin':
                            cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)
                        else:
                            cap = cv2.VideoCapture(camera_id)
                        
                        if cap.isOpened():
                            # Start with a small resolution for fast probe
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                            # Quick test read
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                # Get current camera properties
                                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                fps = cap.get(cv2.CAP_PROP_FPS)

                                # Try to estimate max resolution by attempting a few common modes (descending)
                                common_res = [
                                    (3840, 2160), (2560, 1440), (1920, 1080), (1600, 1200),
                                    (1280, 1024), (1280, 800), (1280, 720), (1024, 768),
                                    (800, 600), (640, 480)
                                ]
                                max_w, max_h = width, height
                                for w, h in common_res:
                                    try:
                                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                                        ok, _ = cap.read()
                                        if not ok:
                                            continue
                                        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                        if aw * ah >= max_w * max_h:
                                            max_w, max_h = aw, ah
                                    except Exception:
                                        continue

                                # Try to estimate max FPS by probing some common values
                                common_fps = [240, 120, 90, 60, 30, 24, 15]
                                max_fps = fps if fps and fps > 0 else 0
                                try:
                                    # Use the best resolution we found
                                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_w)
                                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max_h)
                                    for f in common_fps:
                                        try:
                                            cap.set(cv2.CAP_PROP_FPS, f)
                                            ok, _ = cap.read()
                                            if not ok:
                                                continue
                                            rfps = cap.get(cv2.CAP_PROP_FPS)
                                            # Some backends return 0; prefer requested f when read is ok
                                            eff = rfps if rfps and rfps > 0 else float(f)
                                            if eff > max_fps:
                                                max_fps = eff
                                        except Exception:
                                            continue
                                except Exception:
                                    pass
                                
                                camera_data = {
                                    'id': camera_id,
                                    'resolution': f"{width}x{height}",
                                    'fps': fps if fps and fps > 0 else 30.0,
                                    'backend': 'AVFOUNDATION' if self.platform == 'Darwin' else 'DEFAULT',
                                    'name': None,
                                    'max_resolution': f"{max_w}x{max_h}",
                                    'max_fps': int(max_fps) if max_fps else None
                                }
                                available_cameras.append(camera_data)
                                
                                if camera_info['primary_device'] is None:
                                    camera_info['primary_device'] = camera_data
                        
                        cap.release()
                        
                    except Exception:
                        # Silently skip cameras that can't be accessed
                        continue
                        
            finally:
                # Restore original file descriptors
                os.dup2(original_stderr, 2)
                os.dup2(original_stdout, 1)
                os.close(null_fd)
                os.close(original_stderr)
                os.close(original_stdout)
            
            # On macOS, map names to indices using ffmpeg's AVFoundation listing when available
            if self.platform == 'Darwin':
                name_by_index = {}
                try:
                    if shutil.which('ffmpeg'):
                        # ffmpeg prints device list to stderr and may exit non-zero
                        proc = subprocess.run(
                            ['ffmpeg', '-hide_banner', '-f', 'avfoundation', '-list_devices', 'true', '-i', ''],
                            capture_output=True, text=True, timeout=5
                        )
                        stderr = proc.stderr or ''
                        in_video = False
                        for raw_line in stderr.splitlines():
                            line = raw_line.strip()
                            if 'AVFoundation video devices:' in line:
                                in_video = True
                                continue
                            if 'AVFoundation audio devices:' in line:
                                in_video = False
                                continue
                            if not in_video:
                                continue
                            # Look for patterns like "[0] FaceTime HD Camera"
                            m = re.search(r"\[(\d+)\]\s+(.+)$", line)
                            if m:
                                idx = int(m.group(1))
                                name = m.group(2).strip()
                                name_by_index[idx] = name
                except Exception:
                    name_by_index = {}

                # Apply names to detected cameras by index when reliably available
                if name_by_index:
                    for cam in available_cameras:
                        cid = int(cam.get('id', -1))
                        if cid in name_by_index:
                            cam['name'] = name_by_index[cid]

            camera_info['available'] = len(available_cameras) > 0
            camera_info['devices'] = available_cameras
            
        except Exception as e:
            camera_info['error'] = str(e)
            self.errors.append(f"Camera check error: {str(e)}")
        
        return camera_info
    
    def get_display_info(self) -> Dict[str, Any]:
        """Get display resolution and information"""
        display_info = {
            'primary_display': None,
            'all_displays': [],
            'total_displays': 0,
            'error': None
        }
        
        try:
            if self.platform == 'Darwin':  # macOS
                # Get display info using system_profiler
                result = subprocess.run(
                    ['system_profiler', 'SPDisplaysDataType', '-json'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    displays = data.get('SPDisplaysDataType', [])
                    
                    for i, display in enumerate(displays):
                        display_data = {
                            'name': display.get('_name', f'Display {i+1}'),
                            'resolution': display.get('_spdisplays_resolution', 'Unknown'),
                            'pixel_depth': display.get('_spdisplays_pixeldepth', 'Unknown'),
                            'main_display': display.get('_spdisplays_main', 'No') == 'Yes'
                        }
                        display_info['all_displays'].append(display_data)
                        
                        if display_data['main_display']:
                            display_info['primary_display'] = display_data
                
                # Alternative method using osascript for screen resolution
                script = '''
                tell application "Finder"
                    get bounds of window of desktop
                end tell
                '''
                result = subprocess.run(['osascript', '-e', script], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    bounds = result.stdout.strip().split(', ')
                    if len(bounds) == 4:
                        width = int(bounds[2]) - int(bounds[0])
                        height = int(bounds[3]) - int(bounds[1])
                        if not display_info['primary_display']:
                            display_info['primary_display'] = {
                                'name': 'Primary Display',
                                'resolution': f"{width} x {height}",
                                'pixel_depth': 'Unknown',
                                'main_display': True
                            }
            
            elif self.platform == 'Linux':
                # Try xrandr for Linux
                result = subprocess.run(['xrandr'], capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if ' connected ' in line and '*' in line:
                            parts = line.split()
                            name = parts[0]
                            for part in parts:
                                if 'x' in part and '*' in part:
                                    resolution = part.split('*')[0]
                                    display_data = {
                                        'name': name,
                                        'resolution': resolution,
                                        'pixel_depth': 'Unknown',
                                        'main_display': 'primary' in line
                                    }
                                    display_info['all_displays'].append(display_data)
                                    if 'primary' in line:
                                        display_info['primary_display'] = display_data
                                    break
            
            elif self.platform == 'Windows':
                # Try wmic for Windows
                result = subprocess.run([
                    'wmic', 'desktopmonitor', 'get', 
                    'screenheight,screenwidth,name', '/format:csv'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    for i, line in enumerate(lines):
                        if line.strip():
                            parts = line.split(',')
                            if len(parts) >= 4:
                                display_data = {
                                    'name': parts[1] if parts[1] else f'Display {i+1}',
                                    'resolution': f"{parts[3]} x {parts[2]}" if parts[2] and parts[3] else 'Unknown',
                                    'pixel_depth': 'Unknown',
                                    'main_display': i == 0
                                }
                                display_info['all_displays'].append(display_data)
                                if i == 0:
                                    display_info['primary_display'] = display_data
            
            display_info['total_displays'] = len(display_info['all_displays'])
            
        except Exception as e:
            display_info['error'] = str(e)
            self.errors.append(f"Display info error: {str(e)}")
        
        return display_info
    
    def get_system_specs(self) -> Dict[str, Any]:
        """Get comprehensive system specifications"""
        specs = {
            'platform': self.platform,
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': self.architecture,
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': sys.version,
            'python_executable': sys.executable,
            'node_name': platform.node(),
            'error': None
        }
        
        try:
            # Memory information
            memory = psutil.virtual_memory()
            specs['memory'] = {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_percent': memory.percent
            }
            
            # CPU information
            specs['cpu'] = {
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'max_frequency': getattr(psutil.cpu_freq(), 'max', 'Unknown') if psutil.cpu_freq() else 'Unknown',
                'current_frequency': getattr(psutil.cpu_freq(), 'current', 'Unknown') if psutil.cpu_freq() else 'Unknown'
            }
            
            # Disk information
            disk = psutil.disk_usage('/')
            specs['disk'] = {
                'total_gb': round(disk.total / (1024**3), 2),
                'free_gb': round(disk.free / (1024**3), 2),
                'used_percent': round((disk.used / disk.total) * 100, 2)
            }
            
            # Platform-specific information
            if self.platform == 'Darwin':  # macOS
                try:
                    result = subprocess.run(['sw_vers'], capture_output=True, text=True)
                    specs['macos_version'] = result.stdout.strip() if result.returncode == 0 else 'Unknown'
                    
                    # Get macOS build info
                    result = subprocess.run(['uname', '-v'], capture_output=True, text=True)
                    specs['kernel_version'] = result.stdout.strip() if result.returncode == 0 else 'Unknown'
                    
                except Exception as e:
                    specs['macos_error'] = str(e)
            
            elif self.platform == 'Windows':
                try:
                    specs['windows_version'] = platform.win32_ver()
                except Exception as e:
                    specs['windows_error'] = str(e)
            
            elif self.platform == 'Linux':
                try:
                    specs['linux_distribution'] = platform.freedesktop_os_release()
                except Exception as e:
                    specs['linux_error'] = str(e)
        
        except Exception as e:
            specs['error'] = str(e)
            self.errors.append(f"System specs error: {str(e)}")
        
        return specs
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information if available"""
        gpu_info = {
            'available': False,
            'devices': [],
            'error': None
        }
        
        try:
            if self.platform == 'Darwin':  # macOS
                result = subprocess.run(
                    ['system_profiler', 'SPDisplaysDataType'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    # Parse GPU info from system_profiler output
                    gpu_info['available'] = True
                    gpu_info['raw_output'] = result.stdout[:500]  # Truncated for brevity
            
            elif self.platform == 'Linux':
                # Try lspci for Linux
                result = subprocess.run(['lspci'], capture_output=True, text=True)
                if result.returncode == 0:
                    gpu_lines = [line for line in result.stdout.split('\n') if 'VGA' in line or 'Display' in line]
                    gpu_info['devices'] = gpu_lines
                    gpu_info['available'] = len(gpu_lines) > 0
            
            elif self.platform == 'Windows':
                # Try wmic for Windows
                result = subprocess.run([
                    'wmic', 'path', 'win32_VideoController', 'get', 'name'
                ], capture_output=True, text=True)
                if result.returncode == 0:
                    devices = [line.strip() for line in result.stdout.split('\n') if line.strip() and line.strip() != 'Name']
                    gpu_info['devices'] = devices
                    gpu_info['available'] = len(devices) > 0
        
        except Exception as e:
            gpu_info['error'] = str(e)
            self.errors.append(f"GPU info error: {str(e)}")
        
        return gpu_info
    
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run all system checks and compile results"""
        timestamp = datetime.datetime.now().isoformat()
        
        results = {
            'timestamp': timestamp,
            'scan_info': {
                'version': '1.0',
                'scan_type': 'comprehensive_system_check'
            },
            'compatibility': {},
            'hardware': {},
            'software': {},
            'display': {},
            'camera': {},
            'errors': [],
            'status': 'success'
        }
        
        try:
            # Import and run compatibility checks
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from core.compatibility_checker import CompatibilityChecker
            compatibility_checker = CompatibilityChecker()
            results['compatibility'] = compatibility_checker.run_compatibility_check()
            
            # Software information
            results['software'] = self.get_system_specs()
            
            # Display information
            results['display'] = self.get_display_info()
            
            # Camera information
            results['camera'] = self.check_camera_availability()
            
            # GPU information
            results['hardware']['gpu'] = self.get_gpu_info()
            
            # Compile all errors
            results['errors'] = self.errors
            
            # Determine overall status based on compatibility and system errors
            compatibility_ok = results['compatibility']['overall_status']
            if self.errors:
                results['status'] = 'completed_with_errors'
            elif not compatibility_ok:
                results['status'] = 'compatibility_issues'
            
        except Exception as e:
            results['status'] = 'failed'
            results['critical_error'] = str(e)
            results['traceback'] = traceback.format_exc()
            self.errors.append(f"Critical error: {str(e)}")
            results['errors'] = self.errors
        
        return results
    
    def save_report(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save results to JSON file in logs directory"""
        try:
            if filename is None:
                if results['status'] == 'failed':
                    filename = 'crash_report.json'
                else:
                    filename = 'report.json'
            
            filepath = os.path.join(self.logs_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            return filepath
            
        except Exception as e:
            # Emergency fallback - try to save at least the error
            emergency_path = os.path.join(self.logs_dir, 'emergency_crash_report.json')
            emergency_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'emergency_save': True,
                'original_save_error': str(e),
                'original_results_status': results.get('status', 'unknown'),
                'errors': results.get('errors', []),
                'critical_error': results.get('critical_error', 'Unknown')
            }
            
            try:
                with open(emergency_path, 'w') as f:
                    json.dump(emergency_data, f, indent=2)
                return emergency_path
            except:
                return f"Failed to save report: {str(e)}"
    
    def run_system_checks(self) -> Tuple[Dict[str, Any], str]:
        """Run comprehensive system checks and save report"""
        try:
            results = self.run_comprehensive_check()
            filepath = self.save_report(results)
            return results, filepath
        
        except Exception as e:
            # Create crash report
            crash_results = {
                'timestamp': datetime.datetime.now().isoformat(),
                'status': 'crashed',
                'critical_error': str(e),
                'traceback': traceback.format_exc(),
                'errors': self.errors + [f"System check crashed: {str(e)}"]
            }
            filepath = self.save_report(crash_results, 'crash_report.json')
            return crash_results, filepath

if __name__ == "__main__":
    print("🔍 Running Comprehensive System Check...")
    print("=" * 60)
    
    checker = SystemChecker()
    results, report_path = checker.run_system_checks()
    
    print(f"✅ System check completed with status: {results['status']}")
    print(f"📄 Report saved to: {report_path}")
    
    if results.get('errors'):
        print(f"⚠️  Errors encountered: {len(results['errors'])}")
        for error in results['errors'][:3]:  # Show first 3 errors
            print(f"   - {error}")
    
    # Display key information
    if 'software' in results:
        print(f"\n💻 Platform: {results['software'].get('platform', 'Unknown')}")
        if 'memory' in results['software']:
            print(f"🧠 Memory: {results['software']['memory'].get('total_gb', 'Unknown')} GB total")
    
    if 'camera' in results:
        camera_count = len(results['camera'].get('devices', []))
        print(f"📷 Cameras: {camera_count} detected")
    
    if 'display' in results:
        display_count = results['display'].get('total_displays', 0)
        print(f"🖥️  Displays: {display_count} detected")
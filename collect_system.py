import os
import platform
import psutil
import subprocess
from typing import Dict, Optional, List
import json
from pathlib import Path

class Collect:
    def __init__(self, device_type: str = 'cpu'):
        """
        Initialize the Gather class
        Args:
            device_type: Type of device to gather info for ('cpu', 'cuda', 'musa')
        """
        self.device_type = device_type.lower()
        self.system_info = {}

    def cpu(self) -> Dict:
        """Get CPU information"""
        return {
            'physical_cores': psutil.cpu_count(logical=False),
            'total_cores': psutil.cpu_count(logical=True),
            'max_frequency': psutil.cpu_freq().max if psutil.cpu_freq() else None,
            'current_frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'cpu_percent': psutil.cpu_percent(interval=1, percpu=True),
            'cpu_stats': dict(psutil.cpu_stats()._asdict()),
            'cpu_model': self._cpu_model()
        }

    def memory(self) -> Dict:
        """Get system memory information"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'free': memory.free,
            'percent': memory.percent,
            'swap_total': swap.total,
            'swap_used': swap.used,
            'swap_free': swap.free,
            'swap_percent': swap.percent
        }

    def gpu(self) -> Optional[Dict]:
        """Get GPU information based on device type"""
        if self.device_type == 'cuda':
            return self._nvidia_gpu()
        elif self.device_type == 'musa':
            return self._musa_gpu()
        return None

    def _nvidia_gpu(self) -> Optional[Dict]:
        """Get NVIDIA GPU information"""
        try:
            cmd = "nvidia-smi --query-gpu=gpu_name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv,noheader,nounits"
            output = subprocess.check_output(cmd.split()).decode()
            gpus = []
            
            for idx, line in enumerate(output.strip().split('\n')):
                name, total, used, free, temp, util = line.split(',')
                gpus.append({
                    'id': idx,
                    'name': name.strip(),
                    'memory_total_mb': float(total),
                    'memory_used_mb': float(used),
                    'memory_free_mb': float(free),
                    'temperature_c': float(temp),
                    'utilization_percent': float(util)
                })
            
            return {'gpus': gpus}
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def _musa_gpu(self) -> Optional[Dict]:
        """Get MUSA GPU information using mthreads-gmi"""
        try:
            cmd = "mthreads-gmi -q"
            output = subprocess.check_output(cmd.split()).decode()
            gpus = []
            
            # Parse the output
            current_gpu = None
            for line in output.strip().split('\n'):
                line = line.strip()
                
                # New GPU section starts
                if line.startswith('GPU'):
                    if current_gpu:
                        gpus.append(current_gpu)
                    current_gpu = {
                        'id': len(gpus),
                        'name': None,
                        'memory_total_mb': None,
                        'memory_used_mb': None,
                        'memory_free_mb': None,
                        'temperature_c': None,
                        'utilization_gpu': None,
                        'utilization_memory': None,
                        'power_draw': None,
                        'clock_graphics': None,
                        'clock_memory': None
                    }
                    
                if not current_gpu:
                    continue
                    
                # Parse relevant information
                if ':' in line:
                    key, value = [x.strip() for x in line.split(':', 1)]
                    
                    if key == 'Product Name':
                        current_gpu['name'] = value
                    elif key == 'Total' and 'MiB' in value:
                        current_gpu['memory_total_mb'] = int(value.replace('MiB', ''))
                    elif key == 'Used' and 'MiB' in value:
                        current_gpu['memory_used_mb'] = int(value.replace('MiB', ''))
                    elif key == 'Free' and 'MiB' in value:
                        current_gpu['memory_free_mb'] = int(value.replace('MiB', ''))
                    elif key == 'GPU Current Temp':
                        current_gpu['temperature_c'] = int(value.replace('C', ''))
                    elif key == 'Gpu' and '%' in value:
                        current_gpu['utilization_gpu'] = int(value.replace('%', ''))
                    elif key == 'Memory' and '%' in value:
                        current_gpu['utilization_memory'] = int(value.replace('%', ''))
                    elif key == 'Power Draw':
                        current_gpu['power_draw'] = float(value.replace('W', ''))
                    elif key == 'Graphics':
                        current_gpu['clock_graphics'] = int(value.replace('MHz', ''))
                    elif key == 'Memory' and 'MHz' in value:
                        current_gpu['clock_memory'] = int(value.replace('MHz', ''))
            
            # Add the last GPU
            if current_gpu:
                gpus.append(current_gpu)
            
            return {'gpus': gpus} if gpus else None
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error getting MUSA GPU info: {e}")
            return None

    def _cpu_model(self) -> str:
        """Get CPU model name"""
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":")[1].strip()
            except:
                pass
        return platform.processor()

    def all(self) -> Dict:
        """Get all system information"""
        self.system_info = {
            'system': {
                'platform': platform.system(),
                'platform_release': platform.release(),
                'platform_version': platform.version(),
                'architecture': platform.machine(),
                'hostname': platform.node(),
                'python_version': platform.python_version(),
            },
            'cpu': self.cpu(),
            'memory': self.memory(),
        }
        
        gpu_info = self.gpu()
        if gpu_info:
            self.system_info['gpu'] = gpu_info
            
        return self.system_info

    def save(self, filepath: str) -> None:
        """Save information to a JSON file"""
        if not self.system_info:
            self.all()
            
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.system_info, f, indent=4)

    def summary(self) -> None:
        """Print system information summary"""
        if not self.system_info:
            self.all()
            
        print("\n=== System Information Summary ===")
        print(f"Platform: {self.system_info['system']['platform']}")
        print(f"Architecture: {self.system_info['system']['architecture']}")
        print(f"\nCPU:")
        print(f"- Model: {self.system_info['cpu']['cpu_model']}")
        print(f"- Physical cores: {self.system_info['cpu']['physical_cores']}")
        print(f"- Total cores: {self.system_info['cpu']['total_cores']}")
        
        print(f"\nMemory:")
        print(f"- Total: {self.system_info['memory']['total'] / (1024**3):.2f} GB")
        print(f"- Available: {self.system_info['memory']['available'] / (1024**3):.2f} GB")
        print(f"- Used: {self.system_info['memory']['percent']}%")
        
        if 'gpu' in self.system_info:
            print(f"\nGPU:")
            for gpu in self.system_info['gpu']['gpus']:
                print(f"- Device {gpu['id']}: {gpu['name']}")
                if 'memory_total_mb' in gpu:
                    print(f"  Memory: {gpu['memory_total_mb']} MB")

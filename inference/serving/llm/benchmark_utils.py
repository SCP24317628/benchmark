import subprocess
import time
import os
import signal
from typing import Optional, Literal, Dict
from pathlib import Path

class LiteLLMService:
    def __init__(self, config_path: str):
        """Initialize LiteLLM service with config file path"""
        if not config_path.endswith('.yaml'):
            raise ValueError("Config file must be a YAML file")
        self.config_path = config_path
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        self.process: Optional[subprocess.Popen] = None

    def start(self) -> bool:
        """
        Start the LiteLLM proxy server
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.process:
            print("LiteLLM service is already running")
            return False
        
        try:
            command = f"litellm --config {self.config_path}"
            self.process = subprocess.Popen(
                args=command,
                shell=True,
                preexec_fn=os.setsid
            )
            print(f"\nStarted LiteLLM service with config: {self.config_path}")
            time.sleep(5)  # Wait for service to initialize
            
            # Check if process is still running
            if self.process.poll() is not None:
                print(f"LiteLLM service failed to start")
                return False
                
            return True
            
        except Exception as e:
            print(f"Error starting LiteLLM service: {e}")
            self.terminate()
            return False

    def terminate(self) -> None:
        """Terminate the LiteLLM proxy server"""
        if self.process:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            print(f"\nTerminated LiteLLM service")
            self.process = None

class VLLMService:
    def __init__(self, 
                 model_name: str,
                 device: Literal['cpu', 'cuda', 'musa'], 
                 script_path: str):
        """
        Initialize VLLM service
        Args:
            model_name: Name of the model (e.g., 'qwen2-7b')
            device: Device to run on ('cpu', 'cuda', or 'musa')
            script_path: Path to the VLLM script
        """
        self.model_name = model_name
        self.device = device
        self.process: Optional[subprocess.Popen] = None
        self.script_path = script_path
        self.available_scripts: Dict[str, str] = {}
        
        if not os.path.exists(self.script_path):
            raise FileNotFoundError(f"VLLM script not found at: {self.script_path}")

    @classmethod
    def load_scripts(cls, script_dir: str) -> Dict[str, str]:
        """
        Load available VLLM scripts from directory
        Args:
            script_dir: Directory containing VLLM scripts
        Returns:
            Dictionary mapping model names to script paths
        """
        script_dir = Path(script_dir)
        if not script_dir.exists():
            raise FileNotFoundError(f"Script directory not found: {script_dir}")

        available_scripts = {}
        for script_file in script_dir.glob("*.sh"):
            model_name = script_file.stem.replace("vllm_", "")
            available_scripts[model_name] = str(script_file)

        print("\nAvailable VLLM scripts:")
        for model_name, script_path in available_scripts.items():
            print(f"- {model_name}: {script_path}")
        print()

        return available_scripts

    def start(self) -> bool:
        """
        Start the VLLM server
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.process:
            print("VLLM service is already running")
            return False
        
        try:
            command = f"bash {self.script_path}"
            with open(self.script_path, 'r') as script_file:
                script_content = script_file.read()
                print(f"\nExecute command:\n{script_content}\n")
            
            self.process = subprocess.Popen(
                args=command,
                shell=True,
                preexec_fn=os.setsid
            )
            print(f"\nStarted VLLM service:")
            print(f"- Model: {self.model_name}")
            print(f"- Device: {self.device}")
            print(f"- Script: {self.script_path}\n")
            time.sleep(5)  # Wait for service to initialize
            
            # Check if process is still running
            if self.process.poll() is not None:
                print(f"VLLM service failed to start")
                return False
                
            return True
            
        except Exception as e:
            print(f"Error starting VLLM service: {e}")
            self.terminate()
            return False

    def switch(self, 
               new_model_name: str,
               new_script_path: str) -> None:
        """
        Switch to a different model
        Args:
            new_model_name: Name of the new model
            new_script_path: Path to the new VLLM script
        """
        print(f"\nSwitching VLLM model:")
        print(f"- From: {self.model_name}")
        print(f"- To: {new_model_name}")
        
        self.terminate()
        self.model_name = new_model_name
        self.script_path = new_script_path
        
        if not os.path.exists(self.script_path):
            raise FileNotFoundError(f"VLLM script not found at: {self.script_path}")
            
        self.start()

    def terminate(self) -> None:
        """Terminate the VLLM server"""
        if self.process:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            print(f"\nTerminated VLLM service:")
            print(f"- Model: {self.model_name}")
            print(f"- Device: {self.device}\n")
            self.process = None

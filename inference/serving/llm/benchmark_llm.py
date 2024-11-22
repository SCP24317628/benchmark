import argparse
from benchmark_utils import LiteLLMService, VLLMService
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import os
import time
import datetime
import subprocess
import platform
import shlex
import asyncio
import signal
import json

# Global script mappings
VLLM_SCRIPTS = {
    'chatglm3-6b': 'vllm_chatglm3_6b.sh',
    'glm-4-9b': 'vllm_glm4_9b.sh',
    'qwen2-72b': 'vllm_qwen2_72b.sh',
    'qwen2-7b': 'vllm_qwen2_7b.sh',
    'openai': 'run_vllm_openai.sh'
}

# Get the absolute path to the script directory
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = CURRENT_DIR / "musa/vllm_scripts"
CONFIG_DIR = CURRENT_DIR / "musa"

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True

async def shutdown_vllm(vllm_service):
    """Gracefully shutdown vLLM service"""
    print("Initiating graceful shutdown of vLLM service...")
    try:
        # Give time for current requests to complete
        await asyncio.sleep(5)
        
        # Get the event loop
        loop = asyncio.get_event_loop()
        
        # Cancel all pending tasks
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        
        # Wait for all tasks to complete with a timeout
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Finally terminate the service
        vllm_service.terminate()
        
    except Exception as e:
        print(f"Error during vLLM shutdown: {e}")
    finally:
        print("vLLM service shutdown complete")

def cleanup_processes(processes):
    """Cleanup benchmark processes"""
    for process in processes:
        if process and process.poll() is None:
            try:
                # Send SIGTERM and wait
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if timeout
                try:
                    process.kill()
                    process.wait(timeout=2)
                except:
                    print(f"Failed to kill process {process.pid}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Config file name located in the musa directory")
    parser.add_argument('--model', 
        type=str, 
        required=True, 
        choices=list(VLLM_SCRIPTS.keys()))
    parser.add_argument('--device', type=str, required=True, choices=['cpu', 'cuda', 'musa'])
    return parser.parse_args()

def deploy_services(config_name: str, model: str, device: str, scripts: Dict[str, str]) -> Tuple[Optional[LiteLLMService], Optional[VLLMService]]:
    """
    Deploy services sequentially
    Returns:
        tuple: (litellm_service, vllm_service) or (None, None) if either fails
    """
    # Construct the full path to the config file
    config_path = CONFIG_DIR / config_name

    # 1. Start LiteLLM first
    print("\nStarting LiteLLM service...")
    litellm_service = LiteLLMService(config_path=str(config_path))
    if not litellm_service.start():
        print("Failed to start LiteLLM service")
        return None, None

    # Wait a bit before starting VLLM
    time.sleep(2)
    
    # 2. Then start VLLM
    print("\nStarting VLLM service...")
    vllm_service = VLLMService(
        model_name=model,
        device=device,
        script_path=scripts[model]
    )
    if not vllm_service.start():
        print("Failed to start VLLM service")
        litellm_service.terminate()
        return None, None

    return litellm_service, vllm_service

def run_perf_script(model: str, script: str, save_dir: str) -> Optional[subprocess.Popen]:
    """Run performance script as a background process"""
    command = f"bash {script} {model} {save_dir}"
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        print(f"Started benchmark process for {model} with PID: {process.pid}")
        return process
    except subprocess.SubprocessError as e:
        print(f"Failed to start benchmark process: {e}")
        return None

def load_workload(workload_pth: str = "workload/workload.json") -> Dict[str, Dict[str, str]]:
    """Load workload configuration from JSON file"""
    try:
        workload_pth = Path(CURRENT_DIR) / workload_pth
        with open(workload_pth, 'r') as f:
            workload = json.load(f)
        print(f"Loaded workload configuration from {workload_pth}")
        return workload
    except Exception as e:
        print(f"Error loading workload configuration: {e}")
        return {}

def run_benchmark(workload: Dict[str, Dict[str, str]]) -> List[subprocess.Popen]:
    """Run benchmarks as background processes and return list of processes"""
    processes = []
    for task_id, task_info in workload.items():
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pth = f"{task_info['model']}_result/{task_id}_{time_stamp}"
        print(f"Running benchmark for {task_info['model']} - {task_id}...")
        process = run_perf_script(task_info['model'], task_info['task'], pth)
        if process:
            processes.append(process)
        time.sleep(1)  # Small delay between launches
    return processes

def main():
    args = parse_args()
    killer = GracefulKiller()
    
    scripts = {
        model: str(SCRIPT_DIR / script_name)
        for model, script_name in VLLM_SCRIPTS.items()
    }
    
    litellm_service = None
    vllm_service = None
    benchmark_processes = []
    
    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        litellm_service, vllm_service = deploy_services(
            config_name=args.config,
            model=args.model,
            device=args.device,
            scripts=scripts
        )
        
        if not litellm_service or not vllm_service:
            print("Failed to deploy services")
            return
            
        print("Waiting for 45 seconds...and running benchmark")
        time.sleep(45)
        
        # Load workload configuration from JSON
        workload = load_workload()
        if not workload:
            print("No workload configuration found. Exiting...")
            return
            
        benchmark_processes = run_benchmark(workload)
        print("Started benchmarks. Waiting for completion...")
        
        # Wait for benchmark processes or interruption
        while any(p.poll() is None for p in benchmark_processes):
            time.sleep(3)
            if killer.kill_now:
                break
            
        print("All benchmarks completed or interrupted. Starting cleanup...")
            
    except KeyboardInterrupt:
        print("\nReceived interrupt signal. Starting graceful shutdown...")
    finally:
        print("Beginning shutdown sequence...")
        
        # First cleanup benchmark processes
        cleanup_processes(benchmark_processes)
        
        # Then handle vLLM shutdown
        if vllm_service:
            # Create new event loop for async shutdown
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(shutdown_vllm(vllm_service))
            finally:
                loop.close()
        
        # Finally stop LiteLLM
        if litellm_service:
            print("Stopping LiteLLM service...")
            litellm_service.terminate()
        
        print("Shutdown sequence complete")

if __name__ == "__main__":
    main()


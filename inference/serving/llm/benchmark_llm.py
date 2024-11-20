import argparse
from benchmark_utils import LiteLLMService, VLLMService
from typing import Dict, Optional, Tuple
from pathlib import Path
import os
import time

# Global script mappings
VLLM_SCRIPTS = {
    'chatglm3-6b': 'vllm_chatglm3_6b.sh',
    'glm4-9b': 'vllm_glm4_9b.sh',
    'qwen2-72b': 'vllm_qwen2_72b.sh',
    'qwen2-7b': 'vllm_qwen2_7b.sh',
    'openai': 'run_vllm_openai.sh'
}

# Get the absolute path to the script directory
CURRENT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = CURRENT_DIR / "musa/vllm_scripts"
CONFIG_DIR = CURRENT_DIR / "musa"

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

def main():
    args = parse_args()
    
    # Create full path mappings for scripts
    scripts = {
        model: str(SCRIPT_DIR / script_name)
        for model, script_name in VLLM_SCRIPTS.items()
    }
    
    litellm_service = None
    vllm_service = None
    
    try:
        # Deploy services sequentially
        litellm_service, vllm_service = deploy_services(
            config_name=args.config,
            model=args.model,
            device=args.device,
            scripts=scripts
        )
        
        if not litellm_service or not vllm_service:
            print("Failed to deploy services")
            return
            
        # Run the services for 60 seconds
        print("Services are running for 60 seconds...")
        time.sleep(60)
            
    except KeyboardInterrupt:
        print("\nReceived interrupt signal. Shutting down services...")
    finally:
        # Cleanup
        if litellm_service:
            litellm_service.terminate()
        if vllm_service:
            vllm_service.terminate()

if __name__ == "__main__":
    main()


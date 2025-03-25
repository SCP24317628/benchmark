import argparse
import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

# Precision mapping
PRECISION_MAP = {
    "float16": "FP16",
    "bfloat16": "BF16",
    "int8": "INT8",
    "int4": "INT4",
    "float32": "FP32",
    "float64": "FP64",
    "fp8": "FP8",
    "fp4": "FP4",
    "fp16": "FP16",
    "bf16": "BF16",
    "fp32": "FP32"
}

def get_default_output_path(model_name: str, gpu_name: str, device: str) -> str:
    """Generate default output path with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}-{gpu_name}-{device}-{timestamp}.json"
    return os.path.join("results", filename)

def ensure_output_dir(output_path: str) -> None:
    """Ensure output directory exists."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

def normalize_precision(precision: str) -> str:
    """Normalize precision value to standard format."""
    precision = precision.lower()
    return PRECISION_MAP.get(precision, precision)

def convert_trtllm_data(
    input_file: str,
    gpu_name: str,
    model_name: Optional[str] = None,
    driver: Optional[str] = None,
    driver_version: Optional[str] = None,
    backend: Optional[str] = None,
    backend_version: Optional[str] = None,
    engine: Optional[str] = None,
    engine_version: Optional[str] = None,
) -> List[Dict]:
    """Convert TRT-LLM benchmark data to target format."""
    results = []
    
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            result = {
                'brand': 'NVIDIA',  # Assuming NVIDIA for CUDA backend
                'gpuName': gpu_name,
                'gpuNum': int(row['world_size']),
                'precision': normalize_precision(row['precision']),
                'batchSize': int(row['batch_size']),
                'prefillTokens': int(row['input_length']),
                'decodeTokens': int(row['output_length']),
                'prefillLatency': float(row['prefill_latency(ms)']),
                'throughput': float(row['generation_tokens_per_second']) / int(row['batch_size']),
                'totalThroughput': float(row['generation_tokens_per_second']),
                
                # Optional fields
                'tokens_per_sec': float(row['tokens_per_sec']) if row.get('tokens_per_sec') else None,
                'gpu_peak_mem': float(row['gpu_peak_mem(gb)']) if row.get('gpu_peak_mem(gb)') else None,
                'generation_time': float(row['generation_time(ms)']) if row.get('generation_time(ms)') else None,
                'latency': float(row['latency(ms)']) if row.get('latency(ms)') else None,
                'quantization': row.get('quantization'),
                'total_tokens': float(row['total_generated_tokens']) if row.get('total_generated_tokens') else None,
                
                # Additional fields from arguments
                'modelName': model_name or row.get('engine_dir'),
                'driver': driver,
                'driverVersion': driver_version,
                'backend': backend,
                'backendVersion': backend_version,
                'engine': engine,
                'engineVersion': engine_version,
            }
            results.append(result)
    
    return results

def convert_musa_data(
    input_file: str,
    gpu_name: str,
    model_name: Optional[str] = None,
    driver: Optional[str] = None,
    driver_version: Optional[str] = None,
    backend: Optional[str] = None,
    backend_version: Optional[str] = None,
    engine: Optional[str] = None,
    engine_version: Optional[str] = None,
) -> List[Dict]:
    """Convert MUSA benchmark data to target format."""
    results = []
    
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            result = {
                'brand': 'MT',  # MT
                'gpuName': gpu_name,
                'gpuNum': int(row['GPU_Num']),
                'precision': normalize_precision(row['Data_Type']),
                'batchSize': int(row['batch']),
                'prefillTokens': int(row['prefill_tokens']),
                'decodeTokens': int(row['decode_tokens']),
                'prefillLatency': float(row['prefill_latency(ms)']),
                'throughput': float(row['single_batch_decode_tps']),  # per-batch throughput
                'totalThroughput': float(row['total_decode_tps']),
                
                # Optional fields
                'tokens_per_sec': float(row['tokens_per_second']),
                'gpu_peak_mem': None,  # Not available in MUSA data
                'generation_time': float(row['generation_time(ms)']),
                'latency': float(row['latency(ms)']),
                'quantization': row['quantization'],
                'total_tokens': int(row['generated_tokens']),
                
                # Additional fields from arguments
                'modelName': model_name or row.get('Model'),
                'driver': driver,
                'driverVersion': driver_version,
                'backend': backend,
                'backendVersion': backend_version,
                'engine': engine,
                'engineVersion': engine_version,
            }
            results.append(result)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Convert benchmark data to target format')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', help='Output JSON file path (default: results/model_name-gpu_name-device-timestamp.json)')
    parser.add_argument('--device', choices=['cuda', 'musa'], required=True, help='Device type')
    parser.add_argument('--gpu-name', required=True, help='GPU name')
    parser.add_argument('--model-name', help='Model name (optional)')
    parser.add_argument('--driver', help='Driver name (optional)')
    parser.add_argument('--driver-version', help='Driver version (optional)')
    parser.add_argument('--backend', help='Backend name (optional)')
    parser.add_argument('--backend-version', help='Backend version (optional)')
    parser.add_argument('--engine', help='Engine name (optional)')
    parser.add_argument('--engine-version', help='Engine version (optional)')
    
    args = parser.parse_args()
    
    # Set default model name if not provided
    if not args.model_name:
        args.model_name = os.path.splitext(os.path.basename(args.input))[0]
    
    # Set default output path if not provided
    if not args.output:
        args.output = get_default_output_path(args.model_name, args.gpu_name, args.device)
    
    # Ensure output directory exists
    ensure_output_dir(args.output)
    
    # Convert data based on device type
    if args.device == 'cuda':
        results = convert_trtllm_data(
            args.input,
            args.gpu_name,
            args.model_name,
            args.driver,
            args.driver_version,
            args.backend,
            args.backend_version,
            args.engine,
            args.engine_version,
        )
    else:  # musa
        results = convert_musa_data(
            args.input,
            gpu_name=args.gpu_name,
            model_name=args.model_name,
            driver=args.driver,
            driver_version=args.driver_version,
            backend=args.backend,
            backend_version=args.backend_version,
            engine=args.engine,
            engine_version=args.engine_version,
        )
    
    # Write results to JSON file
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {args.output}")

if __name__ == '__main__':
    main()

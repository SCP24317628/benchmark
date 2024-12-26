#!/bin/bash

# Default values
DEFAULT_DEVICE="cuda"
DEFAULT_GPU_NAME="A100"
DEFAULT_BACKEND="cuda"
DEFAULT_BACKEND_VERSION="12.6"
DEFAULT_ENGINE="trt-llm"
DEFAULT_ENGINE_VERSION="0.14.0.dev2024091000"
DEFAULT_DRIVER="NVIDIA_Linux_x86_64"
DEFAULT_DRIVER_VERSION="560.35.03"

# Help message
usage() {
    echo "Usage: $0 -i <input_csv> -m <model_name> [-o <output_json>]"
    echo
    echo "Convert benchmark data with default settings for CUDA/TRT-LLM"
    echo
    echo "Required arguments:"
    echo "  -i, --input      Input CSV file path"
    echo "  -m, --model      Model name (e.g., 'Llama-2-7b', 'Llama-2-13b')"
    echo
    echo "Optional arguments:"
    echo "  -o, --output     Output JSON file path"
    echo "                   (default: results/model_name-gpu_name-device-timestamp.json)"
    echo "  -h, --help       Show this help message"
    echo
    echo "Default values used:"
    echo "  Device: $DEFAULT_DEVICE"
    echo "  GPU: $DEFAULT_GPU_NAME"
    echo "  Backend: $DEFAULT_BACKEND"
    echo "  Backend Version: $DEFAULT_BACKEND_VERSION"
    echo "  Engine: $DEFAULT_ENGINE"
    echo "  Engine Version: $DEFAULT_ENGINE_VERSION"
    echo "  Driver: $DEFAULT_DRIVER"
    echo "  Driver Version: $DEFAULT_DRIVER_VERSION"
    exit 1
}

# Parse command line arguments
INPUT_FILE=""
OUTPUT_FILE=""
MODEL_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

# Check if input file is provided
if [ -z "$INPUT_FILE" ]; then
    echo "Error: Input file is required"
    usage
fi

# Check if model name is provided
if [ -z "$MODEL_NAME" ]; then
    echo "Error: Model name is required"
    usage
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' does not exist"
    exit 1
fi

# Construct python command
CMD="python convert_data.py \
    --input \"$INPUT_FILE\" \
    --device $DEFAULT_DEVICE \
    --gpu-name \"$DEFAULT_GPU_NAME\" \
    --model-name \"$MODEL_NAME\" \
    --backend $DEFAULT_BACKEND \
    --backend-version $DEFAULT_BACKEND_VERSION \
    --engine $DEFAULT_ENGINE \
    --engine-version $DEFAULT_ENGINE_VERSION \
    --driver $DEFAULT_DRIVER \
    --driver-version $DEFAULT_DRIVER_VERSION"

# Add output file if specified
if [ ! -z "$OUTPUT_FILE" ]; then
    CMD="$CMD --output \"$OUTPUT_FILE\""
fi

# Execute the command
echo "Running conversion..."
eval $CMD

# Check if conversion was successful
if [ $? -eq 0 ]; then
    echo "Conversion completed successfully"
else
    echo "Error: Conversion failed"
    exit 1
fi

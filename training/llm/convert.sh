#!/bin/bash

# Default values
INPUT_FILE="template/zhijiang_training_0121.csv"
DEVICE="nv"
DRIVER="nvidia_linux"
BACKEND="cuda"
DRIVER_VERSION="unkonwn"
BACKEND_VERSION="unkonwn"
OUTPUT_FILE="results/convert_$(date +%Y%m%d_%H%M%S).json"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_file)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="results/$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Create results directory if it doesn't exist
mkdir -p "$(dirname "${OUTPUT_FILE}")"

# Run the conversion script with default values
python3 convert_data.py \
    --input_file "${INPUT_FILE}" \
    --device "${DEVICE}" \
    --driver "${DRIVER}" \
    --backend "${BACKEND}" \
    --driver_version "${DRIVER_VERSION}" \
    --backend_version "${BACKEND_VERSION}" \
    --output_file "${OUTPUT_FILE}"

# Check if conversion was successful
if [ $? -eq 0 ]; then
    echo "Conversion completed successfully"
else
    echo "Error during conversion"
    exit 1
fi

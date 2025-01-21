import pandas as pd
import json
import argparse
from datetime import datetime
import os

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

def convert_csv_to_json(gpu_name, brand, input_file, theory_tflops, output_file=None):
    """
    Convert CSV data to JSON format with specific requirements
    
    Args:
        gpu_name (str): Name of the GPU to replace existing gpu_name values
        brand (str): Brand name of the GPU
        input_file (str): Path to input CSV file
        theory_tflops (float): Theoretical TFLOPS of the GPU
        output_file (str, optional): Path to output JSON file. If None, generates default name
    """
    try:
        # Read CSV file
        df = pd.read_csv(input_file)
        
        # Replace gpu_name with the provided value
        df['gpu_name'] = gpu_name
        
        # Add brand information
        df['brand'] = brand
        
        # Map dtype to standardized precision format
        df['dtype'] = df['dtype'].map(lambda x: PRECISION_MAP.get(x.lower(), x))
        
        # Add new calculated fields with rounding
        df['tflops'] = (df['gops'] / 1000).round(3)
        df['m_n_k'] = df.apply(lambda row: f"{row['m']}_{row['n']}_{row['k']}", axis=1)
        df['efficiency'] = (df['tflops'] / theory_tflops).round(3)
        
        # Rename columns
        df = df.rename(columns={
            'driver_ver': 'driverVersion',
            'backend_ver': 'backendVersion'
        })
        
        # Convert DataFrame to list of dictionaries
        records = df.to_dict('records')
        
        # Generate default output filename if none provided
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            input_dir = os.path.dirname(input_file)
            output_file = os.path.join(input_dir, f'convert_{timestamp}.json')
        
        # Write to JSON file
        with open(output_file, 'w') as f:
            json.dump(records, f, indent=2)
            
        print(f"Successfully converted data to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert CSV data to JSON format')
    parser.add_argument('--gpu_name', '-g', required=True, help='Name of the GPU to replace existing gpu_name values')
    parser.add_argument('--brand', '-b', required=True, help='Brand name of the GPU')
    parser.add_argument('--input_file', '-i', required=True, help='Path to input CSV file')
    parser.add_argument('--theory_tflops', '-t', required=True, type=float, help='Theoretical TFLOPS of the GPU')
    parser.add_argument('--output_file', '-o', help='Path to output JSON file (optional)')
    
    args = parser.parse_args()
    
    # Call conversion function
    convert_csv_to_json(args.gpu_name, args.brand, args.input_file, args.theory_tflops, args.output_file)

if __name__ == "__main__":
    main()

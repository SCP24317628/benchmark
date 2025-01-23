import pandas as pd
import json
import argparse
from datetime import datetime
import os

def convert_training_data(input_file, device, driver, backend, driver_version=None, backend_version=None, output_file=None):
    """
    Convert training CSV data to JSON format with additional device information
    
    Args:
        input_file (str): Path to input CSV file
        device (str): Device name
        driver (str): Driver name
        backend (str): Backend name
        driver_version (str, optional): Driver version
        backend_version (str, optional): Backend version
        output_file (str, optional): Path to output JSON file. If None, generates default name
    """
    try:
        # Read CSV file
        df = pd.read_csv(input_file)
        
        # Add device information
        df['device'] = device
        df['driver'] = driver
        df['backend'] = backend
        
        # Add versions if provided
        if driver_version:
            df['driverVersion'] = driver_version
        if backend_version:
            df['backendVersion'] = backend_version
            
        # Round numeric fields to 3 decimal places
        numeric_columns = ['tflops_per_gpu', 'mfu', 'throughput_per_sec_per_gpu']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].round(3)
        
        # Convert DataFrame to list of dictionaries
        records = df.to_dict('records')
        
        # Generate default output filename if none provided
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Create results directory if it doesn't exist
            results_dir = os.path.join(os.path.dirname(input_file), '../results')
            os.makedirs(results_dir, exist_ok=True)
            output_file = os.path.join(results_dir, f'convert_{timestamp}.json')
        
        # Write to JSON file
        with open(output_file, 'w') as f:
            json.dump(records, f, indent=2)
            
        print(f"Successfully converted training data to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert training CSV data to JSON format')
    parser.add_argument('--input_file', '-i', required=True, help='Path to input CSV file')
    parser.add_argument('--device', '-d', required=True, help='Device name')
    parser.add_argument('--driver', '-dr', required=True, help='Driver name')
    parser.add_argument('--backend', '-b', required=True, help='Backend name')
    parser.add_argument('--driver_version', '-dv', help='Driver version (optional)')
    parser.add_argument('--backend_version', '-bv', help='Backend version (optional)')
    parser.add_argument('--output_file', '-o', help='Path to output JSON file (optional)')
    
    args = parser.parse_args()
    
    # Call conversion function
    convert_training_data(
        args.input_file,
        args.device,
        args.driver,
        args.backend,
        args.driver_version,
        args.backend_version,
        args.output_file
    )

if __name__ == "__main__":
    main()

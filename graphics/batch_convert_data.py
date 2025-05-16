# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
import sys
from convert_data import convert_xlsx_to_json

def batch_convert(input_folder: str):
    """
    Convert all XLSX files in the input folder to JSON format.
    Output files will be saved in an 'output_json' subfolder.
    """
    try:
        # Convert input folder to Path object
        input_path = Path(input_folder)
        if not input_path.exists():
            print(f"Input folder {input_folder} does not exist", file=sys.stderr)
            sys.exit(1)

        # Create output folder
        output_folder = input_path / "output_json"
        output_folder.mkdir(exist_ok=True)

        # Find all XLSX files in the input folder
        xlsx_files = list(input_path.glob("*.xlsx"))
        
        if not xlsx_files:
            print(f"No XLSX files found in {input_folder}")
            return

        # Process each XLSX file
        for xlsx_file in xlsx_files:
            output_file = output_folder / f"{xlsx_file.stem}.json"
            print(f"Converting {xlsx_file.name} to {output_file.name}...")
            convert_xlsx_to_json(str(xlsx_file), str(output_file))

        print(f"\nConversion complete! {len(xlsx_files)} files processed.")
        print(f"Output files are saved in: {output_folder}")

    except Exception as e:
        print(f"Error during batch conversion: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Batch convert XLSX files to JSON format')
    parser.add_argument('--input', '-i', required=True, help='Input folder containing XLSX files')
    
    args = parser.parse_args()
    batch_convert(args.input)

if __name__ == '__main__':
    main()

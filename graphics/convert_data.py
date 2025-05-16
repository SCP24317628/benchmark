import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import argparse
import sys
from pathlib import Path

def parse_date(date_str: str) -> Optional[str]:
    if pd.isna(date_str):
        return None
    try:
        # Try different date formats
        for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y', '%Y%m%d']:
            try:
                return datetime.strptime(str(date_str), fmt).isoformat()
            except ValueError:
                continue

        # Try year-month formats
        for fmt in ['%Y%m', '%Y/%m']:
            try:
                # Default day to 01 for year-month formats
                dt_obj = datetime.strptime(str(date_str), fmt)
                return dt_obj.replace(day=1).isoformat()
            except ValueError:
                continue
        
        return None
    except Exception: # Broad exception to catch any other parsing errors
        return None

def clean_value(value: Any) -> Any:
    """Clean and convert values to appropriate types."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        value = value.strip()
        if value == '':
            return None
    return value

def convert_xlsx_to_json(input_file: str, output_file: str):
    """
    Convert graphics benchmark data from XLSX to JSON without modifying column names.
    """
    try:
        # Read Excel file
        df = pd.read_excel(input_file)
        
        # Convert DataFrame to list of dictionaries
        records = []
        for _, row in df.iterrows():
            record = {}
            
            # Process all columns without changing names
            for column in df.columns:
                value = clean_value(row.get(column))
                
                # Special handling for date fields
                if 'date' in column.lower():
                    value = parse_date(row.get(column))
                
                # Ensure driverVersion is a string
                if column == 'driverVersion' and value is not None:
                    value = str(value)

                # Store the value with the original column name
                record[column] = value
            
            records.append(record)
        
        # Write to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
            
        print(f"Successfully converted {len(records)} records to {output_file}")
        
    except Exception as e:
        print(f"Error converting file: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Convert graphics benchmark XLSX file to JSON format')
    parser.add_argument('--input', '-i', help='Input XLSX file path')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file {args.input} does not exist", file=sys.stderr)
        sys.exit(1)
        
    output_file = args.output if args.output else input_path.with_suffix('.json')
    
    convert_xlsx_to_json(args.input, output_file)

if __name__ == '__main__':
    main()

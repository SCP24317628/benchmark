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
        for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y']:
            try:
                return datetime.strptime(str(date_str), fmt).isoformat()
            except ValueError:
                continue
        return None
    except:
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
    try:
        # Read Excel file
        df = pd.read_excel(input_file)
        
        # Convert DataFrame to list of dictionaries
        records = []
        for _, row in df.iterrows():
            record = {}
            
            # Required fields check
            name = clean_value(row.get('name'))
            brand = clean_value(row.get('brand'))
            
            if not name or not brand:
                print(f"Skipping row due to missing required fields (name or brand)")
                continue
                
            record['name'] = name
            record['brand'] = brand
            
            # Handle usageScenario specially - split by semicolon and clean
            if 'usageScenario' in row:
                usage = clean_value(row['usageScenario'])
                if usage:
                    record['usageScenario'] = [s.strip() for s in str(usage).split(';') if s.strip()]
                else:
                    record['usageScenario'] = []
            
            # Handle numeric fields
            numeric_fields = ['price', 'cores', 'shadingUnits', 'tensorCores', 'RTCores', 
                            'TMUs', 'ROPs', 'SMs', 'GPCs', 'baseClock', 'boostClock', 
                            'transistors', 'L1Cache', 'L2Cache', 'LLCache', 'memory',
                            'memoryBandwidth', 'memoryClock', 'memoryBusWidth', 'power',
                            'fp4', 'fp8', 'int4', 'int8', 'fp16', 'bf16', 'fp32', 'fp64',
                            'pixelRate', 'textureRate', 'linkSpeed', 'dieSize']
            
            for field in numeric_fields:
                if field in row:
                    record[field] = clean_value(row[field])
            
            # Handle date fields
            if 'releaseDate' in row:
                record['releaseDate'] = parse_date(row['releaseDate'])
            
            # Handle boolean fields
            if 'isEstimated' in row:
                record['isEstimated'] = bool(clean_value(row['isEstimated']))
            
            # Handle all other string fields
            string_fields = ['chipType', 'sku', 'architecture', 'foundry', 'manufacturing',
                           'memoryType', 'busInterface', 'lanes', 'moduleType',
                           'thermalType', 'slotWidth', 'decoder', 'encoder', 'scaleUp',
                           'scaleConfig', 'scaleOut', 'marketSegment', 'productType']
            
            for field in string_fields:
                if field in row:
                    record[field] = clean_value(row[field])
            
            records.append(record)
        
        # Write to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
            
        print(f"Successfully converted {len(records)} records to {output_file}")
        
    except Exception as e:
        print(f"Error converting file: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Convert XLSX file to JSON format')
    parser.add_argument('input_file', help='Input XLSX file path')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Input file {args.input_file} does not exist", file=sys.stderr)
        sys.exit(1)
        
    output_file = args.output if args.output else input_path.with_suffix('.json')
    
    convert_xlsx_to_json(args.input_file, output_file)

if __name__ == '__main__':
    main()

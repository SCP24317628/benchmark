import os
import json
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob

class DataConverter:
    def __init__(
        self,
        model: str,
        data_type: str,
        driver: str,
        driver_version: str,
        backend: str,
        backend_version: str,
        engine: str,
        engine_version: str,
        serving: str,
        serving_version: str,
        gpu: str,
        gpu_num: int = 1,
        tp: int = 1,
        pp: Optional[int] = None,
        dp: Optional[int] = None,
        ep: Optional[int] = None,
        model_alias: Optional[str] = None,
        base_dir: str = "result_outputs",
        source: str = "in-house_benchmark",
    ):
        self.model = model
        self.model_alias = model_alias or model  # Use alias if provided, otherwise use original model name
        self.extra_info = {
            "model": self.model_alias,  # Use the alias in extra info
            "dataType": data_type,
            "driver": driver,
            "driverVersion": driver_version,
            "backend": backend,
            "backendVersion": backend_version,
            "engine": engine,
            "engineVersion": engine_version,
            "serving": serving,
            "servingVersion": serving_version,
            "gpu": gpu,
            "gpuNum": gpu_num,
            "tp": tp,
            "pp": pp,
            "dp": dp,
            "ep": ep,
            "source": source,
        }
        # Get the directory where convert_data.py is located
        current_dir = Path(__file__).parent
        # Set paths relative to the script location
        self.base_dir = current_dir / base_dir
        # Create output directory in convert_data folder using original model name for path
        self.output_dir = current_dir / "convert_data" / f"{model}_data"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_result_directories(self) -> List[Path]:
        """Find all directories containing results for the specified model"""
        model_result_dir = self.base_dir / f"{self.model}_result"
        if not model_result_dir.exists():
            raise FileNotFoundError(f"No results directory found for model {self.model}")
        
        # Find all subdirectories that match the pattern
        result_dirs = []
        for dir_path in model_result_dir.glob(f"{self.model}_inp*_out*_conc*_*"):
            if dir_path.is_dir():
                result_dirs.append(dir_path)
        
        return result_dirs

    def process_summary_file(self, summary_file: Path) -> Optional[pd.DataFrame]:
        """Process a single summary JSON file"""
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)
            
            # Convert to DataFrame and add extra columns
            df = pd.DataFrame([data])
            
            # Replace model name with alias if it exists in the data
            if 'model' in df.columns:
                df['model'] = self.model_alias

            # Add extra info columns
            for key, value in self.extra_info.items():
                df[key] = value
                
            return df
        except Exception as e:
            print(f"Error processing summary file {summary_file}: {e}")
            return None

    def process_individual_file(self, individual_file: Path, timestamp: int) -> Optional[pd.DataFrame]:
        """Process a single individual responses JSON file"""
        try:
            with open(individual_file, 'r') as f:
                data = json.load(f)
            
            # Convert to DataFrame and add timestamp
            df = pd.DataFrame(data)
            df['timestamp'] = timestamp
            
            return df
        except Exception as e:
            print(f"Error processing individual file {individual_file}: {e}")
            return None

    def convert_all(self):
        """Convert all JSON files to CSV"""
        summary_dfs = []
        individual_dfs = []
        
        result_dirs = self.find_result_directories()
        print(f"Found {len(result_dirs)} result directories for model {self.model}")
        
        for result_dir in result_dirs:
            # Find summary and individual files
            summary_files = list(result_dir.glob(f"{self.model}_*_summary.json"))
            individual_files = list(result_dir.glob(f"{self.model}_*_individual_responses.json"))
            
            if not summary_files or not individual_files:
                print(f"Missing files in directory {result_dir}")
                continue
                
            # Process summary file
            summary_df = self.process_summary_file(summary_files[0])
            if summary_df is not None:
                timestamp = summary_df['timestamp'].iloc[0]
                summary_dfs.append(summary_df)
                
                # Process individual file with corresponding timestamp
                individual_df = self.process_individual_file(individual_files[0], timestamp)
                if individual_df is not None:
                    individual_dfs.append(individual_df)
        
        # Combine and save all results
        if summary_dfs:
            combined_summary = pd.concat(summary_dfs, ignore_index=True)
            combined_summary.to_csv(
                self.output_dir / f"results_{self.model}_summary.csv",
                index=False
            )
            print(f"Saved combined summary to results_{self.model}_summary.csv")
        
        if individual_dfs:
            combined_individual = pd.concat(individual_dfs, ignore_index=True)
            combined_individual.to_csv(
                self.output_dir / f"results_{self.model}_individual_summary.csv",
                index=False
            )
            print(f"Saved combined individual responses to results_{self.model}_individual_summary.csv")

def parse_args():
    parser = argparse.ArgumentParser(description='Convert benchmark results to CSV format')
    parser.add_argument('--model', type=str, required=True, help='Model name to process')
    parser.add_argument('--model-alias', type=str, help='Alias name for the model (optional)')
    parser.add_argument('--data-type', type=str, required=True, help='Data type')
    parser.add_argument('--driver', type=str, required=True, help='Driver name')
    parser.add_argument('--driver-version', type=str, required=True, help='Driver version')
    parser.add_argument('--backend', type=str, required=True, help='Backend name, e.g. musa, cuda')
    parser.add_argument('--backend-version', type=str, required=True, help='Backend version')
    parser.add_argument('--engine', type=str, required=True, help='Engine name, e.g. mtt, trt-llm')
    parser.add_argument('--engine-version', type=str, required=True, help='Engine version')
    parser.add_argument('--serving', type=str, required=True, help='Serving name, e.g. vllm, triton')
    parser.add_argument('--serving-version', type=str, required=True, help='Serving version')
    parser.add_argument('--gpu', type=str, required=True, help='GPU model, e.g. S4000, A100')
    parser.add_argument('--gpu-num', type=int, required=True, help='Number of GPUs')
    parser.add_argument('--tp', type=int, required=True, help='Tensor Parallelism')
    parser.add_argument('--pp', type=int, help='Pipeline Parallelism')
    parser.add_argument('--dp', type=int, help='Data Parallel')
    parser.add_argument('--ep', type=int, help='Expert Parallel')
    parser.add_argument('--base-dir', type=str, default='result_outputs', 
                       help='Base directory containing result outputs')
    parser.add_argument('--source', type=str, default='in-house_benchmark', 
                       help='Data source')
    return parser.parse_args()

def main():
    args = parse_args()
    
    converter = DataConverter(
        model=args.model,
        data_type=args.data_type,
        driver=args.driver,
        driver_version=args.driver_version,
        backend=args.backend,
        backend_version=args.backend_version,
        engine=args.engine,
        engine_version=args.engine_version,
        serving=args.serving,
        serving_version=args.serving_version,
        gpu=args.gpu,
        gpu_num=args.gpu_num,
        dp=args.dp,
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        model_alias=args.model_alias,  # Pass the model alias
        base_dir=args.base_dir,
        source=args.source,
    )
    
    converter.convert_all()

if __name__ == "__main__":
    main()

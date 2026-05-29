import argparse
import json
import os
import numpy as np
import pandas as pd
from typing import List, Optional

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Concept Configuration from CSV")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to the input metadata CSV.")
    parser.add_argument('--output_path', type=str, default='concept_config.json',
                        help="Path to save the generated JSON/YAML config (default: concept_config.json).")
    parser.add_argument('--ignore_cols', type=str, default='',
                        help="Comma-separated list of column names to completely ignore.")
    return parser.parse_args()

def convert_val(val):
    """Convert numpy types to native Python types for JSON/YAML serialization."""
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    elif isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    elif isinstance(val, np.bool_):
        return bool(val)
    elif pd.isna(val):
        return None
    return val

def main():
    args = parse_args()
    
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"Input CSV not found at: {args.csv_path}")

    # Read CSV
    df = pd.read_csv(args.csv_path)
    
    # Parse ignored columns
    ignore_cols = {c.strip() for c in args.ignore_cols.split(',')} if args.ignore_cols else set()
    
    config = {}
    
    for col in df.columns:
        if col in ignore_cols:
            continue
            
        col_data = df[col].dropna()
        if col_data.empty:
            continue
            
        unique_vals = df[col].unique()
        # Filter out NaN from unique values
        unique_vals = [v for v in unique_vals if not pd.isna(v)]
        num_unique = len(unique_vals)
        dtype = df[col].dtype
        
        # Heuristics for classification
        is_categorical = False
        if dtype == 'object' or dtype == 'bool' or isinstance(dtype, pd.StringDtype):
            is_categorical = True
        elif np.issubdtype(dtype, np.number) and num_unique < 15:
            is_categorical = True
            
        if is_categorical:
            # Categorical: Sort and convert unique values to native Python types
            classes = sorted([convert_val(v) for v in unique_vals])
            config[col] = {
                "type": "categorical",
                "classes": classes
            }
        else:
            # Numerical: Find min/max and convert to native Python types
            min_val = convert_val(col_data.min())
            max_val = convert_val(col_data.max())
            config[col] = {
                "type": "numerical",
                "min": min_val,
                "max": max_val
            }
            
    # Serialize output
    output_ext = os.path.splitext(args.output_path)[1].lower()
    
    if output_ext in ('.yaml', '.yml'):
        import yaml
        with open(args.output_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Successfully generated YAML configuration at: {args.output_path}")
    else:
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print(f"Successfully generated JSON configuration at: {args.output_path}")

if __name__ == "__main__":
    main()

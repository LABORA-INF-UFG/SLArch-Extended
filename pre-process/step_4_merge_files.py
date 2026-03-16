import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent         
PROJECT_ROOT = SCRIPT_DIR.parent                     

PRE_PROCESS_DIR = PROJECT_ROOT / "pre-process"
POST_PROCESS_DIR = PROJECT_ROOT / "post-process"

file1 = PRE_PROCESS_DIR / "ue_net_statistics.csv"
file2 = PRE_PROCESS_DIR / "ue_data_characteristics.csv"

output_file = POST_PROCESS_DIR / "ue_complete_select_params.csv"

def merge_files(file1_path, file2_path, output_path):
    df1 = pd.read_csv(file1_path, sep=';', decimal='.', dtype={'UE_ID': str})
    df1['UE_ID'] = df1['UE_ID'].astype(float).astype(int)
    df1['SliceType'] = df1['SliceType'].str.strip()

    df2 = pd.read_csv(file2_path, sep=',', dtype={'UE_ID': int})
    df2['SliceType'] = df2['SliceType'].str.strip()

    required_cols = ['UE_ID', 'SliceType', 'Train_Size', 'Test_Size', 'Emd_Value']
    if not all(col in df2.columns for col in required_cols):
        raise ValueError("Second file missing required columns. Found: " + ", ".join(df2.columns))

    merged = pd.merge(df1, df2[required_cols], on=['UE_ID', 'SliceType'], how='left')
    
    missing = merged[merged['Train_Size'].isna()]
    if not missing.empty:
        print(f"Warning: {len(missing)} rows from first file had no matching UE_ID and SliceType in second file.")
        print("Missing rows (first few):")
        print(missing[['UE_ID', 'SliceType']].head())

    merged.to_csv(output_path, sep=',', decimal='.', index=False, float_format='%.10f')
    # print(f"Merged file saved to: {output_path}")

if __name__ == "__main__":
    POST_PROCESS_DIR.mkdir(parents=True, exist_ok=True)

    if not file1.exists():
        print(f"Error: {file1} not found. Please ensure it exists in the pre-process directory.")
    elif not file2.exists():
        print(f"Error: {file2} not found. Please ensure it exists in the pre-process directory.")
    else:
        merge_files(file1, file2, output_file)

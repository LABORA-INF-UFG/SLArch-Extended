import pandas as pd
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent     
PROJECT_ROOT = SCRIPT_DIR.parent                

NS3_OUTPUT_DIR = PROJECT_ROOT.parent / "ns-3-dev" / "nr_slicing_ul_res"

output_file = SCRIPT_DIR / "nr_slicing_ul_all_batches.csv"

SLICE_COL = "SliceType"
ITER_COL  = "Iteration"
UEID_COL  = "GlobalUEId"

SLICE_ORDER = ["URLLC", "eMBB", "mMTC"]

EXPECTED_COLS = 26

input_files = []
missing_files = []

for i in range(0, 15):
    fname = f"batch_{i}.csv"
    fpath = NS3_OUTPUT_DIR / fname
    if fpath.exists():
        input_files.append(str(fpath))
    else:
        missing_files.append(str(fpath))

if missing_files:
    print("Missing batch files:")
    for f in missing_files:
        print(f)

print(f"\nTotal batch files found: {len(input_files)}")

if not input_files:
    raise RuntimeError("No batch files found. Aborting.")

dfs = []
total_dropped_rows = 0

b_count = 0
for file in input_files:
    print(f"\nReading batch {b_count}")
    b_count += 1
    
    with open(file, "r") as f:
        raw_rows = sum(1 for _ in f) - 1

    df = pd.read_csv(
        file,
        sep=",",                 
        engine="python",         
        comment="#",             
        on_bad_lines="skip"      
    )

    parsed_rows = len(df)
    dropped = raw_rows - parsed_rows
    total_dropped_rows += max(dropped, 0)

    print(f"  Parsed rows  : {parsed_rows}")
    print(f"  Dropped rows : {max(dropped, 0)}")

    if df.shape[1] != EXPECTED_COLS:
        raise ValueError(
            f"Column mismatch in {file}: "
            f"expected {EXPECTED_COLS}, got {df.shape[1]}"
        )

    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].astype("float64")

    dfs.append(df)
    
combined_df = pd.concat(dfs, ignore_index=True)

print(f"\nTotal rows processed: {len(combined_df)}")
print(f"Total dropped rows overall: {total_dropped_rows}")
print("");
required_cols = {SLICE_COL, ITER_COL, UEID_COL}
missing_cols = required_cols - set(combined_df.columns)

if missing_cols:
    raise RuntimeError(f"Missing required columns: {missing_cols}")

combined_df[SLICE_COL] = pd.Categorical(
    combined_df[SLICE_COL],
    categories=SLICE_ORDER,
    ordered=True
)

combined_df_sorted = (
    combined_df
    .sort_values(by=[SLICE_COL, ITER_COL, UEID_COL])
    .reset_index(drop=True)
)

combined_df_sorted.to_csv(
    output_file,
    sep=",",
    index=False,
    float_format="%.6f",   
    decimal='.'           
)


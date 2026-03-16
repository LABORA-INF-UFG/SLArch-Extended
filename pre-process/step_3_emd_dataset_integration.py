import os
import pickle
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent          
PROJECT_ROOT = SCRIPT_DIR.parent                       
PRE_DIR = PROJECT_ROOT / "pre-process"
DATASET_DIR = PROJECT_ROOT / "split" / "ds-15"

emd_values = [0.171154, 0.42217, 3.052985, 0.61236, 5.855378, 20.35, 1.641502, 0.253991, 0.11791, 0.772897,
              0.50202, 0.670259, 1.881767, 0.203246, 24.75, 1.306283, 14.85, 5.142348, 6.3861, 17.05,
              0.328487, 1.332857, 0.496914, 1.672414, 0.800915, 0.204854, 0.595082, 5.084685, 11.0213, 20.35,
              1.43, 20.35, 0.909041, 14.85, 1.267467, 24.75, 9.821875, 0.314872, 0.386364, 11.256243,
              14.85, 1.733721, 13.75, 0.393724, 17.05, 7.743491, 0.69375, 0.1625, 5.053846, 1.254303,
              0.538961, 14.85, 0.854185, 6.470661, 7.027847, 1.624942, 13.75, 0.289918, 12.288889, 7.012048,
              0.208868, 0.175047, 1.319907, 0.191026, 0.560371, 3.025974, 17.05, 0.751121, 0.189349, 4.392308,
              0.69292, 0.263536, 0.331943, 20.35, 1.842704, 17.05, 0.452495, 17.05, 13.75, 0.58871, 1.64186,
              0.359371, 1.183444, 1.539644, 20.35, 11.716872, 0.101898, 1.430206, 0.528777, 5.280431, 24.75,
              6.02603, 14.85, 0.833861, 24.75, 12.104887, 0.209953, 1.252669, 20.35, 13.75, 3.232258,
              0.417881, 17.05, 6.21513, 1.774254, 0.864557, 0.570213, 7.809585, 7.021889, 1.61236,
              2.62125, 1.106154, 7.206724, 1.349511, 3.080229, 0.361635, 1.759582, 7.138889, 17.149038,
              7.464228, 24.75, 1.677907, 9.748707, 3.789759, 1.78566, 13.75, 5.415455, 3.855785,
              11.786792, 0.186835, 0.948991, 13.75, 4.526431, 0.245528, 0.162694, 3.370626, 0.537046,
              0.262676, 1.899113, 15.359098, 0.168692, 14.85, 0.952018, 0.535556, 2.023529,
              0.699329, 24.75, 0.57205, 0.915966, 1.766667]

def main():
    train_files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith("_train.pickle")])
    test_files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith("_test.pickle")])

    if len(train_files) != len(test_files) or len(train_files) != len(emd_values):
        raise ValueError("Mismatch in number of files or EMD values")

    records = []
    for fname in train_files:
        client_num = int(fname.split("_")[0])     
        ue_id = client_num - 1                      

        train_df = pd.read_pickle(os.path.join(DATASET_DIR, fname))
        train_size = len(train_df)

        test_path = os.path.join(DATASET_DIR, f"{client_num}_test.pickle")
        test_size = 0
        if os.path.exists(test_path):
            test_df = pd.read_pickle(test_path)
            test_size = len(test_df)

        records.append({
            "UE_ID": ue_id,
            "Train_Size": train_size,
            "Test_Size": test_size
        })

    df_base = pd.DataFrame(records)

    df_sorted = df_base.sort_values('Train_Size', ascending=False).reset_index(drop=True)

    emd_sorted = sorted(emd_values)                  
    df_sorted['Emd_Value'] = emd_sorted[:len(df_sorted)]

    df_map = df_sorted.sort_values('UE_ID').reset_index(drop=True)

    stats_path = PRE_DIR / "ue_net_statistics.csv"
    if not stats_path.exists():
        raise FileNotFoundError(f"Required statistics file not found: {stats_path}")

    df_stats = pd.read_csv(stats_path, sep=';')
    df_stats['UE_ID'] = pd.to_numeric(df_stats['UE_ID'], errors='coerce')

    df_stats = df_stats.drop_duplicates(subset=['UE_ID'])

    df_map = df_map.merge(
        df_stats[['UE_ID', 'SliceType', 'SINR_Mean_dB']],
        on='UE_ID',
        how='left'
    )

    if 'SINR_Mean_dB_x' in df_map.columns:
        df_map['SINR_Mean_dB'] = df_map['SINR_Mean_dB_x']
        df_map.drop(columns=['SINR_Mean_dB_x', 'SINR_Mean_dB_y'], inplace=True, errors='ignore')
    elif 'SINR_Mean_dB' not in df_map.columns:
        df_map['SINR_Mean_dB'] = None

    final_cols = ['UE_ID', 'SliceType', 'Train_Size', 'Test_Size', 'Emd_Value', 'SINR_Mean_dB']
    df_map = df_map[final_cols]

    out_path = PRE_DIR / 'ue_data_characteristics.csv'
    df_map.to_csv(out_path, index=False)
    # print(f"✓ Saved: {out_path}")
    print("\nRunning EMD - Datasets Integration...\n");
if __name__ == "__main__":
    main()

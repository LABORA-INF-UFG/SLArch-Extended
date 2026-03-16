import pandas as pd
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent          
PROJECT_ROOT = SCRIPT_DIR.parent                      

INPUT_FILE = SCRIPT_DIR / "nr_slicing_ul_all_batches.csv"
OUTPUT_FILE = SCRIPT_DIR / "ue_net_statistics.csv"

CSV_SEP = ";"              
CSV_DECIMAL = "."          
CSV_ENCODING = "utf-8-sig"

def force_float(df: pd.DataFrame) -> pd.DataFrame:
    
    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".")   
            .astype(float)          
        )

    return df

def generate_ue_statistics():
    df = pd.read_csv(
        INPUT_FILE,
        sep=",",
        decimal=".",
        low_memory=False
    )

    df['GlobalUEId'] = pd.to_numeric(df['GlobalUEId'], errors='coerce')

    unique_ues = sorted(df['GlobalUEId'].dropna().unique())
    print(f"\nProcessing {len(unique_ues)} unique UEs...\n")

    results = []

    for ue_id in unique_ues:
        ue_data = df[df['GlobalUEId'] == ue_id]

        stats = {
            "UE_ID": int(ue_id),
            "Iterations": int(len(ue_data))
        }

        sinr_values = ue_data['SINR_dB'].dropna()
        stats['SINR_Mean_dB'] = sinr_values.mean() if len(sinr_values) > 0 else np.nan
        stats['SINR_Std_dB']  = sinr_values.std()  if len(sinr_values) > 0 else np.nan

        thr_values = ue_data['Throughput(Mbps)'].dropna()
        stats['Throughput_Mean_Mbps'] = thr_values.mean() if len(thr_values) > 0 else np.nan
        stats['Throughput_Std_Mbps']  = thr_values.std()  if len(thr_values) > 0 else np.nan

        plr_values = ue_data['PacketLossRatio(%)'].dropna()
        stats['PacketLossRatio_Mean_%'] = plr_values.mean() if len(plr_values) > 0 else np.nan
        stats['PacketLossRatio_Std_%']  = plr_values.std()  if len(plr_values) > 0 else np.nan

        delay_values = ue_data['AvgDelay(s)'].dropna()
        if len(delay_values) > 0:
            delay_ms = delay_values * 1000
            stats['Delay_Mean_ms'] = delay_ms.mean()
            stats['Delay_Std_ms']  = delay_ms.std()
        else:
            stats['Delay_Mean_ms'] = np.nan
            stats['Delay_Std_ms']  = np.nan

        jitter_values = ue_data['AvgJitter(s)'].dropna()
        if len(jitter_values) > 0:
            jitter_ms = jitter_values * 1000
            stats['Jitter_Mean_ms'] = jitter_ms.mean()
            stats['Jitter_Std_ms']  = jitter_ms.std()
        else:
            stats['Jitter_Mean_ms'] = np.nan
            stats['Jitter_Std_ms']  = np.nan

        energy_values = ue_data['TotalEnergy_J'].dropna()
        stats['Energy_Mean_J'] = energy_values.mean() if len(energy_values) > 0 else np.nan
        stats['Energy_Std_J']  = energy_values.std()  if len(energy_values) > 0 else np.nan

        if 'Distance_m' in ue_data.columns:
            dist_values = ue_data['Distance_m'].dropna()
            stats['Distance_Mean_m'] = dist_values.mean() if len(dist_values) > 0 else np.nan
            stats['Distance_Std_m']  = dist_values.std()  if len(dist_values) > 0 else np.nan
        else:
            stats['Distance_Mean_m'] = np.nan
            stats['Distance_Std_m']  = np.nan

        if 'SliceType' in ue_data.columns:
            stats['SliceType'] = ue_data['SliceType'].iloc[0]

        if 'ServingGnbId' in ue_data.columns:
            stats['ServingGnbId'] = ue_data['ServingGnbId'].iloc[0]

        results.append(stats)

    results_df = pd.DataFrame(results)

    results_df = force_float(results_df)

    column_order = [
        'UE_ID', 'Iterations', 'SliceType', 'ServingGnbId',
        'SINR_Mean_dB', 'SINR_Std_dB',
        'Throughput_Mean_Mbps', 'Throughput_Std_Mbps',
        'PacketLossRatio_Mean_%', 'PacketLossRatio_Std_%',
        'Delay_Mean_ms', 'Delay_Std_ms',
        'Jitter_Mean_ms', 'Jitter_Std_ms',
        'Energy_Mean_J', 'Energy_Std_J',
        'Distance_Mean_m', 'Distance_Std_m'
    ]

    results_df = results_df[[c for c in column_order if c in results_df.columns]]
    
    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
        sep=CSV_SEP,              # ;
        decimal=CSV_DECIMAL,      # .
        float_format="%.10f",
        encoding=CSV_ENCODING
    )
    
    return results_df

if __name__ == "__main__":

    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found!")
        exit(1)

    main_results = generate_ue_statistics()

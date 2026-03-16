import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random
from scipy import stats

# Determine project root based on script location
SCRIPT_DIR = Path(__file__).resolve().parent          
PROJECT_ROOT = SCRIPT_DIR.parent                       
POST_DIR = PROJECT_ROOT / "post-process"
PLOTS_BASE = PROJECT_ROOT / "simulation_plots"    

random.seed(42)
np.random.seed(42)

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 0,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'figure.titlesize': 0,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'figure.autolayout': True,
    'axes.linewidth': 2.0,
    'grid.linewidth': 1.0,
    'lines.linewidth': 3.0,
    'lines.markersize': 10,
    'errorbar.capsize': 10,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
})

def format_3dec(value):
    if pd.isna(value):
        return value
    try:
        return round(float(value), 3)
    except (ValueError, TypeError):
        return value

def load_ue_statistics():
    file_path = POST_DIR / 'ue_complete_select_params.csv'
    
    if not file_path.exists():
        print(f"Error: {file_path} not found!")
        print("Please run the statistics generator first to create this file.")
        return None
    
    df = pd.read_csv(file_path)
    # print(f"Loaded statistics for {len(df)} UEs")
    
    if 'Emd_Value' not in df.columns:
        print("Warning: Emd_Value column not found in data! DataQual scenarios will not work properly.")
    
    if 'SliceType' in df.columns:
        slice_counts = df['SliceType'].value_counts()
        print("\nSlice Type Distribution:")
        for slice_type, count in slice_counts.items():
            print(f"  {slice_type}: {count} UEs ({count/len(df)*100:.1f}%)")
    
    return df

def analyze_slice_performance(df):
    print("\n" + "="*70)
    print("BASELINE SLICE PERFORMANCE ANALYSIS")
    print("="*70)
    
    if 'SliceType' not in df.columns:
        print("SliceType column not found in data!")
        return
    
    slice_types = df['SliceType'].unique()
    
    for slice_type in slice_types:
        slice_data = df[df['SliceType'] == slice_type]
        print(f"\n{slice_type} Slice ({len(slice_data)} UEs):")
        print(f"  SINR: {slice_data['SINR_Mean_dB'].mean():.3f} ± {slice_data['SINR_Std_dB'].mean():.3f} dB")
        print(f"  Throughput: {slice_data['Throughput_Mean_Mbps'].mean():.3f} ± {slice_data['Throughput_Std_Mbps'].mean():.3f} Mbps")
        print(f"  PLR: {slice_data['PacketLossRatio_Mean_%'].mean():.3f} ± {slice_data['PacketLossRatio_Std_%'].mean():.3f} %")
        print(f"  Delay: {slice_data['Delay_Mean_ms'].mean():.3f} ± {slice_data['Delay_Std_ms'].mean():.3f} ms")
        print(f"  Jitter: {slice_data['Jitter_Mean_ms'].mean():.3f} ± {slice_data['Jitter_Std_ms'].mean():.3f} ms")
        print(f"  Energy: {slice_data['Energy_Mean_J'].mean():.3f} ± {slice_data['Energy_Std_J'].mean():.3f} J")
        
        if 'Emd_Value' in slice_data.columns:
            print(f"  EMD: {slice_data['Emd_Value'].mean():.3f} ± {slice_data['Emd_Value'].std():.3f}")

def per_slice_random_selection(df, num_clients=10, iterations=100):
    print("\n" + "="*70)
    print("PER-SLICE RANDOM SELECTION (100 ITERATIONS)")
    print(f"For EACH slice: Select {num_clients} clients randomly")
    print(f"Repeat for {iterations} iterations (per slice)")
    print("="*70)
    
    if 'SliceType' not in df.columns:
        print("Error: SliceType column not found!")
        return None
    
    slice_types = df['SliceType'].unique()
    all_results = {}
    
    for slice_type in slice_types:
        print(f"\nProcessing {slice_type} slice...")
        slice_df = df[df['SliceType'] == slice_type].copy()
        slice_ue_ids = slice_df['UE_ID'].unique().tolist()
        
        if len(slice_ue_ids) < num_clients:
            print(f"  Warning: Only {len(slice_ue_ids)} UEs in {slice_type} slice, cannot select {num_clients}")
            continue
        
        results = []
        for i in range(iterations):
            selected_ids = random.sample(slice_ue_ids, num_clients)
            selected_data = slice_df[slice_df['UE_ID'].isin(selected_ids)]
            
            iteration_results = {
                'Slice_Type': slice_type,
                'Iteration': i + 1,
                'Scenario': 'Random',
                'Num_Clients': len(selected_data),
                'SINR_Mean_dB': format_3dec(selected_data['SINR_Mean_dB'].mean()),
                'SINR_Std_dB': format_3dec(selected_data['SINR_Std_dB'].mean()),
                'Throughput_Mean_Mbps': format_3dec(selected_data['Throughput_Mean_Mbps'].mean()),
                'Throughput_Std_Mbps': format_3dec(selected_data['Throughput_Std_Mbps'].mean()),
                'PacketLossRatio_Mean_%': format_3dec(selected_data['PacketLossRatio_Mean_%'].mean()),
                'PacketLossRatio_Std_%': format_3dec(selected_data['PacketLossRatio_Std_%'].mean()),
                'Delay_Mean_ms': format_3dec(selected_data['Delay_Mean_ms'].mean()),
                'Delay_Std_ms': format_3dec(selected_data['Delay_Std_ms'].mean()),
                'Jitter_Mean_ms': format_3dec(selected_data['Jitter_Mean_ms'].mean()),
                'Jitter_Std_ms': format_3dec(selected_data['Jitter_Std_ms'].mean()),
                'Energy_Mean_J': format_3dec(selected_data['Energy_Mean_J'].mean()),
                'Energy_Std_J': format_3dec(selected_data['Energy_Std_J'].mean())
            }
            
            if 'Emd_Value' in selected_data.columns:
                iteration_results['Emd_Mean'] = format_3dec(selected_data['Emd_Value'].mean())
                iteration_results['Emd_Std'] = format_3dec(selected_data['Emd_Value'].std())
            
            results.append(iteration_results)
        
        results_df = pd.DataFrame(results)
        all_results[slice_type] = results_df
        
        print(f"  {slice_type} slice averages (Random 10, {iterations} iterations):")
        print(f"    SINR: {results_df['SINR_Mean_dB'].mean():.3f} ± {results_df['SINR_Mean_dB'].std():.3f} dB")
        print(f"    Throughput: {results_df['Throughput_Mean_Mbps'].mean():.3f} ± {results_df['Throughput_Mean_Mbps'].std():.3f} Mbps")
        print(f"    Delay: {results_df['Delay_Mean_ms'].mean():.3f} ± {results_df['Delay_Mean_ms'].std():.3f} ms")
        
        if 'Emd_Mean' in results_df.columns:
            print(f"    EMD: {results_df['Emd_Mean'].mean():.3f} ± {results_df['Emd_Mean'].std():.3f}")
    
    return all_results

def per_slice_sinr_selection(df, pool_size=30, num_selected=10, iterations=100):
    print("\n" + "="*70)
    print("PER-SLICE SINR-BASED SELECTION (TOP 10 FROM POOL OF 30)")
    print(f"For EACH slice: Select {pool_size} clients randomly")
    print(f"From the pool, select best {num_selected} by SINR")
    print(f"Repeat for {iterations} iterations (per slice)")
    print("="*70)
    
    if 'SliceType' not in df.columns:
        print("Error: SliceType column not found!")
        return None
    
    slice_types = df['SliceType'].unique()
    all_results = {}
    
    for slice_type in slice_types:
        print(f"\nProcessing {slice_type} slice...")
        slice_df = df[df['SliceType'] == slice_type].copy()
        slice_ue_ids = slice_df['UE_ID'].unique().tolist()
        
        if len(slice_ue_ids) < pool_size:
            print(f"  Warning: Only {len(slice_ue_ids)} UEs in {slice_type} slice, cannot select pool of {pool_size}")
            continue
        
        results = []
        for i in range(iterations):
            pool_ids = random.sample(slice_ue_ids, pool_size)
            pool_data = slice_df[slice_df['UE_ID'].isin(pool_ids)].copy()
            
            pool_data_sorted = pool_data.sort_values('SINR_Mean_dB', ascending=False)
            selected_data = pool_data_sorted.head(num_selected)
            
            iteration_results = {
                'Slice_Type': slice_type,
                'Iteration': i + 1,
                'Scenario': 'ChQual',
                'Num_Clients': len(selected_data),
                'SINR_Mean_dB': format_3dec(selected_data['SINR_Mean_dB'].mean()),
                'SINR_Std_dB': format_3dec(selected_data['SINR_Std_dB'].mean()),
                'Throughput_Mean_Mbps': format_3dec(selected_data['Throughput_Mean_Mbps'].mean()),
                'Throughput_Std_Mbps': format_3dec(selected_data['Throughput_Std_Mbps'].mean()),
                'PacketLossRatio_Mean_%': format_3dec(selected_data['PacketLossRatio_Mean_%'].mean()),
                'PacketLossRatio_Std_%': format_3dec(selected_data['PacketLossRatio_Std_%'].mean()),
                'Delay_Mean_ms': format_3dec(selected_data['Delay_Mean_ms'].mean()),
                'Delay_Std_ms': format_3dec(selected_data['Delay_Std_ms'].mean()),
                'Jitter_Mean_ms': format_3dec(selected_data['Jitter_Mean_ms'].mean()),
                'Jitter_Std_ms': format_3dec(selected_data['Jitter_Std_ms'].mean()),
                'Energy_Mean_J': format_3dec(selected_data['Energy_Mean_J'].mean()),
                'Energy_Std_J': format_3dec(selected_data['Energy_Std_J'].mean())
            }
            
            if 'Emd_Value' in selected_data.columns:
                iteration_results['Emd_Mean'] = format_3dec(selected_data['Emd_Value'].mean())
                iteration_results['Emd_Std'] = format_3dec(selected_data['Emd_Value'].std())
            
            results.append(iteration_results)
        
        results_df = pd.DataFrame(results)
        all_results[slice_type] = results_df
        
        print(f"  {slice_type} slice averages (SINR-based 30→10, {iterations} iterations):")
        print(f"    SINR: {results_df['SINR_Mean_dB'].mean():.3f} ± {results_df['SINR_Mean_dB'].std():.3f} dB")
        print(f"    Throughput: {results_df['Throughput_Mean_Mbps'].mean():.3f} ± {results_df['Throughput_Mean_Mbps'].std():.3f} Mbps")
        print(f"    Delay: {results_df['Delay_Mean_ms'].mean():.3f} ± {results_df['Delay_Mean_ms'].std():.3f} ms")
        
        if 'Emd_Mean' in results_df.columns:
            print(f"    EMD: {results_df['Emd_Mean'].mean():.3f} ± {results_df['Emd_Mean'].std():.3f}")
    
    return all_results

def per_slice_data_quality_selection(df, pool_size=30, num_selected=10, iterations=100):
    print("\n" + "="*70)
    print("PER-SLICE DataQual (EMD) BASED SELECTION (TOP 10 FROM POOL OF 30)")
    print(f"For EACH slice: Select {pool_size} clients randomly")
    print(f"From the pool, select best {num_selected} with lowest EMD (lower is better)")
    print(f"Repeat for {iterations} iterations (per slice)")
    print("="*70)
    
    if 'SliceType' not in df.columns:
        print("Error: SliceType column not found!")
        return None
    
    if 'Emd_Value' not in df.columns:
        print("Error: Emd_Value column not found! Cannot run DataQual selection.")
        return None
    
    slice_types = df['SliceType'].unique()
    all_results = {}
    
    for slice_type in slice_types:
        print(f"\nProcessing {slice_type} slice...")
        slice_df = df[df['SliceType'] == slice_type].copy()
        slice_ue_ids = slice_df['UE_ID'].unique().tolist()
        
        if len(slice_ue_ids) < pool_size:
            print(f"  Warning: Only {len(slice_ue_ids)} UEs in {slice_type} slice, cannot select pool of {pool_size}")
            continue
        
        results = []
        for i in range(iterations):
            pool_ids = random.sample(slice_ue_ids, pool_size)
            pool_data = slice_df[slice_df['UE_ID'].isin(pool_ids)].copy()
            
            pool_data_sorted = pool_data.sort_values('Emd_Value', ascending=True)
            selected_data = pool_data_sorted.head(num_selected)
            
            iteration_results = {
                'Slice_Type': slice_type,
                'Iteration': i + 1,
                'Scenario': 'DataQual',
                'Num_Clients': len(selected_data),
                'SINR_Mean_dB': format_3dec(selected_data['SINR_Mean_dB'].mean()),
                'SINR_Std_dB': format_3dec(selected_data['SINR_Std_dB'].mean()),
                'Throughput_Mean_Mbps': format_3dec(selected_data['Throughput_Mean_Mbps'].mean()),
                'Throughput_Std_Mbps': format_3dec(selected_data['Throughput_Std_Mbps'].mean()),
                'PacketLossRatio_Mean_%': format_3dec(selected_data['PacketLossRatio_Mean_%'].mean()),
                'PacketLossRatio_Std_%': format_3dec(selected_data['PacketLossRatio_Std_%'].mean()),
                'Delay_Mean_ms': format_3dec(selected_data['Delay_Mean_ms'].mean()),
                'Delay_Std_ms': format_3dec(selected_data['Delay_Std_ms'].mean()),
                'Jitter_Mean_ms': format_3dec(selected_data['Jitter_Mean_ms'].mean()),
                'Jitter_Std_ms': format_3dec(selected_data['Jitter_Std_ms'].mean()),
                'Energy_Mean_J': format_3dec(selected_data['Energy_Mean_J'].mean()),
                'Energy_Std_J': format_3dec(selected_data['Energy_Std_J'].mean()),
                'Emd_Mean': format_3dec(selected_data['Emd_Value'].mean()),
                'Emd_Std': format_3dec(selected_data['Emd_Value'].std()),
                'Emd_Min': format_3dec(selected_data['Emd_Value'].min()),
                'Emd_Max': format_3dec(selected_data['Emd_Value'].max())
            }
            
            results.append(iteration_results)
        
        results_df = pd.DataFrame(results)
        all_results[slice_type] = results_df
        
        print(f"  {slice_type} slice averages (DataQual 30→10, {iterations} iterations):")
        print(f"    SINR: {results_df['SINR_Mean_dB'].mean():.3f} ± {results_df['SINR_Mean_dB'].std():.3f} dB")
        print(f"    Throughput: {results_df['Throughput_Mean_Mbps'].mean():.3f} ± {results_df['Throughput_Mean_Mbps'].std():.3f} Mbps")
        print(f"    Delay: {results_df['Delay_Mean_ms'].mean():.3f} ± {results_df['Delay_Mean_ms'].std():.3f} ms")
        print(f"    EMD: {results_df['Emd_Mean'].mean():.3f} ± {results_df['Emd_Mean'].std():.3f}")
        print(f"    EMD Range: {results_df['Emd_Min'].mean():.3f} to {results_df['Emd_Max'].mean():.3f}")
    
    return all_results

def per_slice_combined_quality_selection(df, pool_size=30, num_selected=10, iterations=100):
    print("\n" + "="*70)
    print("PER-SLICE COMBINED ChDataQual SELECTION (TOP 10 FROM POOL OF 30)")
    print(f"For EACH slice: Select {pool_size} clients randomly")
    print(f"From the pool, select best {num_selected} using combined score (SINR normalized - EMD normalized)")
    print(f"Repeat for {iterations} iterations (per slice)")
    print("="*70)
    
    if 'SliceType' not in df.columns:
        print("Error: SliceType column not found!")
        return None
    
    if 'Emd_Value' not in df.columns:
        print("Error: Emd_Value column not found! Cannot run Combined Quality selection.")
        return None
    
    slice_types = df['SliceType'].unique()
    all_results = {}
    
    for slice_type in slice_types:
        print(f"\nProcessing {slice_type} slice...")
        slice_df = df[df['SliceType'] == slice_type].copy()
        slice_ue_ids = slice_df['UE_ID'].unique().tolist()
        
        if len(slice_ue_ids) < pool_size:
            print(f"  Warning: Only {len(slice_ue_ids)} UEs in {slice_type} slice, cannot select pool of {pool_size}")
            continue
        
        results = []
        for i in range(iterations):
            pool_ids = random.sample(slice_ue_ids, pool_size)
            pool_data = slice_df[slice_df['UE_ID'].isin(pool_ids)].copy()
            
            min_sinr = pool_data['SINR_Mean_dB'].min()
            max_sinr = pool_data['SINR_Mean_dB'].max()
            min_emd = pool_data['Emd_Value'].min()
            max_emd = pool_data['Emd_Value'].max()
            
            sinr_range = max_sinr - min_sinr if max_sinr > min_sinr else 1
            emd_range = max_emd - min_emd if max_emd > min_emd else 1
            
            pool_data['SINR_Norm'] = (pool_data['SINR_Mean_dB'] - min_sinr) / sinr_range
            pool_data['EMD_Norm'] = 1 - ((pool_data['Emd_Value'] - min_emd) / emd_range)
            
            pool_data['Combined_Score'] = pool_data['SINR_Norm'] + pool_data['EMD_Norm']
            
            pool_data_sorted = pool_data.sort_values('Combined_Score', ascending=False)
            selected_data = pool_data_sorted.head(num_selected)
            
            iteration_results = {
                'Slice_Type': slice_type,
                'Iteration': i + 1,
                'Scenario': 'ChDataQual',
                'Num_Clients': len(selected_data),
                'SINR_Mean_dB': format_3dec(selected_data['SINR_Mean_dB'].mean()),
                'SINR_Std_dB': format_3dec(selected_data['SINR_Std_dB'].mean()),
                'Throughput_Mean_Mbps': format_3dec(selected_data['Throughput_Mean_Mbps'].mean()),
                'Throughput_Std_Mbps': format_3dec(selected_data['Throughput_Std_Mbps'].mean()),
                'PacketLossRatio_Mean_%': format_3dec(selected_data['PacketLossRatio_Mean_%'].mean()),
                'PacketLossRatio_Std_%': format_3dec(selected_data['PacketLossRatio_Std_%'].mean()),
                'Delay_Mean_ms': format_3dec(selected_data['Delay_Mean_ms'].mean()),
                'Delay_Std_ms': format_3dec(selected_data['Delay_Std_ms'].mean()),
                'Jitter_Mean_ms': format_3dec(selected_data['Jitter_Mean_ms'].mean()),
                'Jitter_Std_ms': format_3dec(selected_data['Jitter_Std_ms'].mean()),
                'Energy_Mean_J': format_3dec(selected_data['Energy_Mean_J'].mean()),
                'Energy_Std_J': format_3dec(selected_data['Energy_Std_J'].mean()),
                'Emd_Mean': format_3dec(selected_data['Emd_Value'].mean()),
                'Emd_Std': format_3dec(selected_data['Emd_Value'].std()),
                'Emd_Min': format_3dec(selected_data['Emd_Value'].min()),
                'Emd_Max': format_3dec(selected_data['Emd_Value'].max())
            }
            
            results.append(iteration_results)
        
        results_df = pd.DataFrame(results)
        all_results[slice_type] = results_df
        
        print(f"  {slice_type} slice averages (Combined 30→10, {iterations} iterations):")
        print(f"    SINR: {results_df['SINR_Mean_dB'].mean():.3f} ± {results_df['SINR_Mean_dB'].std():.3f} dB")
        print(f"    Throughput: {results_df['Throughput_Mean_Mbps'].mean():.3f} ± {results_df['Throughput_Mean_Mbps'].std():.3f} Mbps")
        print(f"    Delay: {results_df['Delay_Mean_ms'].mean():.3f} ± {results_df['Delay_Mean_ms'].std():.3f} ms")
        print(f"    EMD: {results_df['Emd_Mean'].mean():.3f} ± {results_df['Emd_Mean'].std():.3f}")
    
    return all_results

def add_confidence_interval_shadow(ax, x_data, y_data, confidence=0.95, color='gray', alpha=0.3):
    if len(y_data) < 2:
        return
    
    y_mean = np.mean(y_data, axis=0)
    y_std = np.std(y_data, axis=0)
    
    n = len(y_data)
    if n > 1:
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin_error = t_value * y_std / np.sqrt(n)
        
        lower_bound = y_mean - margin_error
        upper_bound = y_mean + margin_error
        
        ax.fill_between(x_data, lower_bound, upper_bound, 
                       color=color, alpha=alpha, linewidth=0,
                       label=f'{int(confidence*100)}% CI')
    
    return y_mean

def create_per_slice_comparison_plots(random_results_dict, sinr_results_dict, data_quality_results_dict, combined_results_dict):
    print("\n" + "="*70)
    print("CREATING PER-SLICE COMPARISON PLOTS")
    print("="*70)
    
    plots_dir = PLOTS_BASE / 'net_plots' / 'per_slice_comparison_plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    individual_dir = PLOTS_BASE / 'net_plots' / 'per_slice_individual_plots'
    individual_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("colorblind")
    
    slice_types = list(random_results_dict.keys())
    
    metrics_config = [
        ('SINR_Mean_dB', 'SINR (dB)', 'SINR'),
        ('Throughput_Mean_Mbps', 'Throughput (Mbps)', 'Throughput'),
        ('PacketLossRatio_Mean_%', 'Packet Loss Ratio (%)', 'Packet Loss'),
        ('Delay_Mean_ms', 'Delay (ms)', 'Delay'),
        ('Jitter_Mean_ms', 'Jitter (ms)', 'Jitter'),
        ('Energy_Mean_J', 'Energy (J)', 'Energy')
    ]
    
    if 'Emd_Mean' in list(random_results_dict.values())[0].columns:
        metrics_config.append(('Emd_Mean', 'DataQual', 'EMD'))
    
    scenario_colors = {
        'Random': '#d62728', 
        'ChQual': '#ff7f0e',
        'DataQual': '#1f77b4',
        'ChDataQual': '#2ca02c'
    }
    
    ci_colors = scenario_colors.copy()
    
    for slice_type in slice_types:
        print(f"\nCreating plots for {slice_type} slice...")
        
        random_results = random_results_dict[slice_type]
        sinr_results = sinr_results_dict[slice_type]
        data_quality_results = data_quality_results_dict.get(slice_type, pd.DataFrame())
        combined_results = combined_results_dict.get(slice_type, pd.DataFrame())
        
        for metric, ylabel, metric_name in metrics_config:
            fig, ax = plt.subplots(figsize=(6, 5))
            
            iterations = np.arange(1, 101)
            
            results_dict = {
                'Random': random_results,
                'ChQual': sinr_results,
                'DataQual': data_quality_results,
                'ChDataQual': combined_results
            }
            
            for scenario, results in results_dict.items():
                if results.empty or metric not in results.columns:
                    continue
                    
                scenario_data = []
                for i in range(1, 101):
                    iter_data = results[results['Iteration'] == i][metric].values
                    if len(iter_data) > 0:
                        scenario_data.append(iter_data[0])
                
                if len(scenario_data) == 100:
                    window_size = 10
                    scenario_array = np.array(scenario_data)
                    
                    scenario_mean = pd.Series(scenario_array).rolling(window=window_size, center=True).mean().values
                    scenario_std = pd.Series(scenario_array).rolling(window=window_size, center=True).std().values
                    
                    scenario_mean[:window_size//2] = scenario_mean[window_size//2]
                    scenario_mean[-window_size//2:] = scenario_mean[-window_size//2-1]
                    scenario_std[:window_size//2] = scenario_std[window_size//2]
                    scenario_std[-window_size//2:] = scenario_std[-window_size//2-1]
                    
                    ax.plot(iterations, scenario_mean, 
                           linewidth=3.0,
                           color=scenario_colors[scenario],
                           label=scenario,
                           alpha=0.8)
                    
                    ax.fill_between(iterations, 
                                   scenario_mean - 1.96*scenario_std/np.sqrt(window_size),
                                   scenario_mean + 1.96*scenario_std/np.sqrt(window_size),
                                   color=ci_colors[scenario], alpha=0.2, linewidth=0)
            
            ax.set_xlabel('Rounds', fontsize=19)
            ax.set_ylabel(ylabel, fontsize=19)
            ax.grid(True, alpha=0.3, linewidth=1.0)
            ax.tick_params(axis='both', which='major', labelsize=19)
            ax.legend(fontsize=19, frameon=True, framealpha=0.5, loc='upper right')
            
            plt.tight_layout(pad=0.5)
            
            skip_plots = [
                'eMBB_emd_line_with_ci.png',
                'eMBB_sinr_line_with_ci.png',
                'mMTC_emd_line_with_ci.png',
                'mMTC_sinr_line_with_ci.png',
                'URLLC_emd_line_with_ci.png',
                'URLLC_sinr_line_with_ci.png'
            ]
            
            filename = f'{individual_dir}/{slice_type}_{metric_name.lower().replace(" ", "_")}_line_with_ci.png'
            if filename not in skip_plots and not any(skip in filename for skip in ['eMBB_emd_line', 'eMBB_sinr_line', 'mMTC_emd_line', 'mMTC_sinr_line', 'URLLC_emd_line', 'URLLC_sinr_line']):
                plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.1)
                plt.close()
                print(f"  ✓ Saved individual line plot with CI: {filename}")
            else:
                plt.close()
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        axes = axes.flatten()
        
        metrics_to_plot = min(6, len(metrics_config))
        
        plot_idx = 0
        for idx in range(len(metrics_config)):
            metric, ylabel, metric_name = metrics_config[idx]
            
            if metric == 'SINR_Mean_dB' or metric == 'Emd_Mean':
                continue
                
            if plot_idx >= metrics_to_plot:
                break
                
            ax = axes[plot_idx]
            
            results_dict = {
                'Random': random_results,
                'ChQual': sinr_results,
                'DataQual': data_quality_results,
                'ChDataQual': combined_results
            }
            
            scenarios = []
            means = []
            cis = []
            colors = []
            
            for scenario, results in results_dict.items():
                if not results.empty and metric in results.columns:
                    scenarios.append(scenario)
                    means.append(results[metric].mean())
                    ci = 1.96 * results[metric].std() / np.sqrt(len(results))
                    cis.append(ci)
                    colors.append(scenario_colors[scenario])
            
            if not scenarios:
                continue
                
            x_pos = np.arange(len(scenarios))
            bars = ax.bar(scenarios, 
                         means,
                         yerr=cis,
                         capsize=15,
                         color=colors,
                         edgecolor='white',
                         alpha=0.8,
                         linewidth=3.0,
                         error_kw={'elinewidth': 2.5, 'capthick': 2.5})
            
            max_bar_height = max([m + c for m, c in zip(means, cis)])
                        
            for i, (bar, value, ci) in enumerate(zip(bars, means, cis)):
                text_x = bar.get_x() + bar.get_width()/2.
                text_y = bar.get_height() + ci + (0.005 * max_bar_height)
                
                ax.text(text_x, text_y,
                       f'{value:.3f}',
                       ha='center', va='bottom', 
                       fontsize=19,
                       fontweight='bold')
            
            ax.set_ylabel(ylabel, fontsize=19)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(scenarios, fontsize=19)
            ax.grid(True, alpha=0.3, axis='y', linewidth=1.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=19)
            
            if 'Random' in scenarios and 'ChQual' in scenarios:
                random_idx = scenarios.index('Random')
                sinr_idx = scenarios.index('ChQual')
                random_mean = means[random_idx]
                sinr_mean = means[sinr_idx]
                
                if metric in ['SINR_Mean_dB', 'Throughput_Mean_Mbps']:
                    improvement = (sinr_mean - random_mean) / random_mean * 100
                else:
                    improvement = (random_mean - sinr_mean) / random_mean * 100
                
                max_text_height = max([bar.get_height() + ci for bar, ci in zip(bars, cis)])
                improvement_y = max_text_height * 1.15
                
                if ((metric in ['SINR_Mean_dB', 'Throughput_Mean_Mbps'] and improvement > 0) or
                    (metric not in ['SINR_Mean_dB', 'Throughput_Mean_Mbps'] and improvement > 0)):
                    imp_color = 'green'
                    symbol = '↑'
                else:
                    imp_color = 'red'
                    symbol = '↓'
                
                ax.text(0.5, improvement_y, 
                       f'Δ = {improvement:+.1f}% {symbol}', 
                       ha='center', va='center', 
                       fontsize=19,
                       fontweight='bold',
                       color=imp_color,
                       bbox=dict(boxstyle="round,pad=0.5",
                                facecolor="white", 
                                alpha=0.5,
                                edgecolor='gray', 
                                linewidth=2.0))
            
            plot_idx += 1
        
        plt.tight_layout(pad=3.0)
        filename = f'{plots_dir}/{slice_type}_all_metrics_comparison_with_ci.png'
        plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.3)
        plt.close()
        print(f"  ✓ Saved combined plot with CI: {filename}")
        
        create_improvement_radar_chart(random_results, sinr_results, data_quality_results, combined_results, slice_type, plots_dir)
        
        for metric, ylabel, metric_name in metrics_config:
            fig, ax = plt.subplots(figsize=(7, 6))
            
            results_dict = {
                'Random': random_results,
                'ChQual': sinr_results,
                'DataQual': data_quality_results,
                'ChDataQual': combined_results
            }
            
            scenarios = []
            means = []
            cis = []
            colors = []
            
            for scenario, results in results_dict.items():
                if not results.empty and metric in results.columns:
                    scenarios.append(scenario)
                    means.append(results[metric].mean())
                    ci = 1.96 * results[metric].std() / np.sqrt(len(results))
                    cis.append(ci)
                    colors.append(scenario_colors[scenario])
            
            if not scenarios:
                continue
            
            x_pos = np.arange(len(scenarios))
            bars = ax.bar(scenarios, 
                         means,
                         yerr=cis,
                         capsize=12,
                         color=colors,
                         edgecolor='white',
                         alpha=0.8,
                         linewidth=2.5,
                         error_kw={'elinewidth': 2.0, 'capthick': 2.0})
            
            max_bar_height = max([m + c for m, c in zip(means, cis)])
                      
            for i, (bar, value, ci) in enumerate(zip(bars, means, cis)):
                text_x = bar.get_x() + bar.get_width()/2.
                text_y = bar.get_height() + ci + (0.005 * max_bar_height)
                
                ax.text(text_x, text_y,
                       f'{value:.3f}',
                       ha='center', va='bottom', 
                       fontsize=19)     # fontweight='bold'
            
            ax.set_ylabel(ylabel, fontsize=19)
            ax.set_xlabel('Scenarios', fontsize=19)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(scenarios, fontsize=19)
            ax.grid(True, alpha=0.3, axis='y', linewidth=1.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=19)
            
            plt.tight_layout()
            
            skip_plots = [
                'eMBB_emd_bar_with_ci.png',
                'eMBB_sinr_bar_with_ci.png',
                'mMTC_emd_bar_with_ci.png',
                'mMTC_sinr_bar_with_ci.png',
                'URLLC_emd_bar_with_ci.png',
                'URLLC_sinr_bar_with_ci.png'
            ]
            
            filename = f'{individual_dir}/{slice_type}_{metric_name.lower().replace(" ", "_")}_bar_with_ci.png'
            if filename not in skip_plots and not any(skip in filename for skip in ['eMBB_emd_bar', 'eMBB_sinr_bar', 'mMTC_emd_bar', 'mMTC_sinr_bar', 'URLLC_emd_bar', 'URLLC_sinr_bar']):
                plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.15)
                plt.close()
                print(f"  ✓ Saved individual bar plot with CI: {filename}")
            else:
                plt.close()

def create_improvement_radar_chart(random_results, sinr_results, data_quality_results, combined_results, slice_type, plots_dir):
    metrics = ['Throughput_Mean_Mbps', 'PacketLossRatio_Mean_%',
               'Delay_Mean_ms', 'Jitter_Mean_ms', 'Energy_Mean_J']
    metric_labels = ['Throughput', 'Packet Loss Ratio', 'Delay', 'Jitter', 'Energy Consumption']
    
    scenario_results = {
        'Random': random_results,
        'ChQual': sinr_results,
        'DataQual': data_quality_results,
        'ChDataQual': combined_results
    }
    
    scenario_colors = {
        'Random': '#d62728',
        'ChQual': '#ff7f0e',
        'DataQual': '#1f77b4',
        'ChDataQual': '#2ca02c'
    }
    
    fig, ax = plt.subplots(figsize=(20, 20), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    angles = np.append(angles, angles[0])
    
    all_improvements = {}
    
    for scenario_name, scenario_data in scenario_results.items():
        if scenario_data.empty:
            continue
            
        improvements = []
        
        for metric in metrics:
            if metric not in random_results.columns or metric not in scenario_data.columns:
                improvements.append(0)
                continue
                
            random_mean = np.mean(random_results[metric].values)
            scenario_mean = np.mean(scenario_data[metric].values)
            
            if random_mean == 0 or not np.isfinite(random_mean) or not np.isfinite(scenario_mean):
                improvements.append(0)
                continue
                
            if metric == 'Throughput_Mean_Mbps':
                imp = (scenario_mean - random_mean) / random_mean * 100
            else:
                imp = (random_mean - scenario_mean) / random_mean * 100
            
            if not np.isfinite(imp):
                imp = 0
                
            improvements.append(imp)
        
        improvements.append(improvements[0])
        all_improvements[scenario_name] = improvements
    
    for scenario_name, improvements in all_improvements.items():
        clean_improvements = [0 if not np.isfinite(x) else x for x in improvements]
        ax.plot(angles, clean_improvements, 'o-', linewidth=5.0, markersize=16,
                color=scenario_colors[scenario_name], label=scenario_name)
    
    placed_labels = []
    min_radial_gap = 28
    
    for metric_idx, angle in enumerate(angles[:-1]):
        
        metric_labels_group = []
        
        for scenario_name, improvements in all_improvements.items():
            imp = improvements[metric_idx]
            if not np.isfinite(imp) or abs(imp) < 1:
                continue
            
            arrow = '↑' if imp > 0 else '↓'
            sign = '+' if imp > 0 else '-'
            text = f'{sign}{abs(imp):.1f}%{arrow}'
            
            metric_labels_group.append({
                'angle': angle,
                'base_angle': angle,
                'r': imp,
                'text': text,
                'color': scenario_colors[scenario_name]
            })
        
        metric_labels_group.sort(key=lambda x: x['r'])
        
        cluster_threshold = 15
        expanded_gap = 32
        
        i = 0
        while i < len(metric_labels_group) - 1:
            
            cluster = [metric_labels_group[i]]
            j = i + 1
            
            while j < len(metric_labels_group) and \
                  abs(metric_labels_group[j]['r'] - cluster[-1]['r']) < cluster_threshold:
                cluster.append(metric_labels_group[j])
                j += 1
            
            if len(cluster) > 1:
                center = np.mean([l['r'] for l in cluster])
                start = center - expanded_gap * (len(cluster)-1)/2
                
                for k, lbl in enumerate(cluster):
                    lbl['r'] = start + k * expanded_gap
            
            i = j
        
        for k in range(1, len(metric_labels_group)):
            if metric_labels_group[k]['r'] - metric_labels_group[k-1]['r'] < min_radial_gap:
                metric_labels_group[k]['r'] = metric_labels_group[k-1]['r'] + min_radial_gap
        
        placed_labels.extend(metric_labels_group)
    
    def polar_to_cart(a, r):
        return r*np.cos(a), r*np.sin(a)
    
    box_w = 22
    box_h = 16
    max_angle_shift = 0.08 
    
    max_iter = 120
    for _ in range(max_iter):
        overlap_found = False
        
        for i in range(len(placed_labels)):
            for j in range(i+1, len(placed_labels)):
                
                l1 = placed_labels[i]
                l2 = placed_labels[j]
                
                if not (l1['color'] == '#1f77b4' and l2['color'] == '#1f77b4'):
                    continue
                
                x1, y1 = polar_to_cart(l1['angle'], l1['r'])
                x2, y2 = polar_to_cart(l2['angle'], l2['r'])
                
                dx = x2 - x1
                dy = y2 - y1
                
                if abs(dx) < 2*box_w and abs(dy) < 2*box_h:
                    
                    overlap_found = True
                    
                    shift_r = 6 
                    
                    if l2['r'] >= l1['r']:
                        l2['r'] += shift_r
                    else:
                        l2['r'] -= shift_r
                    
                    angle_diff = l2['angle'] - l2['base_angle']
                    
                    if abs(angle_diff) < max_angle_shift:
                        if dx > 0:
                            l2['angle'] += 0.02
                        else:
                            l2['angle'] -= 0.02
                    
        if not overlap_found:
            break
    
    for label in placed_labels:
        
        r = label['r']
        if r > 180:
            r = 180
        if r < -80:
            r = -80
        
        ax.plot([label['base_angle'], label['angle']],
                [label['r'], r],
                color=label['color'], linewidth=1,
                alpha=0.3, linestyle=':')
        
        ax.text(label['angle'], r,
                label['text'],
                ha='center', va='center',
                fontsize=40, fontweight='bold',
                color=label['color'],
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white",
                          alpha=0.5,
                          edgecolor=label['color'],
                          linewidth=2.5))
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=44)
    ax.tick_params(axis='y', labelsize=40)
    ax.set_ylim(-100, 150)
    ax.yaxis.grid(True, alpha=0.8, linewidth=6.0)
    ax.xaxis.grid(True, alpha=0.3, linewidth=4.0)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=4, alpha=0.5)
    
    ax.legend(fontsize=40, frameon=True, framealpha=0.9,
              loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    plt.tight_layout()
    filename = f'{plots_dir}/{slice_type}_improvement_radar.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print(f"  ✓ Saved simplified radar plot with arrows: {filename}")


def create_per_slice_summary_tables(random_results_dict, sinr_results_dict, data_quality_results_dict, combined_results_dict):
    print("\n" + "="*70)
    print("PER-SLICE PERFORMANCE SUMMARY TABLES (100 ITERATIONS)")
    print("="*70)
    
    results_dir = PLOTS_BASE / 'net_plots' / 'per_slice_comparison_results_100iter'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    slice_types = list(random_results_dict.keys())
    
    for slice_type in slice_types:
        print(f"\n{slice_type} Slice Performance Summary:")
        print("-" * 100)
        
        random_results = random_results_dict[slice_type]
        sinr_results = sinr_results_dict[slice_type]
        data_quality_results = data_quality_results_dict.get(slice_type, pd.DataFrame())
        combined_results = combined_results_dict.get(slice_type, pd.DataFrame())
        
        comparison_data = []
        metrics = ['SINR_Mean_dB', 'Throughput_Mean_Mbps', 'PacketLossRatio_Mean_%',
                  'Delay_Mean_ms', 'Jitter_Mean_ms', 'Energy_Mean_J']
        
        if 'Emd_Mean' in random_results.columns:
            metrics.append('Emd_Mean')
        
        for metric in metrics:
            row_data = {'Metric': metric.replace('_Mean_', '_').replace('_', ' ').title()}
            
            random_mean = random_results[metric].mean()
            random_std = random_results[metric].std()
            random_ci = 1.96 * random_std / np.sqrt(len(random_results))
            row_data['Random_Mean'] = random_mean
            row_data['Random_CI'] = random_ci
            
            sinr_mean = sinr_results[metric].mean()
            sinr_std = sinr_results[metric].std()
            sinr_ci = 1.96 * sinr_std / np.sqrt(len(sinr_results))
            row_data['Channel_Quality_Mean'] = sinr_mean
            row_data['Channel_Quality_CI'] = sinr_ci
            
            if metric in ['SINR_Mean_dB', 'Throughput_Mean_Mbps']:
                improvement = (sinr_mean - random_mean) / random_mean * 100
                direction = "↑" if improvement > 0 else "↓"
            else:
                improvement = (random_mean - sinr_mean) / random_mean * 100
                direction = "↓" if improvement > 0 else "↑"
            
            row_data['Channel_Quality_Improvement_%'] = improvement
            row_data['Channel_Quality_Direction'] = direction
            
            if not data_quality_results.empty and metric in data_quality_results.columns:
                data_quality_mean = data_quality_results[metric].mean()
                data_quality_std = data_quality_results[metric].std()
                data_quality_ci = 1.96 * data_quality_std / np.sqrt(len(data_quality_results))
                row_data['Data_Quality_Mean'] = data_quality_mean
                row_data['Data_Quality_CI'] = data_quality_ci
                
                if metric in ['SINR_Mean_dB', 'Throughput_Mean_Mbps']:
                    improvement = (data_quality_mean - random_mean) / random_mean * 100
                    direction = "↑" if improvement > 0 else "↓"
                else:
                    improvement = (random_mean - data_quality_mean) / random_mean * 100
                    direction = "↓" if improvement > 0 else "↑"
                
                row_data['Data_Quality_Improvement_%'] = improvement
                row_data['Data_Quality_Direction'] = direction
            
            if not combined_results.empty and metric in combined_results.columns:
                combined_mean = combined_results[metric].mean()
                combined_std = combined_results[metric].std()
                combined_ci = 1.96 * combined_std / np.sqrt(len(combined_results))
                row_data['Combined_Quality_Mean'] = combined_mean
                row_data['Combined_Quality_CI'] = combined_ci
                
                if metric in ['SINR_Mean_dB', 'Throughput_Mean_Mbps']:
                    improvement = (combined_mean - random_mean) / random_mean * 100
                    direction = "↑" if improvement > 0 else "↓"
                else:
                    improvement = (random_mean - combined_mean) / random_mean * 100
                    direction = "↓" if improvement > 0 else "↑"
                
                row_data['Combined_Quality_Improvement_%'] = improvement
                row_data['Combined_Quality_Direction'] = direction
            
            comparison_data.append(row_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print(f"{'Metric':<20} {'Random':>12} {'ChQual':>20} {'DataQual':>20} {'Combined':>20}")
        print(f"{'':<20} {'Mean ± CI':>12} {'Mean ± CI':>20} {'Mean ± CI':>20} {'Mean ± CI':>20}")
        print("-" * 100)
        
        for _, row in comparison_df.iterrows():
            metric_display = row['Metric']
            if 'Sinr' in metric_display:
                metric_display = metric_display.replace('Sinr', 'SINR')
            elif 'Plr' in metric_display:
                metric_display = metric_display.replace('Plr', 'PLR')
            
            line = f"{metric_display:<20} " \
                   f"{row['Random_Mean']:6.3f}±{row['Random_CI']:5.3f}  " \
                   f"{row['Channel_Quality_Mean']:6.3f}±{row['Channel_Quality_CI']:5.3f} "
            
            if 'Data_Quality_Mean' in row and not pd.isna(row['Data_Quality_Mean']):
                line += f"{row['Data_Quality_Mean']:6.3f}±{row['Data_Quality_CI']:5.3f}  "
            else:
                line += f"{'N/A':>13}  "
            
            if 'Combined_Quality_Mean' in row and not pd.isna(row['Combined_Quality_Mean']):
                line += f"{row['Combined_Quality_Mean']:6.3f}±{row['Combined_Quality_CI']:5.3f}"
            else:
                line += f"{'N/A':>13}"
            
            print(line)
        
        print("-" * 100)
        
        comparison_df_formatted = comparison_df.copy()
        for col in comparison_df_formatted.columns:
            if col != 'Metric' and '_Direction' not in col:
                comparison_df_formatted[col] = comparison_df_formatted[col].round(3)
        
        comparison_df_formatted.to_csv(f'{results_dir}/{slice_type}_performance_comparison_with_ci.csv', index=False, float_format='%.3f')
        
        random_results.to_csv(f'{results_dir}/{slice_type}_random_results.csv', index=False, float_format='%.3f')
        sinr_results.to_csv(f'{results_dir}/{slice_type}_channel_quality_results.csv', index=False, float_format='%.3f')
        
        if not data_quality_results.empty:
            data_quality_results.to_csv(f'{results_dir}/{slice_type}_data_quality_results.csv', index=False, float_format='%.3f')
        
        if not combined_results.empty:
            combined_results.to_csv(f'{results_dir}/{slice_type}_combined_quality_results.csv', index=False, float_format='%.3f')

def main():    
    print("Comparing four scenarios per slice:")
    print("1. Random: Select 10 clients randomly from that slice (100 times)")
    print("2. ChQual: Select 30 random clients from that slice,")
    print("   then choose best 10 with highest SINR (100 times)")
    print("3. DataQual: Select 30 random clients from that slice,")
    print("   then choose best 10 with lowest EMD (lower is better) (100 times)")
    print("4. Combined ChDataQual: Select 30 random clients from that slice,")
    print("   then choose best 10 using combined score (SINR normalized - EMD normalized) (100 times)")
    print("All scenarios run for 100 iterations PER SLICE")
    # print("=" * 70)
    
    df = load_ue_statistics()
    if df is None:
        return
    
    analyze_slice_performance(df)
    
    print("\n" + "="*70)
    print("RUNNING PER-SLICE RANDOM SELECTION (100 ITERATIONS)...")
    print("="*70)
    random_results_dict = per_slice_random_selection(df, num_clients=10, iterations=100)
    
    print("\n" + "="*70)
    print("RUNNING PER-SLICE SINR-BASED SELECTION (100 ITERATIONS)...")
    print("="*70)
    sinr_results_dict = per_slice_sinr_selection(df, pool_size=30, num_selected=10, iterations=100)
    
    print("\n" + "="*70)
    print("RUNNING PER-SLICE DataQual SELECTION (100 ITERATIONS)...")
    print("="*70)
    data_quality_results_dict = {}
    if 'Emd_Value' in df.columns:
        data_quality_results_dict = per_slice_data_quality_selection(df, pool_size=30, num_selected=10, iterations=100)
    else:
        print("Skipping DataQual selection - Emd_Value column not found.")
    
    combined_results_dict = {}
    if 'Emd_Value' in df.columns:
        combined_results_dict = per_slice_combined_quality_selection(df, pool_size=30, num_selected=10, iterations=100)
    else:
        print("Skipping Combined Quality selection - Emd_Value column not found.")
    
    if not random_results_dict or not sinr_results_dict:
        print("Error: Could not run basic scenarios. Check if SliceType column exists.")
        return
        
    create_per_slice_comparison_plots(random_results_dict, sinr_results_dict, data_quality_results_dict, combined_results_dict)
    # create_per_slice_summary_tables(random_results_dict, sinr_results_dict, data_quality_results_dict, combined_results_dict)
    
if __name__ == "__main__":
    main()

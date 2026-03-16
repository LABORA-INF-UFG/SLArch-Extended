#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import concurrent.futures
import threading
import queue
import time
import random
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent          
PROJECT_ROOT = SCRIPT_DIR.parent                       
POST_DIR = PROJECT_ROOT / "post-process"               
DATA_DIR = SCRIPT_DIR / "ds-15"                        
MODEL_DIR = SCRIPT_DIR / "trained_models"              
PROGRESS_DIR = SCRIPT_DIR / "training_progress"       
ROUND_RESULTS_PATH = PROJECT_ROOT / "split_round_results"     

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
ROUND_RESULTS_PATH.mkdir(parents=True, exist_ok=True)

TOTAL_UES = 150
CLIENTS_PER_BATCH = 10         
BATCH_SIZE = 16
EPOCHS = 5
LR = 0.001
BASE_SEED = 42
MONTE_CARLO_ITERATIONS = 1      
PROGRESS_SAVE_INTERVAL = 50

CONFIG = {
    'overfit_control': 0.7,
    'min_param': 0.01,
    'seed': BASE_SEED,
    'transmission_baseline_std': 0.1
}

class SLDeviceSelection:
    def __init__(self, dataset_folder=None, pool_size=30, n_select=10, n_monte_carlo=50):
        if dataset_folder is None:
            dataset_folder = DATA_DIR   
        self.dataset_folder = dataset_folder
        self.pool_size = pool_size
        self.n_select = n_select
        self.n_monte_carlo = n_monte_carlo
        self.rng = np.random.default_rng(CONFIG['seed'])
        self.min_param = CONFIG['min_param']
        self.overfit_control = CONFIG['overfit_control']
        self.sinr_baseline_std = None
        self.emd_baseline_std = None
        self.transmission_baseline_std = CONFIG['transmission_baseline_std']
        self.dataset_cache = {}

    def normalize_dataframe(self, df):
        numeric_cols = df.select_dtypes(include=np.number).columns
        df_norm = df.copy()
        for col in numeric_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df_norm[col] = 0.0
        return df_norm

    def get_device_dataset(self, ue_id):
        
        return pd.DataFrame()

    def calculate_baseline_variabilities(self, df):
        self.sinr_baseline_std = df['SINR_Mean_dB'].std()
        self.emd_baseline_std = df['Emd_Value'].std()
        print(f"\nBaseline Variabilities:")
        print(f"  SINR std: {self.sinr_baseline_std:.3f} dB")
        print(f"  EMD std: {self.emd_baseline_std:.6f}")
        print(f"  Overfit Control: {self.overfit_control}")
        print(f"  Effective SINR noise: ±{self.overfit_control * self.sinr_baseline_std:.3f} dB")
        print(f"  Effective EMD noise: ±{self.overfit_control * self.emd_baseline_std:.6f}")

    def apply_sinr_measurement_noise(self, sinr_values):
        if self.overfit_control == 0 or self.sinr_baseline_std is None:
            return sinr_values
        noise_std = self.overfit_control * self.sinr_baseline_std
        noise = self.rng.normal(0, noise_std, size=len(sinr_values))
        return sinr_values + noise

    def apply_emd_measurement_noise(self, emd_values):
        if self.overfit_control == 0 or self.emd_baseline_std is None:
            return emd_values
        noise_std = self.overfit_control * self.emd_baseline_std
        noise = self.rng.normal(0, noise_std, size=len(emd_values))
        return np.maximum(emd_values + noise, 0.001)

    def apply_transmission_model_noise(self, probabilities):
        if self.overfit_control == 0:
            return probabilities
        noise_std = self.overfit_control * self.transmission_baseline_std
        noisy_probs = probabilities + self.rng.normal(0, noise_std, size=len(probabilities))
        return np.clip(noisy_probs, 0.05, 0.95)

    def random_selection(self, df_slice):
        if len(df_slice) < self.n_select:
            return None
        indices = self.rng.choice(len(df_slice), size=self.n_select, replace=False)
        selected = df_slice.iloc[indices].copy()
        if self.overfit_control > 0:
            selected['SINR_Mean_dB'] = self.apply_sinr_measurement_noise(selected['SINR_Mean_dB'].values)
        return selected

    def chqual_selection(self, df_slice):
        if len(df_slice) < self.pool_size:
            return None
        pool = df_slice.sample(n=self.pool_size, random_state=self.rng).copy()
        sinr_noisy = self.apply_sinr_measurement_noise(pool['SINR_Mean_dB'].values)
        top_indices = np.argsort(sinr_noisy)[-self.n_select:]
        selected = pool.iloc[top_indices].copy()
        selected['SINR_Mean_dB'] = sinr_noisy[top_indices]
        return selected

    def dataqual_selection(self, df_slice, apply_overfit_control=True):
        if len(df_slice) < self.pool_size:
            return None
        pool = df_slice.sample(n=self.pool_size, random_state=self.rng).copy()
        emd_noisy = self.apply_emd_measurement_noise(pool['Emd_Value'].values)
        top_indices = np.argsort(emd_noisy)[:self.n_select]  
        selected = pool.iloc[top_indices].copy()
        if apply_overfit_control and self.overfit_control > 0:
            selected['SINR_Mean_dB'] = self.apply_sinr_measurement_noise(selected['SINR_Mean_dB'].values)
        return selected

    def chdataqual_selection(self, df_slice, phi, beta, apply_overfit_control=True):
        if len(df_slice) < self.pool_size:
            return None, phi, beta

        phi = max(self.min_param, min(phi, 1 - self.min_param))
        beta = max(self.min_param, min(beta, 1 - self.min_param))
        total = phi + beta
        phi /= total
        beta /= total

        pool = df_slice.sample(n=self.pool_size, random_state=self.rng).copy()

        sinr_noisy = self.apply_sinr_measurement_noise(pool['SINR_Mean_dB'].values)
        emd_noisy = self.apply_emd_measurement_noise(pool['Emd_Value'].values)

        sinr_norm = (sinr_noisy - sinr_noisy.min()) / (sinr_noisy.max() - sinr_noisy.min() + 1e-9)
        emd_norm = (emd_noisy - emd_noisy.min()) / (emd_noisy.max() - emd_noisy.min() + 1e-9)

        score = phi * emd_norm + beta * (1 - sinr_norm)

        top_indices = np.argsort(score)[:self.n_select]
        selected = pool.iloc[top_indices].copy()

        if apply_overfit_control and self.overfit_control > 0:
            selected['SINR_Mean_dB'] = self.apply_sinr_measurement_noise(selected['SINR_Mean_dB'].values)
            selected['Emd_Value'] = self.apply_emd_measurement_noise(selected['Emd_Value'].values)
        else:
            selected['SINR_Mean_dB'] = sinr_noisy[top_indices]
            selected['Emd_Value'] = emd_noisy[top_indices]

        return selected, phi, beta

    def evaluate_selection(self, selected_df):
        if selected_df is None or len(selected_df) == 0:
            return float('inf')
        return np.mean(selected_df['Emd_Value'].values)

    def optimize_chdataqual_params(self, df_slice):
        best_phi, best_beta = 0.5, 0.5
        best_emd = float('inf')
        param_values = np.linspace(self.min_param, 1 - self.min_param, 11)

        print("\nOptimizing φ and β (minimising mean EMD)...")
        for phi in param_values:
            for beta in param_values:
                if phi + beta > 1:
                    continue
                emds = []
                for _ in range(min(self.n_monte_carlo, 100)):
                    sel, _, _ = self.chdataqual_selection(df_slice, phi, beta)
                    emds.append(self.evaluate_selection(sel))
                mean_emd = np.mean(emds)
                print(f"  Testing φ={phi:.2f}, β={beta:.2f} -> mean EMD={mean_emd:.6f}")
                if mean_emd < best_emd:
                    best_emd = mean_emd
                    best_phi, best_beta = phi, beta
                    print(f"    >> New best found! φ={best_phi:.2f}, β={best_beta:.2f}, mean EMD={best_emd:.6f}")

        _, phi_opt_norm, beta_opt_norm = self.chdataqual_selection(df_slice, best_phi, best_beta)
        print(f"Optimization complete: Optimal φ={phi_opt_norm:.2f}, β={beta_opt_norm:.2f}, best mean EMD={best_emd:.6f}\n")
        return phi_opt_norm, beta_opt_norm

    def get_optimized_params(self, df):
        optimized_params = {}
        self.calculate_baseline_variabilities(df)
        for slice_type in df['SliceType'].unique():
            df_slice = df[df['SliceType'] == slice_type].copy()
            print(f"\n{slice_type}: {len(df_slice)} devices")
            phi_opt, beta_opt = self.optimize_chdataqual_params(df_slice)
            optimized_params[slice_type] = (phi_opt, beta_opt)
        return optimized_params

class ml_model_in(nn.Module):
    def __init__(self):
        super(ml_model_in, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.features(x)
        return x

class ml_model_hidden(nn.Module):
    def __init__(self):
        super(ml_model_hidden, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.features(x)
        return x

class ml_model_out(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(ml_model_out, self).__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x

class PickleDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        pixel_values = row.iloc[:-1].values.astype(np.float32)
        image = pixel_values.reshape(28, 28)
        image = image / 255.0
        label = int(row.iloc[-1])
        image = torch.from_numpy(image).unsqueeze(0)
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.RandomRotation(25),
    transforms.Normalize((0.1307,), (0.3081,))
])

def load_client_data(client_id, data_type='train'):
    file_num = client_id + 1
    filename = f"{file_num}_{data_type}.pickle"
    filepath = DATA_DIR / filename
    try:
        if not filepath.exists():
            return None
        with open(filepath, 'rb') as f:
            df = pd.read_pickle(f)
        if df.empty:
            return None
        if 'label' not in df.columns:
            return None
        return PickleDataset(df, transform=transform)
    except Exception as e:
        print(f"[Data Loading] Error loading {filename}: {e}")
        return None

class TrainingProgressTracker:
    def __init__(self):
        self.progress_data = []
        self.progress_lock = threading.Lock()
        self.progress_file = PROGRESS_DIR / "training_progress.csv"

    def get_progress_filename(self, round_id):
        return PROGRESS_DIR / f"progress_round_{round_id}.csv"

    def record_progress(self, round_id, global_step, epoch, batch_idx,
                       server_loss, avg_client_loss, validation_accuracy=None):
        with self.progress_lock:
            progress_point = {
                'round_id': round_id,
                'global_step': global_step,
                'epoch': epoch,
                'batch_index': batch_idx,
                'server_loss': server_loss,
                'avg_client_loss': avg_client_loss,
                'validation_accuracy': validation_accuracy,
                'timestamp': time.time(),
                'wall_clock_time': time.strftime('%Y-%m-%d,%H:%M:%S')
            }
            self.progress_data.append(progress_point)
            self._save_to_file(progress_point, self.progress_file)
            round_progress_file = self.get_progress_filename(round_id)
            self._save_to_file(progress_point, round_progress_file)
            return progress_point

    def _save_to_file(self, progress_point, filename):
        df = pd.DataFrame([progress_point])
        if filename.exists():
            df.to_csv(filename, mode='a', header=False, index=False)
        else:
            df.to_csv(filename, index=False)

    def get_progress_summary(self):
        if not self.progress_data:
            return None
        df = pd.DataFrame(self.progress_data)
        summary = {
            'total_progress_points': len(df),
            'total_global_steps': df['global_step'].max() if 'global_step' in df.columns else 0,
            'rounds_tracked': df['round_id'].nunique(),
            'avg_server_loss': df['server_loss'].mean() if 'server_loss' in df.columns else None,
            'avg_client_loss': df['avg_client_loss'].mean() if 'avg_client_loss' in df.columns else None,
            'final_validation_accuracy': df['validation_accuracy'].iloc[-1] if 'validation_accuracy' in df.columns and not df['validation_accuracy'].isna().all() else None
        }
        return summary

def train_single_client(client_id, progress_tracker, round_id):
    client_model_path = MODEL_DIR / f"model_client_{client_id}.pth"
    server_body_path = MODEL_DIR / f"model_server_body_{client_id}.pth"
    server_head_path = MODEL_DIR / f"model_server_head_{client_id}.pth"

    if client_model_path.exists() and server_body_path.exists() and server_head_path.exists():
        print(f"[Client {client_id}] Models already exist. Loading model {client_id} for evaluation...")
        client_model = ml_model_in()
        server_body = ml_model_hidden()
        server_head = ml_model_out(10)
        client_model.load_state_dict(torch.load(client_model_path, map_location='cpu', weights_only=True))
        server_body.load_state_dict(torch.load(server_body_path, map_location='cpu', weights_only=True))
        server_head.load_state_dict(torch.load(server_head_path, map_location='cpu', weights_only=True))
        return client_model, server_body, server_head

    print(f"[Client {client_id}] Training new models using dataset {client_id+1}...")

    train_dataset = load_client_data(client_id, 'train')
    if train_dataset is None or len(train_dataset) == 0:
        print(f"[Client {client_id}] No training data, skipping.")
        return None, None, None

    client_model = ml_model_in()
    server_body = ml_model_hidden()
    server_head = ml_model_out(10)
    device = 'cpu'
    client_model.to(device)
    server_body.to(device)
    server_head.to(device)

    optimizer = optim.Adam(
        list(client_model.parameters()) + list(server_body.parameters()) + list(server_head.parameters()),
        lr=LR
    )
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    client_model.train()
    server_body.train()
    server_head.train()

    global_step = 0
    for epoch in range(EPOCHS):
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            smashed = client_model(x)
            hidden = server_body(smashed)
            out = server_head(hidden)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % PROGRESS_SAVE_INTERVAL == 0:
                progress_tracker.record_progress(
                    round_id=round_id,
                    global_step=global_step,
                    epoch=epoch+1,
                    batch_idx=batch_idx,
                    server_loss=loss.item(),
                    avg_client_loss=loss.item()  
                )

    torch.save(client_model.state_dict(), client_model_path)
    torch.save(server_body.state_dict(), server_body_path)
    torch.save(server_head.state_dict(), server_head_path)
    print(f"[Client {client_id}] Models saved.")

    return client_model, server_body, server_head

def evaluate_client(client_id, client_model, server_body, server_head):
    test_dataset = load_client_data(client_id, 'test')
    if test_dataset is None or len(test_dataset) == 0:
        return 0.0
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    device = 'cpu'
    client_model.eval()
    server_body.eval()
    server_head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            smashed = client_model(x)
            hidden = server_body(smashed)
            out = server_head(hidden)
            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    acc = correct / total if total > 0 else 0.0
    return acc

def run_round_experiment():

    csv_path = POST_DIR / "ue_complete_select_params.csv"
    if not csv_path.exists():
        print(f"ERROR: Required file not found: {csv_path}")
        print("Please ensure 'ue_complete_select_params.csv' exists in the post-process folder.")
        sys.exit(1)

    ue_df = pd.read_csv(csv_path)
    ue_df['UE_ID'] = pd.to_numeric(ue_df['UE_ID'], errors='coerce')
    ue_df['SINR_Mean_dB'] = pd.to_numeric(ue_df['SINR_Mean_dB'], errors='coerce')
    ue_df['Emd_Value'] = pd.to_numeric(ue_df['Emd_Value'], errors='coerce')
    ue_df.dropna(subset=['UE_ID', 'SINR_Mean_dB', 'Emd_Value'], inplace=True)
    ue_df['UE_ID'] = ue_df['UE_ID'].astype(int)

    selector = SLDeviceSelection(
        dataset_folder=DATA_DIR,
        pool_size=30,
        n_select=10,
        n_monte_carlo=50 
    )

    optimized_params = selector.get_optimized_params(ue_df)

    slices = ue_df['SliceType'].unique()
    scenarios = ['Random', 'ChQual', 'DataQual', 'ChDataQual']

    progress_tracker = TrainingProgressTracker()

    for round_num in range(1, 101):
        
        print(f"\nROUND {round_num}")
        print(f"{'='*60}")

        selector.rng = np.random.default_rng(BASE_SEED + round_num)

        round_results = []  

        for slc in slices:
            df_slice = ue_df[ue_df['SliceType'] == slc].copy()
            phi_opt, beta_opt = optimized_params[slc]

            for scen in scenarios:
                print(f"\n--- Slice: {slc}, Scenario: {scen} ---")

                if scen == 'Random':
                    selected = selector.random_selection(df_slice)
                elif scen == 'ChQual':
                    selected = selector.chqual_selection(df_slice)
                elif scen == 'DataQual':
                    selected = selector.dataqual_selection(df_slice)
                elif scen == 'ChDataQual':
                    selected, _, _ = selector.chdataqual_selection(df_slice, phi_opt, beta_opt)
                else:
                    continue

                if selected is None or len(selected) == 0:
                    print(f"  No clients selected for {slc} {scen} round {round_num}, skipping.")
                    continue

                selected_ids = selected['UE_ID'].values.astype(int).tolist()
                print(f"  Selected clients: {selected_ids}")

                timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                for cid in selected_ids:
                    client_model, server_body, server_head = train_single_client(
                        cid, progress_tracker, round_num
                    )
                    if client_model is not None:
                        acc = evaluate_client(cid, client_model, server_body, server_head)
                    else:
                        acc = 0.0

                    round_results.append({
                        'client_id': cid,
                        'round': round_num,
                        'slice_type': slc,
                        'scenario': scen,
                        'accuracy': acc,
                        'timestamp': timestamp
                    })

        if round_results:
            round_df = pd.DataFrame(round_results)
            round_df = round_df[['client_id', 'round', 'slice_type', 'scenario', 'accuracy', 'timestamp']]
            round_filename = ROUND_RESULTS_PATH / f"round_{round_num}.csv"
            round_df.to_csv(round_filename, index=False)
            # print(f"\n✓ Saved {round_filename} with {len(round_results)} entries.")
        else:
            print(f"\n⚠ No results for round {round_num}.")

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    # print(f"Round results saved in: {ROUND_RESULTS_PATH}")
    # print(f"Client models saved in: {MODEL_DIR}")
    print("="*80)

def main():

    run_round_experiment()

if __name__ == "__main__":
    main()

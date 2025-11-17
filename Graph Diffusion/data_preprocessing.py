import h5py
import numpy as np
import glob
import os
import pickle
import json # Used to save simple parameters
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform

# --- Project Parameters ---
# IMPORTANT: Adjust the path if running from a different location relative to 'data'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data/GC_data") # Assuming 'Graph Diffusion' is a subfolder
OUTPUT_DIR = BASE_DIR

import sys
sys.path.append(os.path.join(BASE_DIR, '..'))
from Helpers import extract_metadata

directory = DATA_DIR 
file_paths = sorted(glob.glob(os.path.join(directory, '*.hdf5')))
print(f"Number of files in file_paths: {len(file_paths)}")

# --- Hyperparameters ---
W_TIME = 300
C_PATCH = 128
S_STRIDE = 150
CHANNEL_STEP = 4
GRAPH_THRESHOLD_D = 3

TRAIN_RATIO = 0.8
SIMULATED_ANOMALY_THRESHOLD = 0.8 # Used for simulated labels (not used here, but kept)

# --- 1. DATA LOADING AND INITIAL DOWNSAMPLING ---

print(f"--- 1. Loading and Initial Downsampling (W={W_TIME}, C_PATCH={C_PATCH}) ---")

all_data = []
file_metadata = []

if not file_paths:
    raise FileNotFoundError(f"No HDF5 files found in {directory}. Check the path.")

# Get the initial metadata from the first file
start_time, dt, dx, channels_raw, num_samples_raw = extract_metadata(file_paths[0])
fs = 1/dt
distance_array_m = channels_raw * dx
distance_array_km = distance_array_m / 1000

# Masking logic
dmin_km, dmax_km = 0, 30 
dist_mask = (distance_array_km >= dmin_km) & (distance_array_km <= dmax_km)

downsampled_channels = channels_raw[dist_mask][::CHANNEL_STEP]
C_TOTAL_DOWNSAMPLED = len(downsampled_channels)

print(f"Total Channels in file: {len(channels_raw)}")
print(f"Downsampled Channels (C_TOTAL_DOWNSAMPLED): {C_TOTAL_DOWNSAMPLED}")

for file_idx, file_path in enumerate(tqdm(file_paths, desc="Loading Files")):
    try:
        with h5py.File(file_path, 'r') as f:
            data = f['data'][:] # Load all data (Time, Channels)
            
            # Apply distance mask and downsampling
            data = data[:, dist_mask][::1, ::CHANNEL_STEP] # Spatial downsampling
            
            all_data.append(data)
            file_metadata.append({
                'file_idx': file_idx,
                'file_name': os.path.basename(file_path),
                'start_time': extract_metadata(file_path)[0],
                'time_steps': data.shape[0]
            })
    except Exception as e:
        print(f"\nError loading {file_path}. Skipping. Error: {e}")

combined_data = np.vstack(all_data)
print(f"Combined Data Shape (Time, Dist): {combined_data.shape}")

# --- 2. NORMALIZATION ---
print("\n--- 2. Normalization ---")
scaler = MinMaxScaler()
# Normalize the data and reshape back
normalized_data = scaler.fit_transform(combined_data.astype(np.float32).ravel().reshape(-1, 1)).reshape(combined_data.shape)

# --- 3. GRAPH CONSTRUCTION ---
print("\n--- 3. Graph Construction ---")

def build_index_based_adj(num_nodes, d_threshold):
    """Creates a sparse Adjacency Matrix based on index proximity (single row)."""
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if 0 < abs(i - j) <= d_threshold:
                adj[i, j] = 1.0 / (abs(i - j) + 1e-6) 
    
    A_tilde = adj + np.eye(num_nodes, dtype=np.float32)
    D = np.sum(A_tilde, axis=1)
    D_inv = np.diag(1.0 / D)
    A_hat = D_inv @ A_tilde
    return A_hat.astype(np.float32)

if C_TOTAL_DOWNSAMPLED < C_PATCH:
    print(f"Warning: Channel count {C_TOTAL_DOWNSAMPLED} is less than C_PATCH {C_PATCH}. Adjusting C_PATCH.")
    C_PATCH = C_TOTAL_DOWNSAMPLED
    
ADJACENCY_MATRIX = build_index_based_adj(C_PATCH, GRAPH_THRESHOLD_D)
print(f"Adjacency Matrix Shape: {ADJACENCY_MATRIX.shape}")

# --- 4. PATCH EXTRACTION AND TRAIN/TEST SPLIT ---
print("\n--- 4. Patch Extraction and Train/Test Split ---")

def create_patches_and_track(data_matrix, time_window, dist_window, stride, file_metadata):
    patches = []
    file_map_list = []
    
    time_steps, dist_channels = data_matrix.shape
    d_start = (dist_channels - dist_window) // 2 
    d_end = d_start + dist_window
    
    cumulative_time = 0
    file_map_index = 0

    for t in tqdm(range(0, time_steps - time_window + 1, stride), desc="Patching"):
        patch = data_matrix[t:t + time_window, d_start:d_end]
        
        if patch.shape == (time_window, dist_window):
            patches.append(patch)
            
            # Find which original HDF5 file this time step belongs to
            while file_map_index < len(file_metadata) - 1 and \
                  t + time_window > cumulative_time + file_metadata[file_map_index]['time_steps']:
                cumulative_time += file_metadata[file_map_index]['time_steps']
                file_map_index += 1
            
            file_map_list.append({
                'file_idx': file_metadata[file_map_index]['file_idx'],
                'file_name': file_metadata[file_map_index]['file_name'],
                'time_start_step': t
            })

    return np.array(patches), file_map_list # Return as list/array of dicts

# --- Apply Patching ---
X_all_raw, FILE_MAP_LIST = create_patches_and_track(
    normalized_data, W_TIME, C_PATCH, S_STRIDE, file_metadata
)

# Reshape for CNN/GNN input (batch_size, time_window, dist_channels_patch, 1)
X_all = X_all_raw[..., np.newaxis]

# --- Train/Test Split ---
N_PATCHES = X_all.shape[0]
N_TRAIN = int(N_PATCHES * TRAIN_RATIO)

X_train = X_all[:N_TRAIN]
X_test = X_all[N_TRAIN:]

FILE_MAP_TEST = FILE_MAP_LIST[N_TRAIN:]

print(f"Total Patches Created: {N_PATCHES}")
print(f"X_train Shape: {X_train.shape}")
print(f"X_test Shape: {X_test.shape}")

# --- 5. SAVING ARRAYS AND METADATA ---
print("\n--- 5. Saving Data and Config Files ---")

# Save NumPy arrays
np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test)
np.save(os.path.join(OUTPUT_DIR, 'ADJACENCY_MATRIX.npy'), ADJACENCY_MATRIX)

# Save test file map (for evaluation)
with open(os.path.join(OUTPUT_DIR, 'FILE_MAP_TEST.pkl'), 'wb') as f:
    pickle.dump(FILE_MAP_TEST, f)

# Save key configuration parameters (Hyperparameters)
config = {
    'W_TIME': W_TIME,
    'C_PATCH': C_PATCH,
    'C_TOTAL_DOWNSAMPLED': C_TOTAL_DOWNSAMPLED,
    'GRAPH_THRESHOLD_D': GRAPH_THRESHOLD_D,
}
with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
    json.dump(config, f)

print("Data preprocessing complete. Files saved to the Graph Diffusion directory.")
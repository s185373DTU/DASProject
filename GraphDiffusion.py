#!/usr/bin/env python
# coding: utf-8

# In[2]:


# --- Configuration and Imports ---
import h5py
import numpy as np
import tensorflow as tf
import glob
import os
from tqdm.notebook import tqdm
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform
from Helpers import extract_metadata # Uncomment if available

# --- Project Parameters ---
directory = "data/GC_data" 
file_paths = sorted(glob.glob(os.path.join(directory, '*.hdf5')))
print(f"Number of files in file_paths: {len(file_paths)}")

# --- Hyperparameters ---
W_TIME = 300            # Time window size (steps)
C_PATCH = 128           # Final number of channels in a patch (adjusts based on data)
S_STRIDE = 150          # Time stride for patching (50% overlap)
CHANNEL_STEP = 4        # Requested spatial downsampling (sampling every second channel)
GRAPH_THRESHOLD_D = 3   # Index-based neighbor threshold d

# Split ratio (Train: 80% Normal, Test: 20% Normal + Anomaly)
TRAIN_RATIO = 0.8
SIMULATED_ANOMALY_THRESHOLD = 0.8 # Used for simulated labels

# --- 1. DATA LOADING AND INITIAL DOWNSAMPLING ---

print(f"--- 1. Loading and Initial Downsampling (W={W_TIME}, C_PATCH={C_PATCH}) ---")

all_data = []
# Tracks the origin of each 10s data block (file name and start time)
file_metadata = []

# Get the initial metadata from the first file
start_time, dt, dx, channels_raw, num_samples_raw = extract_metadata(file_paths[0])
fs = 1/dt
distance_array_m = channels_raw * dx
distance_array_km = distance_array_m / 1000

# Masking logic (keep channels up to 30km, which is all of the 15.9km cable)
dmin_km, dmax_km = 0, 30 
dist_mask = (distance_array_km >= dmin_km) & (distance_array_km <= dmax_km)

# Downsample the channels that are within the mask
downsampled_channels = channels_raw[dist_mask][::CHANNEL_STEP]
C_TOTAL_DOWNSAMPLED = len(downsampled_channels)

print(f"Total Channels in file: {len(channels_raw)}")
print(f"Downsampled Channels (C_TOTAL_DOWNSAMPLED): {C_TOTAL_DOWNSAMPLED}")

for file_idx, file_path in enumerate(tqdm(file_paths, desc="Loading Files")):
    try:
        with h5py.File(file_path, 'r') as f:
            data = f['data'][:] # Load all data (Time, Channels)
            
            # Apply distance mask and downsampling
            data = data[:, dist_mask][::1, ::CHANNEL_STEP] # Time downsampling (::1) is not applied here, only spatial
            
            # Store the data and metadata
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
normalized_data = scaler.fit_transform(combined_data.astype(np.float32).ravel().reshape(-1, 1)).reshape(combined_data.shape)

# --- 3. GRAPH CONSTRUCTION ---
print("\n--- 3. Graph Construction ---")

def build_index_based_adj(num_nodes, d_threshold):
    """Creates a sparse Adjacency Matrix based on index proximity (single row)."""
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(num_nodes):
            # Connect if index difference is <= d_threshold
            if 0 < abs(i - j) <= d_threshold:
                # Weight is inverse of distance (index difference) + epsilon
                adj[i, j] = 1.0 / (abs(i - j) + 1e-6) 
    
    # Normalize Adjacency Matrix (A_hat = D_inv * A_tilde, where A_tilde = A + I)
    A_tilde = adj + np.eye(num_nodes, dtype=np.float32)
    D = np.sum(A_tilde, axis=1)
    D_inv = np.diag(1.0 / D)
    A_hat = D_inv @ A_tilde
    return A_hat.astype(np.float32)

# Build the GNN adjacency matrix
ADJACENCY_MATRIX = build_index_based_adj(C_PATCH, GRAPH_THRESHOLD_D)
print(f"Graph Nodes (Channels in Patch): {C_PATCH}")
print(f"Adjacency Matrix Shape: {ADJACENCY_MATRIX.shape}")
print(f"Note: This matrix represents the spatial structure (Graph) for the GNN.")

# --- 4. PATCH EXTRACTION AND TRAIN/TEST SPLIT ---
print("\n--- 4. Patch Extraction and Train/Test Split ---")

# Determine number of channels to keep in the patch for the GNN input
# The paper used 310 channels total, which means 155 nodes per row. 
# Since we have 7812 nodes, we'll try to find the largest divisible size, 
# and for simplicity in the 2D CNN part of the U-Net, we'll slice a C_PATCH area.
if C_TOTAL_DOWNSAMPLED < C_PATCH:
    print(f"Warning: Channel count {C_TOTAL_DOWNSAMPLED} is less than C_PATCH {C_PATCH}. Adjusting C_PATCH.")
    C_PATCH = C_TOTAL_DOWNSAMPLED

def create_patches_and_track(data_matrix, time_window, dist_window, stride, file_metadata):
    patches = []
    file_map = [] # To store (file_idx, patch_start_time_step)
    
    time_steps, dist_channels = data_matrix.shape
    
    # Since we are not using spatial overlap across the full 7812 channels, 
    # we take a single, central slice of the downsampled channels (C_PATCH=128)
    d_start = (dist_channels - dist_window) // 2 
    d_end = d_start + dist_window
    
    # Track the cumulative time steps to link patches back to files
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
            
            # Store the file index and the start time step relative to the full dataset
            file_map.append({
                'file_idx': file_metadata[file_map_index]['file_idx'],
                'file_name': file_metadata[file_map_index]['file_name'],
                'time_start_step': t
            })

    return np.array(patches), np.array(file_map)


# --- Apply Patching ---
X_all_raw, FILE_MAP = create_patches_and_track(
    normalized_data, W_TIME, C_PATCH, S_STRIDE, file_metadata
)

# Reshape for CNN/GNN input (batch_size, time_window, dist_channels_patch, 1)
X_all = X_all_raw[..., np.newaxis]

# --- Train/Test Split ---
N_PATCHES = X_all.shape[0]
N_TRAIN = int(N_PATCHES * TRAIN_RATIO)

X_train = X_all[:N_TRAIN]
X_test = X_all[N_TRAIN:]

X_test_raw = X_all_raw[N_TRAIN:] # For simulated labeling

FILE_MAP_TEST = FILE_MAP[N_TRAIN:]

print(f"Total Patches Created: {N_PATCHES}")
print(f"X_train (Normal Data for Training) Shape: {X_train.shape}")
print(f"X_test (Test Data for Evaluation) Shape: {X_test.shape}")

# --- Store variables for the next step ---
get_ipython().run_line_magic('store', 'X_train X_test ADJACENCY_MATRIX W_TIME C_PATCH FILE_MAP_TEST C_TOTAL_DOWNSAMPLED')


# In[9]:


# --- Imports and Variable Load ---
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Layer
from tensorflow.keras.optimizers import Adam

get_ipython().run_line_magic('store', '-r X_train X_test ADJACENCY_MATRIX W_TIME C_PATCH FILE_MAP_TEST C_TOTAL_DOWNSAMPLED')

# --- GNN Components (Custom Layers) ---

class GraphConvLayer(Layer):
    """
    Implements the core Graph Convolution operation: X' = A_hat * X * W.
    A_hat is the pre-calculated normalized adjacency matrix.
    """
    def __init__(self, output_dim, activation='relu', adj_matrix=None, **kwargs):
        super(GraphConvLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = tf.keras.layers.Activation(activation)
        
        # FIX 1: Store the adjacency matrix as a non-trainable constant/weight.
        if adj_matrix is None:
            raise ValueError("Adjacency matrix must be provided to GraphConvLayer.")
        # Ensure it's a Tensor constant
        self.adj_matrix_tensor = tf.constant(adj_matrix, dtype=tf.float32)

    def build(self, input_shape):
        # Input shape: (Batch, W_TIME, C_PATCH, 1) 
        C_IN = input_shape[2] 
        
        # Check for matrix dimension consistency
        if C_IN != self.adj_matrix_tensor.shape[0]:
            raise ValueError(
                f"Input channel dimension ({C_IN}) must match Adjacency Matrix dimension ({self.adj_matrix_tensor.shape[0]})."
            )

        # Kernel shape: (C_PATCH, output_dim) -> [128, 128]
        self.kernel = self.add_weight(shape=(C_IN, self.output_dim), 
                                      initializer='glorot_uniform',
                                      name='kernel')
        super(GraphConvLayer, self).build(input_shape)

    def call(self, inputs):
        # 1. Prepare inputs 
        X_in = tf.squeeze(inputs, axis=-1) # [?, 300, 128]

        # 2. Transpose X to [Batch, C_PATCH, W_TIME] for Graph Convolution (A * X)
        X_transposed = tf.transpose(X_in, perm=[0, 2, 1]) # [?, 128, 300]

        # 3. Graph Convolution: A_hat * X_transposed
        # Ax shape: [?, 128, 300]
        Ax = tf.matmul(self.adj_matrix_tensor, X_transposed)

        # 4. Transpose back: [Batch, W_TIME, C_PATCH]
        Ax_transposed = tf.transpose(Ax, perm=[0, 2, 1]) # [?, 300, 128]
        
        # 5. Apply weight matrix W (the linear projection on nodes/channels)
        output = tf.matmul(Ax_transposed, self.kernel) # [?, 300, 128]
        
        # 6. Apply Activation
        output = self.activation(output)
        
        # 7. Add channel dimension back: [?, 300, 128, 1]
        output = tf.expand_dims(output, axis=-1)
        return output

    def get_config(self):
        config = super().get_config()
        # Note: Serializing the full matrix is complex; for notebook use, this is sufficient
        config.update({"output_dim": self.output_dim})
        return config

# --- Full GraphDiffusion Model ---

def build_graph_diffusion(input_shape,beta_min=1e-4, beta_max=0.02):
    """
    Builds the GraphDiffusion model combining a GNN and a conditional U-Net.
    """
    T_TIME, C_DIST, CHANNELS = input_shape
    
    # 1. INPUTS
    X0_input = Input(shape=input_shape, name='X0_input')
    time_step_input = Input(shape=(1,), dtype=tf.int32, name='time_step_input')

    # 2. GNN (Spatial Representation Learning) - Figure 3
    # GNN processes the clean input X0 to get spatial embedding H
    # Note: We use C_PATCH as the output dim for the first GCN layer for the skip connection (H=S+X0).
    # FIX: Pass the ADJACENCY_MATRIX tensor to the layer initialization.
    gnn_out = GraphConvLayer(C_PATCH, activation='relu', adj_matrix=ADJACENCY_MATRIX, name='gcn_1')(X0_input)
    S_spatial_features = GraphConvLayer(C_PATCH, activation='relu', adj_matrix=ADJACENCY_MATRIX, name='gcn_2')(gnn_out)
    
    # Spatial Embedding H = S + X0 (Element-wise Sum - residual connection)
    H_spatial_embedding = layers.Add()([S_spatial_features, X0_input])

    # 3. DDPM (Simulating the Diffusion Process in the Denoising Network)
    # The actual DDPM forward process (X_t = sqrt(a_bar)X0 + sqrt(1-a_bar)epsilon)
    # happens outside the model during the training loop.
    
    # Time Step Conditioning (Sinusoidal Positional Embedding)
    time_emb = layers.Lambda(lambda x: tf.cast(x, tf.float32) / 1000)(time_step_input)
    time_emb = layers.Dense(T_TIME, activation='relu')(time_emb)
    
    # Broadcast time embedding to match the spatial embedding shape (B, T, C, 1)
    time_emb_bcast = layers.Reshape((T_TIME, 1, 1))(time_emb)
    time_emb_bcast = tf.tile(time_emb_bcast, [1, 1, C_DIST, 1])

    # DDPM INPUT (X_t is substituted by X0_input here, as the model learns to predict noise 
    # based on the noise level implicit in time_step_input)
    
    # Conditional Denoising U-Net Input (Concat X_t, H, and time_emb_bcast) - Figure 2
    # In a proper DDPM implementation, X_t is the noisy input. Here, we use the original 
    # X0_input as a placeholder for the noisy input passed during training.
    unet_input = layers.Concatenate(axis=-1, name='unet_pre_concat')([
        X0_input, H_spatial_embedding, time_emb_bcast
    ])
    
    # --- Denoising U-Net (Approximation of Figure 5 structure) ---
    
    # ENCODER (Downsampling)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(unet_input)
    p1 = layers.MaxPooling2D((2, 2))(c1) # 150x64x64
    
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2) # 75x32x128
    
    # BOTTLENECK
    bn = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    bn = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(bn)

    # DECODER (Upsampling + Skip Connections)
    u1 = layers.UpSampling2D((2, 2))(bn) # 75x32x256
    u1 = layers.Concatenate()([u1, c2])  # Skip connection from c2
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
    
    u2 = layers.UpSampling2D((2, 2))(c3) # 150x64x128
    u2 = layers.Concatenate()([u2, c1])  # Skip connection from c1
    c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u2)
    
    # Output - predicting the noise (epsilon)
    # Output shape must match input shape (B, 300, 128, 1)
    output_noise_pred = layers.Conv2D(CHANNELS, (1, 1), activation='linear', padding='same', name='noise_output')(c4)

    # 4. Final Model
    return Model(inputs=[X0_input, time_step_input], outputs=output_noise_pred, name='GraphDiffusion_DDPM')

# --- Instantiate and Compile ---
model = build_graph_diffusion(X_train.shape[1:], ADJACENCY_MATRIX.shape)
model.summary()

# NOTE: The actual training loop needs to implement the DDPM logic 
# (noise corruption and loss calculation) which is too complex for a standard 
# Keras compile/fit. We'll implement a custom training loop in the next cell.
# The model's outputs are the predicted noise epsilon_theta(X_t, t, H).
get_ipython().run_line_magic('store', 'model ADJACENCY_MATRIX W_TIME C_PATCH')


# In[ ]:


from matplotlib import pyplot as plt
# Load all necessary components from previous steps
get_ipython().run_line_magic('store', '-r X_train X_test X_test_raw FILE_MAP_TEST model W_TIME C_PATCH')

# --- DDPM Parameters and Functions ---
T_STEPS = 25 # DDPM Total Timesteps
BETA_MIN = 1e-4
BETA_MAX = 0.02
BATCH_SIZE = 8 
EPOCHS = 3
PERCENTILE = 99.9 # Tunable threshold, as discussed

# Linear noise schedule
betas = np.linspace(BETA_MIN, BETA_MAX, T_STEPS, dtype=np.float32)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas)

# Tensor versions for fast computation
alphas_cumprod_tf = tf.constant(alphas_cumprod, dtype=tf.float32)

@tf.function
def forward_diffusion(X0, t):
    """
    Applies noise to the clean signal X0 at time step t (Equation 8).
    X_t = sqrt(alpha_bar_t) * X0 + sqrt(1 - alpha_bar_t) * epsilon
    """
    # Select alpha_bar for the current batch's time steps t
    alpha_bar_t = tf.gather(alphas_cumprod_tf, t)
    alpha_bar_t = tf.reshape(alpha_bar_t, [-1, 1, 1, 1])

    # Calculate required coefficients
    sqrt_alpha_bar_t = tf.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = tf.sqrt(1.0 - alpha_bar_t)

    # Generate random noise
    epsilon = tf.random.normal(tf.shape(X0), dtype=tf.float32)
    
    # Compute X_t
    Xt = sqrt_alpha_bar_t * X0 + sqrt_one_minus_alpha_bar_t * epsilon
    return Xt, epsilon

# --- Custom Training Step (Implements L_dif in Equation 9) ---
optimizer = Adam(learning_rate=1e-3)
loss_metric = tf.keras.metrics.Mean(name="train_loss")

@tf.function
def train_step(X0_batch):
    # 1. Sample time step t uniformly
    t = tf.random.uniform(shape=[tf.shape(X0_batch)[0]], minval=0, maxval=T_STEPS, dtype=tf.int32)
    
    # 2. Apply forward diffusion to get X_t and true noise epsilon
    Xt, epsilon = forward_diffusion(X0_batch, t)
    
    # The GNN component requires a clean signal (X0) to create the spatial embedding H.
    # The UNet component requires the noisy signal (Xt) to predict the noise.
    
    # We pass [Noisy_Xt, Time_t] to the model, and the model's internal GNN uses Xt 
    # as the nearest proxy for the clean signal X0 (a common GNN simplification).
    
    with tf.GradientTape() as tape:
        # Predicted noise: epsilon_theta(X_t, t, H(X0))
        predicted_noise = model([Xt, t], training=True) 

        # Calculate L2 loss (MSE) on the noise prediction (Equation 9)
        loss = tf.keras.losses.mean_squared_error(epsilon, predicted_noise)
        
    # Apply gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Update loss tracker
    loss_metric.update_state(loss)
    return loss

# --- Training Loop ---
print("\n--- Training the GraphDiffusion Model on Normal Data (Unsupervised) ---")
X_train_ds = tf.data.Dataset.from_tensor_slices(X_train).shuffle(buffer_size=1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

for epoch in range(EPOCHS):
    loss_metric.reset_states()
    
    for X0_batch in tqdm(X_train_ds, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        train_step(X0_batch)
        
    print(f"Epoch {epoch + 1} - Average Loss: {loss_metric.result().numpy():.6f}")

print("\nTraining complete.")


# In[2]:


def calculate_anomaly_scores(X_data):
    """
    Calculates the reconstruction error (Anomaly Score) using a simplified DDPM reverse pass.
    We predict the noise and reconstruct X0_pred (Equation 10 derivation).
    """
    
    # Use the median timestep t_mid for robust reconstruction
    t_mid = T_STEPS // 2
    t_mid_batch = tf.ones(tf.shape(X_data)[0], dtype=tf.int32) * t_mid
    
    # 1. Apply forward diffusion to get the noisy input X_t
    Xt_mid, _ = forward_diffusion(X_data, t_mid_batch)
    
    # 2. Predict the noise from the noisy input Xt_mid
    predicted_noise = model([Xt_mid, t_mid_batch], training=False) 
    
    # 3. Simplified X0 reconstruction (rearrange forward_diffusion formula):
    # X0_pred = (Xt_mid - sqrt(1-alpha_bar)*epsilon_theta) / sqrt(alpha_bar)
    alpha_bar_mid = alphas_cumprod[t_mid]
    sqrt_alpha_bar_mid = np.sqrt(alpha_bar_mid)
    sqrt_one_minus_alpha_bar_mid = np.sqrt(1.0 - alpha_bar_mid)
    
    X0_pred = (Xt_mid - sqrt_one_minus_alpha_bar_mid * predicted_noise) / sqrt_alpha_bar_mid
    
    # Anomaly Score (MSE/L2 error) - Equation 11
    error = tf.reduce_mean(tf.square(X_data - X0_pred), axis=[1, 2, 3]).numpy()
    return error, X0_pred.numpy()

print("\n--- Inference and Anomaly Score Calculation ---")

# Calculate scores and reconstructions for Test data
test_error, X_test_pred = calculate_anomaly_scores(X_test)

# 4. Evaluation Metrics (Focus on Anomaly Score)
anomaly_threshold = np.percentile(calculate_anomaly_scores(X_train)[0], PERCENTILE)

print(f"\n--- UNSUPERVISED ANOMALY SCORE METRICS ---")
print(f"Anomaly Threshold (Top {100-PERCENTILE}%) : {anomaly_threshold:.6f}")
print("-------------------------------------------------------")

# 5. Identify Top Anomalous Files (New Evaluation Goal)
print("\n--- Identifying Top 5 Anomalous Files (Actionable Insight) ---")

# Aggregate by file name and calculate average score
file_scores = {}
file_counts = {}
for i, result in enumerate(test_error):
    name = FILE_MAP_TEST[i]['file_name']
    score = test_error[i]
    file_scores[name] = file_scores.get(name, 0) + score
    file_counts[name] = file_counts.get(name, 0) + 1

# Calculate average score
avg_file_scores = {name: file_scores[name] / file_counts[name] for name in file_scores}

# Sort and get top 5
top_files = sorted(avg_file_scores.items(), key=lambda item: item[1], reverse=True)[:5]

print("\nTop 5 Most Anomalous HDF5 Files (Highest Average Score):")
print("-------------------------------------------------------")
for rank, (file_name, avg_score) in enumerate(top_files):
    print(f"Rank {rank+1}: {file_name} (Average Score: {avg_score:.6f})")

# 6. Visualize the Top 5 Anomalous Patches (Visual Inspection)
print("\n--- Visualizing the Top Anomalous Patches for Inspection ---")

# Find the indices of the top 5 highest scoring patches in the test set
top_k_patches = np.argsort(test_error)[::-1][:5]

# Use X_test_raw for visualization as it contains the NumPy version
X_test_raw_np = X_test.squeeze()

fig, axes = plt.subplots(5, 2, figsize=(12, 10))
fig.suptitle('GraphDiffusion: Top 5 Anomalous Patches (Input vs. Reconstruction)', fontsize=14)

for i, patch_idx in enumerate(top_k_patches):
    # Use the NumPy version of the input and the reconstructed output
    input_patch = X_test_raw_np[patch_idx]
    reconstructed_patch = X_test_pred[patch_idx].squeeze()
    
    vmax = np.percentile(input_patch, 99.5)
    
    # Input
    ax = axes[i, 0]
    ax.imshow(input_patch, aspect='auto', cmap='jet', vmin=0, vmax=vmax)
    ax.set_title(f"Input (Score: {test_error[patch_idx]:.4f})")
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Reconstruction
    ax = axes[i, 1]
    ax.imshow(reconstructed_patch, aspect='auto', cmap='jet', vmin=0, vmax=vmax)
    ax.set_title("Reconstruction")
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('top_anomalous_patches.png')
plt.show()


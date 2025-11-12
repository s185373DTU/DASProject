import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Layer
from tensorflow.keras import layers, Model
import numpy as np
import os
import pickle
import json
from tqdm import tqdm
from matplotlib import pyplot as plt

# --- 1. Load Data, Config, and Weights ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = BASE_DIR

print("--- 1. Loading Data, Config, and Weights ---")
X_test = np.load(os.path.join(OUTPUT_DIR, 'X_test.npy'))
ADJACENCY_MATRIX = np.load(os.path.join(OUTPUT_DIR, 'ADJACENCY_MATRIX.npy'))

with open(os.path.join(OUTPUT_DIR, 'FILE_MAP_TEST.pkl'), 'rb') as f:
    FILE_MAP_TEST = pickle.load(f)

with open(os.path.join(OUTPUT_DIR, 'config.json'), 'r') as f:
    config = json.load(f)

with open(os.path.join(OUTPUT_DIR, 'anomaly_threshold.json'), 'r') as f:
    threshold_data = json.load(f)
    anomaly_threshold = threshold_data['threshold']
    PERCENTILE = threshold_data['percentile']

W_TIME = config['W_TIME']
C_PATCH = config['C_PATCH']

# --- 2. Re-define Custom GNN Components and Model Structure (for loading weights) ---

class GraphConvLayer(Layer):
    """
    Must be defined again to load the model weights correctly.
    """
    def __init__(self, output_dim, activation='relu', adj_matrix=None, **kwargs):
        super(GraphConvLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = tf.keras.layers.Activation(activation)
        # adj_matrix is not used after loading, but must be accepted by __init__
        if adj_matrix is None:
            # We must set a placeholder when reconstructing the model
            adj_matrix = np.eye(output_dim, dtype=np.float32) 
        self.adj_matrix_tensor = tf.constant(adj_matrix, dtype=tf.float32)

    def build(self, input_shape):
        C_IN = input_shape[2] 
        self.kernel = self.add_weight(shape=(C_IN, self.output_dim), 
                                      initializer='glorot_uniform',
                                      name='kernel')
        super(GraphConvLayer, self).build(input_shape)
        
    # The call method needs to be exactly as in train_model.py
    def call(self, inputs):
        X_in = tf.squeeze(inputs, axis=-1)
        X_transposed = tf.transpose(X_in, perm=[0, 2, 1])
        Ax = tf.matmul(self.adj_matrix_tensor, X_transposed)
        Ax_transposed = tf.transpose(Ax, perm=[0, 2, 1])
        output = tf.matmul(Ax_transposed, self.kernel)
        output = self.activation(output)
        output = tf.expand_dims(output, axis=-1)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"output_dim": self.output_dim, "activation": self.activation.name})
        return config

def build_graph_diffusion(input_shape, adj_matrix):
    T_TIME, C_DIST, CHANNELS = input_shape
    X0_input = Input(shape=input_shape, name='X0_input')
    time_step_input = Input(shape=(1,), dtype=tf.int32, name='time_step_input')
    
    # Use the loaded ADJACENCY_MATRIX here!
    gcn_params = {'adj_matrix': adj_matrix, 'activation': 'relu'}
    gnn_out = GraphConvLayer(C_PATCH, **gcn_params, name='gcn_1')(X0_input)
    S_spatial_features = GraphConvLayer(C_PATCH, **gcn_params, name='gcn_2')(gnn_out)
    H_spatial_embedding = layers.Add()([S_spatial_features, X0_input])

    time_emb = layers.Lambda(lambda x: tf.cast(x, tf.float32) / 1000)(time_step_input)
    time_emb = layers.Dense(T_TIME, activation='relu')(time_emb)
    time_emb_bcast = layers.Reshape((T_TIME, 1, 1))(time_emb)
    time_emb_bcast = tf.tile(time_emb_bcast, [1, 1, C_DIST, 1])

    unet_input = layers.Concatenate(axis=-1, name='unet_pre_concat')([
        X0_input, H_spatial_embedding, time_emb_bcast
    ])
    
    # Denoising U-Net
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(unet_input)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    bn = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    bn = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(bn)
    u1 = layers.UpSampling2D((2, 2))(bn)
    u1 = layers.Concatenate()([u1, c2])
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
    u2 = layers.UpSampling2D((2, 2))(c3)
    u2 = layers.Concatenate()([u2, c1])
    c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u2)
    output_noise_pred = layers.Conv2D(1, (1, 1), activation='linear', padding='same', name='noise_output')(c4)

    return Model(inputs=[X0_input, time_step_input], outputs=output_noise_pred, name='GraphDiffusion_DDPM')


model = build_graph_diffusion(X_test.shape[1:], ADJACENCY_MATRIX)
model.load_weights(os.path.join(OUTPUT_DIR, 'graph_diffusion_weights.h5'))
print("Model built and weights loaded successfully.")


# --- 3. DDPM Inference Logic ---
T_STEPS = 25
BETA_MIN = 1e-4
BETA_MAX = 0.02

betas = np.linspace(BETA_MIN, BETA_MAX, T_STEPS, dtype=np.float32)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas)

alphas_cumprod_tf = tf.constant(alphas_cumprod, dtype=np.float32)

@tf.function
def forward_diffusion(X0, t):
    alpha_bar_t = tf.gather(alphas_cumprod_tf, t)
    alpha_bar_t = tf.reshape(alpha_bar_t, [-1, 1, 1, 1])
    sqrt_alpha_bar_t = tf.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = tf.sqrt(1.0 - alpha_bar_t)
    epsilon = tf.random.normal(tf.shape(X0), dtype=tf.float32)
    Xt = sqrt_alpha_bar_t * X0 + sqrt_one_minus_alpha_bar_t * epsilon
    return Xt, epsilon


def calculate_anomaly_scores(X_data):
    """Calculates the reconstruction error (Anomaly Score) using the trained model."""
    t_mid = T_STEPS // 2
    t_mid_batch = tf.ones(tf.shape(X_data)[0], dtype=tf.int32) * t_mid
    
    Xt_mid, _ = forward_diffusion(tf.cast(X_data, tf.float32), t_mid_batch)
    predicted_noise = model([Xt_mid, t_mid_batch], training=False) 
    
    alpha_bar_mid = alphas_cumprod[t_mid]
    sqrt_alpha_bar_mid = np.sqrt(alpha_bar_mid)
    sqrt_one_minus_alpha_bar_mid = np.sqrt(1.0 - alpha_bar_mid)
    
    X0_pred = (Xt_mid - sqrt_one_minus_alpha_bar_mid * predicted_noise) / sqrt_alpha_bar_mid
    
    error = tf.reduce_mean(tf.square(tf.cast(X_data, tf.float32) - X0_pred), axis=[1, 2, 3]).numpy()
    return error, X0_pred.numpy()

# --- 4. Inference and Evaluation ---

print("\n--- 4. Inference and Anomaly Score Calculation on Test Data ---")
BATCH_SIZE = 8 # Use a small batch size for inference to manage memory
X_test_ds = tf.data.Dataset.from_tensor_slices(X_test).batch(BATCH_SIZE)

test_error = []
X_test_pred_list = []

for X_batch in tqdm(X_test_ds, desc="Calculating Test Scores"):
    error_batch, pred_batch = calculate_anomaly_scores(X_batch)
    test_error.extend(error_batch.tolist())
    X_test_pred_list.append(pred_batch)

test_error = np.array(test_error)
X_test_pred = np.vstack(X_test_pred_list)

print(f"\n--- UNSUPERVISED ANOMALY SCORE METRICS ---")
print(f"Anomaly Threshold (Top {100-PERCENTILE}%) : {anomaly_threshold:.6f}")
print("-------------------------------------------------------")

# --- 5. Identify Top Anomalous Files ---
print("\n--- 5. Identifying Top 5 Anomalous Files (Actionable Insight) ---")

file_scores = {}
file_counts = {}
for i, result in enumerate(test_error):
    name = FILE_MAP_TEST[i]['file_name']
    score = test_error[i]
    file_scores[name] = file_scores.get(name, 0) + score
    file_counts[name] = file_counts.get(name, 0) + 1

avg_file_scores = {name: file_scores[name] / file_counts[name] for name in file_scores}
top_files = sorted(avg_file_scores.items(), key=lambda item: item[1], reverse=True)[:5]

print("\nTop 5 Most Anomalous HDF5 Files (Highest Average Score):")
print("-------------------------------------------------------")
for rank, (file_name, avg_score) in enumerate(top_files):
    print(f"Rank {rank+1}: {file_name} (Average Score: {avg_score:.6f})")

# --- 6. Visualize the Top 5 Anomalous Patches ---
print("\n--- 6. Visualizing the Top Anomalous Patches for Inspection ---")

top_k_patches = np.argsort(test_error)[::-1][:5]
X_test_raw_np = X_test.squeeze() # Original input patches

fig, axes = plt.subplots(5, 2, figsize=(12, 10))
fig.suptitle('GraphDiffusion: Top 5 Anomalous Patches (Input vs. Reconstruction)', fontsize=14)

for i, patch_idx in enumerate(top_k_patches):
    input_patch = X_test_raw_np[patch_idx]
    reconstructed_patch = X_test_pred[patch_idx].squeeze()
    
    # Use a conservative vmax based on the input patch
    vmax = np.percentile(input_patch, 99.5)
    
    # Input
    ax = axes[i, 0]
    im = ax.imshow(input_patch, aspect='auto', cmap='jet', vmin=0, vmax=vmax)
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
plt.savefig(os.path.join(OUTPUT_DIR, 'top_anomalous_patches.png'))
print(f"Visualization saved to {os.path.join(OUTPUT_DIR, 'top_anomalous_patches.png')}")
# plt.show() # Uncomment to show plot during interactive execution
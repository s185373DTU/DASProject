import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Layer
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import pickle
import json
from tqdm import tqdm

# --- 1. Load Data and Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = BASE_DIR

print("--- 1. Loading Preprocessed Data and Config ---")
X_train = np.load(os.path.join(OUTPUT_DIR, 'X_train.npy'))
ADJACENCY_MATRIX = np.load(os.path.join(OUTPUT_DIR, 'ADJACENCY_MATRIX.npy'))

with open(os.path.join(OUTPUT_DIR, 'config.json'), 'r') as f:
    config = json.load(f)

W_TIME = config['W_TIME']
C_PATCH = config['C_PATCH']

# --- 2. GNN Components (Custom Layers) ---

class GraphConvLayer(Layer):
    """
    Implements the core Graph Convolution operation: X' = A_hat * X * W.
    A_hat is the pre-calculated normalized adjacency matrix.
    """
    def __init__(self, output_dim, activation='relu', adj_matrix=None, **kwargs):
        super(GraphConvLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = tf.keras.layers.Activation(activation)
        
        if adj_matrix is None:
            raise ValueError("Adjacency matrix must be provided to GraphConvLayer.")
        # Store the adjacency matrix as a non-trainable constant
        self.adj_matrix_tensor = tf.constant(adj_matrix, dtype=tf.float32)

    def build(self, input_shape):
        C_IN = input_shape[2] 
        if C_IN != self.adj_matrix_tensor.shape[0]:
            raise ValueError(
                f"Input channel dim ({C_IN}) must match Adj Matrix dim ({self.adj_matrix_tensor.shape[0]})."
            )

        self.kernel = self.add_weight(shape=(C_IN, self.output_dim), 
                                      initializer='glorot_uniform',
                                      name='kernel')
        super(GraphConvLayer, self).build(input_shape)

    def call(self, inputs):
        X_in = tf.squeeze(inputs, axis=-1)
        X_transposed = tf.transpose(X_in, perm=[0, 2, 1]) # [Batch, C_PATCH, W_TIME]
        Ax = tf.matmul(self.adj_matrix_tensor, X_transposed)
        Ax_transposed = tf.transpose(Ax, perm=[0, 2, 1]) # [Batch, W_TIME, C_PATCH]
        output = tf.matmul(Ax_transposed, self.kernel)
        output = self.activation(output)
        output = tf.expand_dims(output, axis=-1)
        return output

    def get_config(self):
        config = super().get_config()
        # NOTE: Cannot serialize the full matrix here. Must pass it during loading.
        config.update({"output_dim": self.output_dim, "activation": self.activation.name})
        return config

# --- 3. Full GraphDiffusion Model Definition ---

def build_graph_diffusion(input_shape, adj_matrix):
    T_TIME, C_DIST, CHANNELS = input_shape
    
    # 1. INPUTS
    X0_input = Input(shape=input_shape, name='X0_input')
    time_step_input = Input(shape=(1,), dtype=tf.int32, name='time_step_input')

    # 2. GNN (Spatial Representation Learning)
    gcn_params = {'adj_matrix': adj_matrix, 'activation': 'relu'}
    gnn_out = GraphConvLayer(C_PATCH, **gcn_params, name='gcn_1')(X0_input)
    S_spatial_features = GraphConvLayer(C_PATCH, **gcn_params, name='gcn_2')(gnn_out)
    
    H_spatial_embedding = layers.Add()([S_spatial_features, X0_input])

    # 3. Time Step Conditioning
    time_emb = layers.Lambda(lambda x: tf.cast(x, tf.float32) / 1000)(time_step_input)
    time_emb = layers.Dense(T_TIME, activation='relu')(time_emb)
    time_emb_bcast = layers.Reshape((T_TIME, 1, 1))(time_emb)
    time_emb_bcast = tf.tile(time_emb_bcast, [1, 1, C_DIST, 1])

    # Conditional Denoising U-Net Input (Concat X_t, H, and time_emb_bcast)
    unet_input = layers.Concatenate(axis=-1, name='unet_pre_concat')([
        X0_input, H_spatial_embedding, time_emb_bcast
    ])
    
    # --- Denoising U-Net ---
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
    
    output_noise_pred = layers.Conv2D(CHANNELS, (1, 1), activation='linear', padding='same', name='noise_output')(c4)

    return Model(inputs=[X0_input, time_step_input], outputs=output_noise_pred, name='GraphDiffusion_DDPM')

# --- 4. DDPM Parameters and Functions ---
T_STEPS = 25 # DDPM Total Timesteps
BETA_MIN = 1e-4
BETA_MAX = 0.02
BATCH_SIZE = 8 
EPOCHS = 3
PERCENTILE = 99.9 # For anomaly threshold saving

# Linear noise schedule
betas = np.linspace(BETA_MIN, BETA_MAX, T_STEPS, dtype=np.float32)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas)

alphas_cumprod_tf = tf.constant(alphas_cumprod, dtype=tf.float32)

@tf.function
def forward_diffusion(X0, t):
    """X_t = sqrt(alpha_bar_t) * X0 + sqrt(1 - alpha_bar_t) * epsilon"""
    alpha_bar_t = tf.gather(alphas_cumprod_tf, t)
    alpha_bar_t = tf.reshape(alpha_bar_t, [-1, 1, 1, 1])

    sqrt_alpha_bar_t = tf.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = tf.sqrt(1.0 - alpha_bar_t)

    epsilon = tf.random.normal(tf.shape(X0), dtype=tf.float32)
    
    Xt = sqrt_alpha_bar_t * X0 + sqrt_one_minus_alpha_bar_t * epsilon
    return Xt, epsilon

# --- 5. Custom Training Loop ---
print("--- 5. Building and Training the Model ---")
model = build_graph_diffusion(X_train.shape[1:], ADJACENCY_MATRIX)
optimizer = Adam(learning_rate=1e-3)
loss_metric = tf.keras.metrics.Mean(name="train_loss")

# Required for model's GNN component, passed via the custom call logic:
# We wrap the model call logic in the `train_step` function.

@tf.function
def train_step(X0_batch):
    t = tf.random.uniform(shape=[tf.shape(X0_batch)[0]], minval=0, maxval=T_STEPS, dtype=tf.int32)
    Xt, epsilon = forward_diffusion(X0_batch, t)
    
    with tf.GradientTape() as tape:
        # Pass Xt as the 'X0_input' placeholder; the model uses it to get spatial features (H)
        predicted_noise = model([Xt, t], training=True) 

        loss = tf.keras.losses.mean_squared_error(epsilon, predicted_noise)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    loss_metric.update_state(loss)
    return loss

# --- Training Execution ---
print(f"\n--- Starting Training ({EPOCHS} Epochs) ---")
X_train_ds = tf.data.Dataset.from_tensor_slices(X_train).shuffle(buffer_size=1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

for epoch in range(EPOCHS):
    loss_metric.reset_states()
    
    # Reset the dataset iterator for each epoch
    for X0_batch in tqdm(X_train_ds, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        train_step(X0_batch)
        
    print(f"Epoch {epoch + 1} - Average Loss: {loss_metric.result().numpy():.6f}")

print("\nTraining complete.")

# --- 6. Save Model and Threshold ---

# Calculate anomaly threshold on the training data (required for evaluation)
print("--- 6. Calculating and Saving Anomaly Threshold ---")

# Re-define forward_diffusion and anomaly score calculation for thresholding
alphas_cumprod_np = alphas_cumprod
T_STEPS_NP = T_STEPS

def calculate_anomaly_scores(X_data):
    """Calculates the reconstruction error (Anomaly Score) on a given dataset."""
    t_mid = T_STEPS_NP // 2
    t_mid_batch = tf.ones(tf.shape(X_data)[0], dtype=tf.int32) * t_mid
    
    # 1. Apply forward diffusion to get the noisy input X_t
    Xt_mid, _ = forward_diffusion(X_data, t_mid_batch)
    
    # 2. Predict the noise from the noisy input Xt_mid
    predicted_noise = model([Xt_mid, t_mid_batch], training=False) 
    
    # 3. Simplified X0 reconstruction 
    alpha_bar_mid = alphas_cumprod_np[t_mid]
    sqrt_alpha_bar_mid = np.sqrt(alpha_bar_mid)
    sqrt_one_minus_alpha_bar_mid = np.sqrt(1.0 - alpha_bar_mid)
    
    X0_pred = (Xt_mid - sqrt_one_minus_alpha_bar_mid * predicted_noise) / sqrt_alpha_bar_mid
    
    # Anomaly Score (MSE/L2 error)
    error = tf.reduce_mean(tf.square(tf.cast(X_data, tf.float32) - X0_pred), axis=[1, 2, 3]).numpy()
    return error

# Calculate scores on the whole training set (may take time)
train_scores = []
for X0_batch in tqdm(X_train_ds, desc="Calculating Train Scores for Threshold"):
    train_scores.extend(calculate_anomaly_scores(X0_batch).tolist())
    
anomaly_threshold = np.percentile(train_scores, PERCENTILE)

# Save the threshold
with open(os.path.join(OUTPUT_DIR, 'anomaly_threshold.json'), 'w') as f:
    json.dump({'threshold': float(anomaly_threshold), 'percentile': PERCENTILE}, f)

# Save the trained model weights (better than saving the full model due to custom layer)
model.save_weights(os.path.join(OUTPUT_DIR, 'graph_diffusion_weights.h5'))
print("\nTraining weights and anomaly threshold saved.")
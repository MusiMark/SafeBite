import numpy as np
import pandas as pd
import joblib
import os
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from fastapi import HTTPException

# --- Config (Mirror from original scripts) ---
FEATURE_COLS = [
    'T', 'H', 'PMS1', 'PMS2_5', 'PMS10', 'CO2', 'NO2', 'CO', 'VoC', 'C2H5OH',
    'pm25_max_2min', 'pm25_mean_2min', 'co2_max_2min', 'co2_mean_1min',
    'co2_rise_1to2min', 'T_rise_1min', 'pm10_mean_30s', 'pm10_spike',
    'H_mean_2min', 'voc_mean_1min', 'ethanol_mean_1min'
]

# --- Actions Definition ---
ACTIONS = [
    "do_nothing",
    "open_window",
    "turn_on_fan",
    "reduce_stove_load",
    "alert_user",
    "activate_air_purifier",
]

# --- Define a simple Contextual Bandit Class (LinUCB) ---
class LinearContextualBandit:
    def __init__(self, n_features, actions, alpha=1.0):
        self.n_features = n_features
        self.actions = actions
        self.alpha = alpha # Exploration parameter (confidence bound scale)

        # Initialize parameters for LinUCB for each action
        # A[action] is the covariance matrix (d x d), initialized as identity * alpha
        self.A = {a: np.identity(self.n_features) for a in self.actions}
        # b[action] is the response vector (d x 1), initialized as zeros
        self.b = {a: np.zeros(self.n_features) for a in self.actions} # Changed to 1D array (d,)
        # A_inv[action] is the inverse of A[action], initialized as identity / alpha
        self.A_inv = {a: np.identity(self.n_features) for a in self.actions}
        # theta[action] is the learned weight vector (d x 1), initialized as zeros
        self.theta = {a: np.zeros(self.n_features) for a in self.actions} # Changed to 1D array (d,)

        # To store feature stats for normalization (optional but often helpful)
        self.feature_means = np.zeros(n_features)
        self.feature_stds = np.ones(n_features)
        self.n_samples_seen = 0
        # To track counts for potential analysis
        self.action_counts = {a: 0 for a in self.actions}

    def _prepare_features(self, context_vector):
        # Normalize features based on running stats (simple approach)
        if self.n_samples_seen > 0:
            return (context_vector - self.feature_means) / (self.feature_stds + 1e-8)
        else:
            return context_vector # Return as-is if no stats collected yet

    def update_feature_stats(self, contexts):
        # Update running mean and std for normalization
        # contexts shape: (n_samples, n_features)
        if contexts.ndim == 1:
            contexts = contexts.reshape(1, -1)
        self.feature_means = (self.feature_means * self.n_samples_seen + contexts.sum(axis=0)) / (self.n_samples_seen + len(contexts))
        # Simplified std update - ideally use Welford's method for online calculation
        # Approximation: weighted average of old and new batch stds
        current_batch_std = contexts.std(axis=0)
        if self.n_samples_seen > 0:
             # Weighted variance update (approximate)
             old_var = self.feature_stds**2
             new_var = current_batch_std**2
             combined_var = ((self.n_samples_seen - 1) * old_var + (len(contexts) - 1) * new_var) / (self.n_samples_seen + len(contexts) - 1)
             self.feature_stds = np.sqrt(combined_var)
        else:
             self.feature_stds = current_batch_std
        self.n_samples_seen += len(contexts)

    def select_action(self, context_vector, strategy="ucb_lin"):
        # Prepare the single context vector
        norm_context = self._prepare_features(context_vector) # Shape: (n_features,)

        if strategy == "ucb_lin":
            # Calculate UCB for all actions
            ucb_values = {}
            for a in self.actions:
                 # Calculate mean prediction: theta_a.T @ context
                 mean_pred = self.theta[a].T @ norm_context # Scalar result
                 # Calculate uncertainty (confidence bound): alpha * sqrt(context.T @ A_inv_a @ context)
                 uncertainty = self.alpha * np.sqrt(norm_context.T @ self.A_inv[a] @ norm_context) # Scalar result
                 ucb_values[a] = mean_pred + uncertainty
            # Select action with highest UCB
            chosen_action = max(ucb_values, key=ucb_values.get)
            return chosen_action
        else: # Default to greedy
            # Calculate greedy value for all actions
            greedy_values = {}
            for a in self.actions:
                 # Calculate mean prediction: theta_a.T @ context
                 mean_pred = self.theta[a].T @ norm_context # Scalar result
                 greedy_values[a] = mean_pred
            # Select action with highest mean prediction
            chosen_action = max(greedy_values, key=greedy_values.get)
            return chosen_action

    def update(self, context_vector, chosen_action, reward):
        # Prepare the context vector
        norm_context = self._prepare_features(context_vector) # Shape: (n_features,)
        if norm_context.ndim == 1:
            norm_context = norm_context.reshape(-1, 1) # Shape: (n_features, 1)
        else:
            norm_context = norm_context.T # Ensure it's column vector (n_features, 1)

        # Ensure norm_context is 1D for the outer product calculation if needed
        norm_context_1d = norm_context.flatten() # Shape: (n_features,)

        # --- LinUCB Update Rule ---
        # A[action] += context * context^T
        self.A[chosen_action] += np.outer(norm_context_1d, norm_context_1d) # Uses 1D arrays
        # b[action] += reward * context
        self.b[chosen_action] += reward * norm_context_1d # Uses 1D array

        # Update inverse covariance A_inv[action] = (A[action])^-1
        # In practice, you might use Sherman-Morrison formula for efficiency
        # self.A_inv[chosen_action] = np.linalg.inv(self.A[chosen_action])
        # Using Sherman-Morrison for efficiency (recommended for online updates):
        A_inv_old = self.A_inv[chosen_action] # Shape: (d, d)
        # Calculate: A_inv_new = A_inv_old - (A_inv_old @ outer(ctx, ctx) @ A_inv_old) / (1 + ctx.T @ A_inv_old @ ctx)
        numerator_term = A_inv_old @ np.outer(norm_context_1d, norm_context_1d) @ A_inv_old # Shape: (d, d)
        denominator_term = 1.0 + norm_context_1d.T @ A_inv_old @ norm_context_1d # Scalar
        self.A_inv[chosen_action] = A_inv_old - numerator_term / denominator_term

        # Update estimated weights theta[action] = A_inv[action] @ b[action]
        # Ensure b is treated as a column vector when needed for matmul
        self.theta[chosen_action] = self.A_inv[chosen_action] @ self.b[chosen_action] # Result is 1D array

        # Update feature stats
        self.update_feature_stats(norm_context_1d.reshape(1, -1)) # Pass as (1, n_features) array

        # Update action count
        self.action_counts[chosen_action] += 1

    def save_bandit(self, filepath):
        """Save the bandit's state."""
        state = {
            'A': self.A,
            'b': self.b,
            'A_inv': self.A_inv,
            'theta': self.theta,
            'alpha': self.alpha,
            'actions': self.actions,
            'n_features': self.n_features,
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
            'n_samples_seen': self.n_samples_seen,
            'action_counts': self.action_counts
        }
        joblib.dump(state, filepath)

    def load_bandit(self, filepath):
        """Load the bandit's state."""
        state = joblib.load(filepath)
        self.A = state['A']
        self.b = state['b']
        self.A_inv = state['A_inv']
        self.theta = state['theta']
        self.alpha = state['alpha']
        self.actions = state['actions']
        self.n_features = state['n_features']
        self.feature_means = state['feature_means']
        self.feature_stds = state['feature_stds']
        self.n_samples_seen = state['n_samples_seen']
        self.action_counts = state['action_counts']

# --- End of Bandit Class ---

def engineer(df):
    df = df.copy()
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df['co2_pm25_ratio'] = df['CO2'] / (df['PMS2_5'] + 1e-6)
    df['temp_humidity_ratio'] = df['T'] / (df['H'] + 1e-6)
    df['voc_ethanol_ratio'] = df['VoC'] / (df['C2H5OH'] + 1e-6)
    df['co_rise_rate'] = df['CO'].diff().fillna(0)
    if len(df) > 5:
        df['pm25_co2_correlation'] = df['PMS2_5'].rolling(5).corr(df['CO2']).fillna(0)
    else:
        df['pm25_co2_correlation'] = 0.0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df

def build_corr_knn_graph(X, k=8):
    n = X.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=np.float32)
    # Normalize rows for cosine similarity
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    Xn = X / norms
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n), metric='cosine').fit(X)
    _, inds = nbrs.kneighbors(X)
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in inds[i]:
            if j == i:
                continue
            A[i, j] = 1.0
            A[j, i] = 1.0
    for i in range(n - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    return A

def generate_reason(row, label):

    if label == 'normal':
        return 'no rule triggered; baseline normal'

    reasons = []

    # Fire
    if label == 'fire':
        if row.get('CO', 0) > 100:
            reasons.append('CO levels are high')
        if row.get('PMS2_5', 0) > 150 or row.get('PMS10', 0) > 200:
            reasons.append('PM levels are high')

    # Smoke
    elif label == 'smoke':
        if row.get('PMS2_5', 0) > 100:
            reasons.append('PM2.5 is above threshold')
        if row.get('VoC', 0) > 300:
            reasons.append('VOC levels elevated')

    # Cooking Spill
    elif label == 'cooking_spill':
        if row.get('co2_rise_1to2min', 0) > 150:
            reasons.append('CO2 rising rapidly')
        if row.get('PMS2_5', 0) > 80:
            reasons.append('PM2.5 detected from cooking activity')

    # Possible gas/chemical spill
    elif label == 'possible_gas_or_chemical_spill':
        if row.get('C2H5OH', 0) > 200:
            reasons.append('Ethanol sensor spike')
        if row.get('VoC', 0) > 300:
            reasons.append('VOC levels abnormal')

    # Dishwashing
    elif label == 'dishwashing':
        if row.get('VoC', 0) > 400:
            reasons.append('High VOC concentration')
        if row.get('CO2', 0) > 1000:
            reasons.append('CO2 levels elevated')

    # If no specific rule fired:
    if not reasons:
        return 'no rule triggered; baseline normal'

    return '; '.join(reasons)

def infer(args):
    print("Loading model and preprocessing...")
    model = torch.jit.load(args.model, map_location='cpu')
    preproc = joblib.load(args.preproc)
    feature_list = preproc['feature_list']
    scaler = preproc['scaler']
    label_encoder = preproc['label_encoder']
    reason_rules = preproc.get('reason_rules', {})

    print(f"✓ Model loaded")
    print(f"✓ Preprocessing loaded ({len(feature_list)} features)")
    print(f"✓ Classes: {label_encoder.classes_}")
    print()

    df = args.input
    print(df.columns)

    # Convert the 'ts' column to datetime
    df['ts'] = pd.to_datetime(df['ts'])
    # Step 2: Sort by the 'ts' column
    df = df.sort_values("ts").reset_index(drop=True)

    total_samples = len(df)
    print(f"✓ Loaded {total_samples} samples for inference")

    # Check if we need to process in batches
    max_samples = getattr(args, 'max_samples', 10000)
    if total_samples > max_samples:
        print(f"⚠ Large dataset detected ({total_samples} samples)")
        print(f"⚠ Processing first {max_samples} samples to avoid memory issues")
        print(f"⚠ To process all data, run inference on smaller chunks")
        df = df.head(max_samples)
        print(f"✓ Using {len(df)} samples")
    print()

    print("Engineering features...")
    df_proc = engineer(df)
    X = df_proc[feature_list].values.astype(np.float32)
    X = np.where(np.isinf(X), np.nan, X)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    print(f"✓ Features engineered")

    print("Scaling features...")
    Xs = scaler.transform(X)
    print(f"✓ Features scaled")

    print("Building graph structure...")
    adj = build_corr_knn_graph(Xs, k=args.k_neighbors)
    # fallback minimal connectivity if degenerate
    if adj.sum() == 0:
        n = Xs.shape[0]
        adj = np.zeros((n, n), dtype=np.float32)
        for i in range(max(1, n-1)):
            adj[i, i+1] = 1.0
            adj[i+1, i] = 1.0

    print(f"✓ Graph built")
    print()

    # to tensors
    x_t = torch.FloatTensor(Xs)
    adj_t = torch.FloatTensor(adj)

    print("Running inference...")
    model.eval()
    with torch.no_grad():
        logits, conf, alphas = model(x_t, adj_t)  # traced wrapper returns logits, conf, alpha
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds_idx = np.argmax(probs, axis=1)
        conf_np = conf.cpu().numpy()
        alpha_np = alphas.cpu().numpy()
    print(f"✓ Predictions completed")
    print()

    # Map predicted class indices to label strings via the loaded LabelEncoder
    try:
        labels = label_encoder.inverse_transform(preds_idx)
    except Exception:
        # fallback: if label_encoder is not a sklearn object, use classes list if present
        classes = preproc.get('classes', None)
        if classes:
            labels = [classes[i] for i in preds_idx]
        else:
            labels = [str(int(i)) for i in preds_idx]

    # Generate reasons using df_proc (engineered features)
    reasons = []
    for i in range(len(df_proc)):
        row_dict = df_proc.iloc[i].to_dict()
        label = labels[i]
        reason = generate_reason(row_dict, label)
        reasons.append(reason)

    # Build output DataFrame
    df_out = df.copy().reset_index(drop=True)
    df_out['Label'] = labels
    # Compute confidence: mix model confidence and softmax max prob
    prob_max = probs.max(axis=1)
    combined_conf = 0.6 * prob_max + 0.4 * (conf_np.clip(0, 1))
    combined_conf = np.clip(combined_conf, 0.0, 1.0)
    df_out['Confidence'] = combined_conf
    df_out['Reason'] = reasons

    # --- NEW: Contextual Bandit Integration ---
    print("Initializing Contextual Bandit...")
    # Define the features for the bandit context
    # Example: Use scaled features + confidence + label encoding
    bandit_feature_cols = feature_list + ['Confidence']
    # Encode the label (e.g., LabelEncoder) or use one-hot
    label_encoded = label_encoder.transform(labels) # Get numeric codes
    # Add label code as a feature
    df_out['Label_Code'] = label_encoded
    bandit_feature_cols.append('Label_Code') # Add the numeric label code

    to_remove = [
        'co2_pm25_ratio',
        'temp_humidity_ratio',
        'voc_ethanol_ratio',
        'co_rise_rate',
        'pm25_co2_correlation'
    ]

    # Remove only if they exist
    bandit_feature_cols = [c for c in bandit_feature_cols if c not in to_remove]

    # Ensure all bandit features are numeric and handle potential NaN if new columns were added
    context_matrix = df_out[bandit_feature_cols].fillna(0.0).values.astype(np.float32)

    n_features = context_matrix.shape[1]

    # Initialize the bandit (or load a pre-trained one if available)
    bandit_model_path = os.path.join(os.path.dirname(args.model), 'bandit_model.joblib') # Assume it's saved in the same dir
    if os.path.exists(bandit_model_path):
        print(f"Loading existing bandit model from {bandit_model_path}")
        bandit = LinearContextualBandit(n_features=0, actions=ACTIONS) # Initialize with dummy n_features
        bandit.load_bandit(bandit_model_path)
        # Ensure the loaded model's n_features matches
        if bandit.n_features != n_features:
             print(f"Warning: Loaded bandit n_features ({bandit.n_features}) does not match calculated n_features ({n_features}). Re-initializing bandit.")
             bandit = LinearContextualBandit(n_features=n_features, actions=ACTIONS, alpha=args.bandit_alpha)
        else:
             print("Loaded bandit successfully.")
    else:
        print(f"No existing bandit model found at {bandit_model_path}. Initializing new bandit.")
        bandit = LinearContextualBandit(n_features=n_features, actions=ACTIONS, alpha=args.bandit_alpha)


    # --- Simulate or Define Reward Function ---
    # In a real system, you would define a reward based on the outcome of the action.
    # For simulation, you might define a reward based on the anomaly type and a hypothetical "correct" action.
    # Example: Reward is high if the action aligns with the anomaly type (e.g., "open_window" for "smoke")
    # Or reward is based on a subsequent improvement in sensor readings (requires future data).
    # For now, let's define a simple hypothetical reward function based on anomaly type.
    # This function needs to be defined *before* the loop.
    def calculate_reward(anomaly_label, chosen_action, confidence):
        # Define hypothetical rewards based on anomaly type
        # Higher confidence in anomaly detection might lead to higher potential reward for correct action
        base_reward = 1.0
        if anomaly_label == 'smoke' or anomaly_label == 'fire':
            if chosen_action in ['open_window', 'turn_on_fan']:
                return base_reward * (0.8 + 0.2 * confidence) # Higher reward with higher confidence
            elif chosen_action == 'do_nothing':
                return -base_reward * (1.0 + 0.1 * confidence) # Penalty for inaction
            else:
                return 0.0 # Neutral or small penalty
        elif anomaly_label == 'possible_gas_or_chemical_spill':
            if chosen_action in ['open_window', 'turn_on_fan', 'alert_user']:
                return base_reward * (0.7 + 0.3 * confidence)
            elif chosen_action == 'do_nothing':
                return -base_reward * (1.0 + 0.1 * confidence)
            else:
                return 0.0
        elif anomaly_label == 'cooking_spill':
            if chosen_action in ['turn_on_fan', 'reduce_stove_load']:
                return base_reward * (0.6 + 0.4 * confidence)
            elif chosen_action == 'do_nothing':
                return -base_reward * (0.8 + 0.2 * confidence)
            else:
                return 0.0
        else: # 'normal' or other
            if chosen_action == 'do_nothing':
                return base_reward * 0.5 # Small positive reward for correct inaction
            else:
                return -base_reward * 0.2 # Small penalty for unnecessary action

        # Default neutral reward if no specific rule matches
        return 0.0


    # --- Loop to Select Action for Each Row ---
    selected_actions = []
    rewards_received = [] # Store rewards for potential future learning/evaluation
    for i in range(len(df_out)):
        context_vector = context_matrix[i]

        # Select action using the bandit
        chosen_action = bandit.select_action(context_vector, strategy="ucb_lin") # Use UCB strategy
        selected_actions.append(chosen_action)

        # --- CRITICAL: Define Reward based on outcome (simulated here) ---
        # In a real system, you would observe the outcome *after* taking the action.
        # Here, we simulate a reward based on the detected anomaly and the action taken.
        anomaly_type = df_out.iloc[i]['Label']
        confidence = df_out.iloc[i]['Confidence']
        reward = calculate_reward(anomaly_type, chosen_action, confidence)
        rewards_received.append(reward)

        # Update the bandit with the context, action, and reward
        # This step is crucial for the bandit to learn the relationship between context, action, and reward
        bandit.update(context_vector, chosen_action, reward)

    # Add selected actions and rewards to the output DataFrame
    df_out['Selected_Action'] = selected_actions
    df_out['Simulated_Reward'] = rewards_received

    # --- Save the updated bandit model ---
    print(f"Saving updated bandit model to {bandit_model_path}")
    bandit.save_bandit(bandit_model_path)

    # --- Existing Code Continuation ---
    counts = df_out['Label'].value_counts().sort_values(ascending=False)
    print(f"\nGenerating anomalies and actions")
    for label, count in counts.items():
        pct = 100 * count / len(df_out)
        print(f"  {label:30s}: {count:6d} samples ({pct:5.1f}%)")

    action_counts = df_out['Selected_Action'].value_counts().sort_values(ascending=False)
    print(f"\nSelected Action distribution:")
    for action, count in action_counts.items():
        pct = 100 * count / len(df_out)
        print(f"  {action:30s}: {count:6d} samples ({pct:5.1f}%)")

    # Return the DataFrame with actions
    return df_out

# if __name__ == "__main__":
def anomaly_detector(df):
    def get_file_path(filename):
        path = os.path.join(os.path.dirname(__file__), '..', 'anomaly_detector', 'out_train', filename)
        if not os.path.exists(path):
            raise HTTPException(
                status_code=500,
                detail=f"Sample data file not found at {path}"
            )
        return path

    args = argparse.Namespace(
        model=get_file_path('model.pt'),
        preproc= get_file_path('preproc.joblib'),
        input= df,
        k_neighbors=8,
        output_csv='anomaly_predictions2.csv',
        max_samples=10000,
        bandit_alpha=0.1 # Add alpha parameter for the bandit's exploration
    )

    anomaly_df = infer(args)
    return anomaly_df

generate_reason()

# Example usage (if run as main script)
if __name__ == "__main__":
    # Load your input data
    df = pd.read_csv("unlabeled_sample.csv")
    result_df = anomaly_detector(df)
    print(result_df[['ts', 'Label', 'Confidence', 'Selected_Action', 'Simulated_Reward']])
    result_df.to_csv("out_infer/anomaly_predictions2.csv", index=False)
    pass
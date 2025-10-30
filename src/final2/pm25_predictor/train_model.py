# train_model_optimized.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
import os
import pickle
from tqdm import tqdm
import time

warnings.filterwarnings('ignore')

# Set device with enhanced GPU detection and configuration
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"   CUDA Version: {torch.version.cuda}")
    # Enable cuDNN autotuner for better performance
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
    print("⚠️  No GPU detected - using CPU (training will be slower)")
    print("   To use GPU, install CUDA-enabled PyTorch:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

print(f"Using device: {device}")


# --- Optimized Data Loading ---
def load_and_preprocess_dataset_optimized():
    """
    Optimized data loading with chunk processing for large files
    """
    # Use relative path from current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    input_file = os.path.join(project_root, 'data', 'raw', '2025_raw.csv')
    processed_file = os.path.join(os.path.dirname(__file__), '2025_optimized.csv')

    # Check if processed file exists
    if os.path.exists(processed_file):
        print(f"Loading preprocessed data from {processed_file}")
        return pd.read_csv(processed_file)

    print("Loading and processing dataset...")

    # Load only necessary columns to reduce memory
    usecols = ['pm2_5', 'latitude', 'longitude', 'datetime']
    # Load ALL data (removed nrows limit)
    df = pd.read_csv(input_file, usecols=usecols)

    print(f"Original dataset shape: {df.shape}")

    # Efficient filtering
    mask = (df['pm2_5'].notna() &
            df['latitude'].notna() &
            df['longitude'].notna() &
            df['datetime'].notna() &
            (df['pm2_5'] > 0) & (df['pm2_5'] != -999))

    df = df[mask].copy()

    # Convert datetime efficiently
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year

    # Save processed data
    df.to_csv(processed_file, index=False)
    print(f"Processed dataset saved to {processed_file}")
    print(f"Final dataset shape: {df.shape}")

    return df


# --- Optimized KG Builder ---
class OptimizedKGBuilder:
    def __init__(self):
        self.entity_to_id = {}
        self.edge_index = None
        self.node_features = None

    def build_kg_fast(self, df):
        """Optimized KG building without data leakage"""
        print("Building optimized knowledge graph...")

        # Use efficient data structures
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Create location and time entities efficiently
        locations = [f"loc_{lat:.4f}_{lon:.4f}"
                     for lat, lon in zip(df['latitude'], df['longitude'])]
        hours = df['datetime'].dt.hour
        time_periods = [f"time_{hour:02d}" for hour in hours]

        # Create unique entities
        unique_locations = list(set(locations))
        unique_times = list(set(time_periods))
        all_entities = unique_locations + unique_times

        self.entity_to_id = {entity: idx for idx, entity in enumerate(all_entities)}

        # Pre-allocate node features
        node_features = []
        for entity in all_entities:
            if entity.startswith("loc_"):
                parts = entity.split('_')
                lat, lon = float(parts[1]), float(parts[2])
                features = [lat, lon, 1.0, 0.0]  # Location type indicator
            else:
                hour = int(entity.split('_')[1])
                features = [
                    np.sin(2 * np.pi * hour / 24),
                    np.cos(2 * np.pi * hour / 24),
                    0.0, 1.0  # Time type indicator
                ]
            node_features.append(features)

        self.node_features = torch.tensor(node_features, dtype=torch.float32)

        # Create edges efficiently using batch operations
        edges = []
        loc_to_id = {loc: self.entity_to_id[loc] for loc in unique_locations}
        time_to_id = {time: self.entity_to_id[time] for time in unique_times}

        for loc, time in zip(locations, time_periods):
            loc_id = loc_to_id[loc]
            time_id = time_to_id[time]
            edges.append([loc_id, time_id])
            edges.append([time_id, loc_id])

        if edges:
            self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            self.edge_index = torch.empty((2, 0), dtype=torch.long)

        print(f"Optimized KG built with {len(all_entities)} nodes and {len(edges)} edges")
        return self.edge_index, self.node_features

    def get_data_mapping_fast(self, df):
        """Optimized data mapping"""
        locations = [f"loc_{lat:.4f}_{lon:.4f}"
                     for lat, lon in zip(df['latitude'], df['longitude'])]
        time_periods = [f"time_{hour:02d}" for hour in df['datetime'].dt.hour]

        data_to_nodes = {}
        for idx, (loc, time) in enumerate(zip(locations, time_periods)):
            data_to_nodes[idx] = {
                'location': self.entity_to_id[loc],
                'time': self.entity_to_id[time]
            }
        return data_to_nodes


# --- Optimized GNN Model ---
class OptimizedGNNPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(OptimizedGNNPredictor, self).__init__()
        # Fewer GNN layers for faster training
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.gnn_output_dim = hidden_dim // 2

        # Simplified feature network
        self.original_feature_net = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Dropout(0.2)  # Reduced dropout
        )

        # Simplified classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.gnn_output_dim + 32, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim)
        )

    def forward(self, x, edge_index, original_features=None):
        # Optimized GNN processing
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x

    def predict_from_embeddings(self, embeddings, original_features):
        """Separate method for final prediction from embeddings"""
        orig_feat = self.original_feature_net(original_features)
        combined = torch.cat([embeddings, orig_feat], dim=1)
        return self.classifier(combined)


# --- Optimized Predictor Class ---
class OptimizedPM25Predictor:
    def __init__(self):
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        self.kg_builder = OptimizedKGBuilder()
        self.model = None
        self.device = device

    def prepare_features_optimized(self, df):
        """Optimized feature preparation"""
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Essential temporal features only
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['month'] = df['datetime'].dt.month

        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        feature_cols = [
            'latitude', 'longitude', 'hour_sin', 'hour_cos',
            'month_sin', 'month_cos', 'day_of_week', 'day_of_year'
        ]

        X = df[feature_cols].values
        y = df['pm2_5'].values
        return X, y, feature_cols

    def train_optimized(self, df, epochs=100, learning_rate=0.001, batch_size=1024):
        """Optimized training with batching and progress tracking"""
        print("Starting optimized training...")
        print(f"🖥️  Training on: {self.device}")
        
        # Adjust batch size based on device for maximum speed
        if self.device.type == 'cuda':
            # GPU can handle much larger batches for speed
            batch_size = 8192  # Increased for faster GPU training
            print(f"   Using GPU-optimized batch size: {batch_size}")
        else:
            # CPU also benefits from larger batches
            batch_size = 4096  # Increased for faster CPU training
            print(f"   Using CPU batch size: {batch_size}")
        
        start_time = time.time()

        # Ensure datetime is properly formatted
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Prepare features
        X, y, _ = self.prepare_features_optimized(df)
        X_scaled = self.scaler_features.fit_transform(X)
        y_scaled = self.scaler_target.fit_transform(y.reshape(-1, 1)).flatten()

        # Build KG
        edge_index, node_features = self.kg_builder.build_kg_fast(df)
        data_mapping = self.kg_builder.get_data_mapping_fast(df)

        # Split data
        train_val_idx, test_idx = train_test_split(range(len(df)), test_size=0.2, random_state=42)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=42)

        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        # Initialize model
        self.model = OptimizedGNNPredictor(
            input_dim=node_features.shape[1],
            hidden_dim=64,
            output_dim=1
        ).to(self.device)

        # Move data to device
        print(f"📊 Moving data to {self.device}...")
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(self.device)
        
        if self.device.type == 'cuda':
            print(f"   GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f"   GPU Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

        # Optimization with better settings
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        criterion = nn.HuberLoss()

        # Training variables with faster early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15  # Reduced from 20 for faster training
        train_losses = []
        val_losses = []
        
        # Pre-compute node embeddings less frequently for speed
        embedding_update_frequency = 5  # Update embeddings every N epochs

        print("Training with progress tracking...")
        
        # Pre-compute initial embeddings
        cached_embeddings = None
        
        for epoch in range(epochs):
            self.model.train()
            epoch_train_loss = 0
            batch_count = 0
            
            # Update embeddings only periodically for speed
            if cached_embeddings is None or epoch % embedding_update_frequency == 0:
                with torch.no_grad():
                    cached_embeddings = self.model(node_features, edge_index)

            # Batch processing for training - VECTORIZED
            for i in range(0, len(train_idx), batch_size):
                optimizer.zero_grad()

                batch_indices = train_idx[i:i + batch_size]
                if not batch_indices:
                    continue

                # Vectorized batch processing (MUCH FASTER)
                batch_mappings = [data_mapping[idx] for idx in batch_indices]
                loc_indices = torch.tensor([m['location'] for m in batch_mappings], device=self.device)
                time_indices = torch.tensor([m['time'] for m in batch_mappings], device=self.device)
                
                # Get embeddings in batch
                loc_embs = cached_embeddings[loc_indices]
                time_embs = cached_embeddings[time_indices]
                combined_embs = (loc_embs + time_embs) / 2
                
                # Get original features in batch
                original_feats = X_tensor[batch_indices]
                
                # Predict in batch (single forward pass)
                batch_preds = self.model.predict_from_embeddings(combined_embs, original_feats).squeeze()
                batch_targets = y_tensor[batch_indices]
                
                # Compute loss
                batch_loss = criterion(batch_preds, batch_targets)

                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_train_loss += batch_loss.item()
                batch_count += 1
                
                # Update embeddings after optimizer step
                if epoch % embedding_update_frequency == 0:
                    with torch.no_grad():
                        cached_embeddings = self.model(node_features, edge_index)

            avg_train_loss = epoch_train_loss / batch_count if batch_count > 0 else 0
            train_losses.append(avg_train_loss)

            # Validation - also vectorized for speed
            self.model.eval()
            with torch.no_grad():
                # Get fresh embeddings for validation
                node_embeddings = self.model(node_features, edge_index)
                
                # Process validation in larger batches
                val_batch_size = batch_size * 4
                all_val_preds = []
                all_val_targets = []
                
                for i in range(0, len(val_idx), val_batch_size):
                    batch_indices = val_idx[i:i + val_batch_size]
                    
                    # Vectorized validation
                    batch_mappings = [data_mapping[idx] for idx in batch_indices]
                    loc_indices = torch.tensor([m['location'] for m in batch_mappings], device=self.device)
                    time_indices = torch.tensor([m['time'] for m in batch_mappings], device=self.device)
                    
                    loc_embs = node_embeddings[loc_indices]
                    time_embs = node_embeddings[time_indices]
                    combined_embs = (loc_embs + time_embs) / 2
                    
                    original_feats = X_tensor[batch_indices]
                    preds = self.model.predict_from_embeddings(combined_embs, original_feats).squeeze()
                    targets = y_tensor[batch_indices]
                    
                    all_val_preds.append(preds)
                    all_val_targets.append(targets)
                
                if all_val_preds:
                    val_preds = torch.cat(all_val_preds)
                    val_targets = torch.cat(all_val_targets)
                    val_loss = criterion(val_preds, val_targets).item()
                else:
                    val_loss = float('inf')

                val_losses.append(val_loss)

            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            # Progress reporting - less frequent for speed
            if epoch % 5 == 0 or epoch < 10:  # Report every 5 epochs (or first 10)
                elapsed = time.time() - start_time
                epoch_time = elapsed / (epoch + 1)
                remaining = epoch_time * (epochs - epoch - 1)
                gpu_mem = f", GPU Mem: {torch.cuda.memory_allocated(0) / 1024**2:.0f}MB" if self.device.type == 'cuda' else ""
                print(f'Epoch {epoch:3d}/{epochs}: Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}'
                      f'{gpu_mem} | ⏱️ {elapsed/60:.1f}m elapsed, ~{remaining/60:.1f}m remaining')

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        if 'best_model_state' in locals():
            self.model.load_state_dict(best_model_state)

        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
        
        # Clear GPU cache if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            print(f"   GPU Memory freed")

        # Final evaluation
        print("📈 Evaluating on test set...")
        test_metrics = self.evaluate_optimized(test_idx, data_mapping, node_features,
                                               edge_index, X_tensor, y_tensor, batch_size)

        return test_metrics, (train_losses, val_losses)

    def evaluate_optimized(self, indices, data_mapping, node_features, edge_index,
                           X_tensor, y_tensor, batch_size=2048):
        """Optimized evaluation with vectorized batching"""
        self.model.eval()
        with torch.no_grad():
            node_embeddings = self.model(node_features, edge_index)
            
            all_preds = []
            all_targets = []
            
            # Vectorized evaluation with large batches
            eval_batch_size = batch_size * 4

            # Process in large batches
            for i in range(0, len(indices), eval_batch_size):
                batch_indices = indices[i:i + eval_batch_size]
                
                # Vectorized batch processing
                batch_mappings = [data_mapping[idx] for idx in batch_indices]
                loc_indices = torch.tensor([m['location'] for m in batch_mappings], device=self.device)
                time_indices = torch.tensor([m['time'] for m in batch_mappings], device=self.device)
                
                loc_embs = node_embeddings[loc_indices]
                time_embs = node_embeddings[time_indices]
                combined_embs = (loc_embs + time_embs) / 2
                
                original_feats = X_tensor[batch_indices]
                preds = self.model.predict_from_embeddings(combined_embs, original_feats).squeeze()
                targets = y_tensor[batch_indices]
                
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

            if not all_preds:
                return {'MAE': 0.0, 'R2_Score': 0.0}

            preds_scaled = torch.cat(all_preds).numpy()
            targets_scaled = torch.cat(all_targets).numpy()

            # Inverse transform
            preds_original = self.scaler_target.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
            targets_original = self.scaler_target.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()

            metrics = {
                'MAE': mean_absolute_error(targets_original, preds_original),
                'R2_Score': r2_score(targets_original, preds_original)
            }

            return metrics

    def save_model(self, filepath_prefix):
        """Save the trained model"""
        if self.model is None:
            print("No model to save!")
            return

        # Move model to CPU before saving for compatibility
        original_device = next(self.model.parameters()).device
        self.model.to('cpu')
        self.device = torch.device('cpu')
        
        # Also move tensors to CPU
        if hasattr(self.kg_builder, 'node_features') and self.kg_builder.node_features is not None:
            self.kg_builder.node_features = self.kg_builder.node_features.cpu()
        if hasattr(self.kg_builder, 'edge_index') and self.kg_builder.edge_index is not None:
            self.kg_builder.edge_index = self.kg_builder.edge_index.cpu()

        try:
            with open(f"{filepath_prefix}.pkl", 'wb') as f:
                pickle.dump(self, f)
            print(f"💾 Model saved successfully to {filepath_prefix}.pkl")
            print(f"   Model can be loaded on both CPU and GPU")
        except Exception as e:
            print(f"⚠️  Warning: Could not save full predictor: {e}")
            print("Saving model state dict instead...")
            torch.save(self.model.state_dict(), f"{filepath_prefix}.pth")


# --- Main Execution ---
def main_optimized():
    try:
        print("--- Starting Optimized Pipeline ---")

        # Load data
        processed_df = load_and_preprocess_dataset_optimized()
        print(f"Data shape: {processed_df.shape}")
        print(f"Using ALL {len(processed_df)} records for training!")

        # Train model with all data
        predictor = OptimizedPM25Predictor()
        metrics, losses = predictor.train_optimized(
            processed_df,
            epochs=100,  # Reduced epochs with faster convergence
            learning_rate=0.002,  # Slightly higher LR for faster training
            batch_size=4096  # Will be auto-adjusted based on device
        )

        # Save model
        predictor.save_model("pm25_gnn_complete_complete_model")

        # Display results
        print("\nFinal Test Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # Plot training history
        plt.figure(figsize=(10, 5))
        train_losses, val_losses = losses
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main_optimized()
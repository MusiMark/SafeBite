
import os
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
import joblib
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------- Config ----------
FEATURE_COLS = [
    'T', 'H', 'PMS1', 'PMS2_5', 'PMS10', 'CO2', 'NO2', 'CO', 'VoC', 'C2H5OH',
    'pm25_max_2min', 'pm25_mean_2min', 'co2_max_2min', 'co2_mean_1min',
    'co2_rise_1to2min', 'T_rise_1min', 'pm10_mean_30s', 'pm10_spike',
    'H_mean_2min', 'voc_mean_1min', 'ethanol_mean_1min'
]
DERIVED = ['co2_pm25_ratio', 'temp_humidity_ratio', 'voc_ethanol_ratio', 'co_rise_rate', 'pm25_co2_correlation']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- Data utilities ----------
def load_data(path, nrows=None):
    df = pd.read_csv(path, parse_dates=['ts'], nrows=nrows)
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df.sort_values('ts').reset_index(drop=True)
    df = df.dropna(subset=FEATURE_COLS)
    return df

def engineer(df):
    df = df.copy()
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
    """
    Build graph adjacency (sparse) from correlation / similarity between row feature vectors.
    Returns adjacency matrix A (n x n) as sparse CSR matrix to save memory.
    We compute cosine similarity then select k nearest neighbors (excluding self).
    """
    n = X.shape[0]
    print(f"  Building graph for {n:,} nodes with k={k}...")
    
    if n <= 1:
        return np.zeros((n, n), dtype=np.float32)
    
    # Build kNN using sklearn (memory efficient)
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n), metric='cosine', n_jobs=-1)
    nbrs.fit(X)
    _, inds = nbrs.kneighbors(X)
    
    # Build sparse adjacency matrix
    rows = []
    cols = []
    for i in range(n):
        for j in inds[i]:
            if j == i:
                continue
            rows.append(i)
            cols.append(j)
    
    # Connect consecutive timestamps to encode temporality
    for i in range(n - 1):
        rows.extend([i, i + 1])
        cols.extend([i + 1, i])
    
    # Create sparse matrix
    data = np.ones(len(rows), dtype=np.float32)
    A_sparse = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    
    # Convert to dense (only if small enough)
    if n < 20000:  # Safe threshold for dense conversion
        return A_sparse.toarray()
    else:
        # For large datasets, we'll need to work with batches
        # Return None to signal we need batch processing
        return None

# ---------- Model: Pure PyTorch Graph Attention (traceable) ----------
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, concat=True):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(out_dim, 1, bias=False)  # produces scalar per node embedding
        self.leaky = nn.LeakyReLU(0.2)
        self.concat = concat

    def forward(self, h, adj):
        # h: (n, in_dim), adj: (n, n) dense mask (0/1)
        Wh = self.W(h)  # (n, out_dim)
        # compute attention potential scalar for each node: f_i = a(Wh_i)
        f = self.a(Wh).squeeze(-1)  # (n,)
        # compute e_ij = LeakyReLU(f_i + f_j)
        e = self.leaky(f.unsqueeze(1) + f.unsqueeze(0))  # (n, n)
        # mask with adjacency
        mask = (adj > 0).to(Wh.dtype)  # (n, n)
        # set e where mask==0 to large negative to zero out after softmax
        NEG_INF = -1e9
        e_masked = e * mask + (1.0 - mask) * NEG_INF
        # normalize
        alpha = torch.softmax(e_masked, dim=1)  # attention weights over neighbors for each i
        # aggregate: new_h_i = sum_j alpha_ij * Wh_j
        h_prime = alpha @ Wh  # (n, out_dim)
        return F.elu(h_prime), alpha  # return alpha for explainability

class TemporalGAT(nn.Module):
    def __init__(self, in_dim, hid_dim=64, out_dim=64, n_classes=2, n_heads=2):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hid_dim)
        # multi-head attention implemented by stacking heads
        self.gat_heads = nn.ModuleList([GraphAttentionLayer(hid_dim, out_dim) for _ in range(n_heads)])
        self.final_proj = nn.Linear(out_dim * n_heads, out_dim)
        # classifier & confidence
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Linear(out_dim // 2, n_classes)
        )
        self.confidence = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Linear(out_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, adj):
        # x: (n, f), adj: (n, n) dense
        h0 = F.relu(self.input_proj(x))
        head_outs = []
        head_alphas = []
        for head in self.gat_heads:
            out, alpha = head(h0, adj)
            head_outs.append(out)
            head_alphas.append(alpha.unsqueeze(0))  # (1, n, n)
        # concatenate heads on feature dim
        h_cat = torch.cat(head_outs, dim=1)  # (n, out_dim * n_heads)
        h_final = F.relu(self.final_proj(h_cat))
        logits = self.classifier(h_final)  # (n, n_classes)
        conf = self.confidence(h_final)    # (n, 1)
        # combine attention for explainability: average heads
        alphas = torch.cat(head_alphas, dim=0)  # (n_heads, n, n)
        mean_alpha = torch.mean(alphas, dim=0)  # (n, n)
        return logits, conf.squeeze(-1), mean_alpha  # logits, confidence, attention matrix

# ---------- Training ----------
def train(args):
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("ANOMALY DETECTION TRAINING - TEMPORAL GAT MODEL")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Dataset: {args.input}")
    print(f"Output directory: {args.out_dir}")
    print()

    print("Loading dataset...")
    df = load_data(args.input, nrows=args.nrows)
    print(f"✓ Loaded {len(df)} samples")
    
    df = engineer(df)
    features = [f for f in FEATURE_COLS + DERIVED if f in df.columns]
    X = df[features].values.astype(np.float32)
    # sanitize
    X = np.where(np.isinf(X), np.nan, X)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # labels
    if 'Label' not in df.columns:
        raise ValueError("Training requires 'Label' column in input CSV")
    labels_raw = df['Label'].astype(str).fillna('normal').values
    le = LabelEncoder()
    y = le.fit_transform(labels_raw)
    
    print(f"✓ Using {len(features)} features")
    print(f"✓ Found {len(le.classes_)} classes: {list(le.classes_)}")
    
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for cls_idx, count in zip(unique, counts):
        cls_name = le.classes_[cls_idx]
        pct = 100 * count / len(y)
        print(f"  {cls_name:20s}: {count:6d} samples ({pct:5.1f}%)")
    print()
    
    # scaler
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)
    
    # Train-validation split
    print("Creating train/validation split (80/20)...")
    train_idx, val_idx = train_test_split(
        np.arange(len(Xs)), 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    X_train, X_val = Xs[train_idx], Xs[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"✓ Train set: {len(X_train)} samples")
    print(f"✓ Val set:   {len(X_val)} samples")
    print()

    # adjacency from correlations
    print("Building graph structure (k-NN correlation graph)...")
    A_train = build_corr_knn_graph(X_train, k=args.k_neighbors)
    A_val = build_corr_knn_graph(X_val, k=args.k_neighbors)
    
    if A_train is None or A_val is None:
        raise MemoryError(
            f"Dataset too large ({len(X_train):,} train samples) for dense graph representation.\n"
            f"Please reduce nrows parameter to max 20,000 samples.\n"
            f"Current setting: nrows={args.nrows if args.nrows else 'unlimited'}"
        )
    
    print(f"✓ Graph built with k={args.k_neighbors} neighbors")
    print()

    # torch tensors
    x_train_t = torch.FloatTensor(X_train).to(DEVICE)
    adj_train_t = torch.FloatTensor(A_train).to(DEVICE)
    y_train_t = torch.LongTensor(y_train).to(DEVICE)
    
    x_val_t = torch.FloatTensor(X_val).to(DEVICE)
    adj_val_t = torch.FloatTensor(A_val).to(DEVICE)
    y_val_t = torch.LongTensor(y_val).to(DEVICE)

    n_classes = len(le.classes_)
    model = TemporalGAT(in_dim=Xs.shape[1], hid_dim=args.hidden, out_dim=args.latent, n_classes=n_classes, n_heads=args.n_heads).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    best_epoch = 0
    
    print("=" * 70)
    print("TRAINING START")
    print("=" * 70)
    print(f"Model: TemporalGAT")
    print(f"Hidden dim: {args.hidden}, Latent dim: {args.latent}, Heads: {args.n_heads}")
    print(f"Learning rate: {args.lr}, Weight decay: {args.weight_decay}")
    print(f"Epochs: {args.epochs}")
    print("=" * 70)
    print()
    
    for epoch in tqdm(range(1, args.epochs + 1), desc="Training Progress", ncols=100):
        # Training phase
        model.train()
        optimizer.zero_grad()
        logits, conf, _ = model(x_train_t, adj_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Calculate training accuracy
        with torch.no_grad():
            train_preds = torch.argmax(logits, dim=1)
            train_acc = (train_preds == y_train_t).float().mean().item()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_logits, val_conf, _ = model(x_val_t, adj_val_t)
            val_loss = criterion(val_logits, y_val_t)
            val_preds = torch.argmax(val_logits, dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()
        
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(args.out_dir, 'best_model_checkpoint.pth'))
        
        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            tqdm.write(f"Epoch {epoch:3d}/{args.epochs} | "
                      f"Train Loss: {loss.item():.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss.item():.4f} Acc: {val_acc:.4f} | "
                      f"Best: {best_val_acc:.4f}")
    
    print()
    print("=" * 70)
    print(f"✓ Training complete!")
    print(f"✓ Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print("=" * 70)
    print()

    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(args.out_dir, 'best_model_checkpoint.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded best model from epoch {checkpoint['epoch']}")
    print()

    # Evaluate on validation set
    print("Evaluating on validation set...")
    model.eval()
    with torch.no_grad():
        logits, conf, alphas = model(x_val_t, adj_val_t)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        conf_np = conf.cpu().numpy()
        y_true = y_val_t.cpu().numpy()

    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, average='weighted', zero_division=0)
    rec = recall_score(y_true, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_true, preds, average='weighted', zero_division=0)
    print("Validation Set Classification Metrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print()

    # Save classification report & confusion matrix
    report = classification_report(y_true, preds, target_names=le.classes_, zero_division=0)
    print("Detailed Classification Report:")
    print(report)
    with open(os.path.join(args.out_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    print(f"✓ Saved classification report")

    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix (Validation Set)', fontsize=14, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, le.classes_, rotation=45, ha='right')
    plt.yticks(tick_marks, le.classes_)
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'confusion_matrix.png'), dpi=200)
    plt.close()
    print(f"✓ Saved confusion matrix")

    # Plot training curves
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', alpha=0.8, linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', alpha=0.8, linewidth=2)
    plt.title('Training and Validation Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', alpha=0.8, linewidth=2)
    plt.plot(history['val_acc'], label='Val Accuracy', alpha=0.8, linewidth=2)
    plt.axhline(y=best_val_acc, color='r', linestyle='--', alpha=0.5, label=f'Best Val Acc: {best_val_acc:.4f}')
    plt.title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'training_curves.png'), dpi=200)
    plt.close()
    print(f"✓ Saved training curves")

    # Save preproc artifacts
    preproc = {
        'feature_list': features,
        'scaler': scaler,
        'label_encoder': le,
        # simple reason rule thresholds (example defaults) - you can edit this JSON later
        'reason_rules': {
            'fire': {'CO': 100.0, 'PMS2_5': 150.0},
            'smoke': {'PMS2_5': 100.0, 'VoC': 300.0},
            'cooking_spill': {'co2_rise_1to2min': 150.0},
            'possible_gas_or_chemical_spill': {'C2H5OH': 200.0},
            'dishwashing': {'VoC': 400.0, 'CO2': 800.0}
        }
    }
    joblib.dump(preproc, os.path.join(args.out_dir, 'preproc.joblib'))
    print("Preprocessing artifacts saved.")

    # Build a traceable inference wrapper and trace with TorchScript
    class InferenceWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x, adj):
            # returns logits, confidence, attention
            logits, conf, alpha = self.model(x, adj)
            return logits, conf, alpha

    wrapper = InferenceWrapper(model.cpu())
    # create example inputs
    n_ex = max(2, x_val_t.size(0))
    x_ex = torch.zeros((n_ex, x_val_t.size(1)), dtype=torch.float32)
    adj_ex = torch.zeros((n_ex, n_ex), dtype=torch.float32)
    for i in range(n_ex - 1):
        adj_ex[i, i + 1] = 1.0
        adj_ex[i + 1, i] = 1.0

    print("\nTracing model to TorchScript (model.pt)...")
    try:
        traced = torch.jit.trace(wrapper, (x_ex, adj_ex), check_trace=True)
        traced.save(os.path.join(args.out_dir, 'model.pt'))
        print("✓ Traced model saved to model.pt")
    except Exception as e:
        # If tracing fails, save state dicts as fallback
        print("⚠ TorchScript tracing failed:", e)
        torch.save({'model_state_dict': model.state_dict()}, os.path.join(args.out_dir, 'model_state_dict.pth'))
        print("✓ Saved state dicts (requires class definitions to reload).")

    # Save training summary
    summary = {
        'timestamp': datetime.utcnow().isoformat(),
        'n_samples': int(Xs.shape[0]),
        'n_train': int(len(X_train)),
        'n_val': int(len(X_val)),
        'n_features': len(features),
        'classes': list(le.classes_),
        'hyperparameters': {
            'epochs': args.epochs,
            'lr': args.lr,
            'hidden_dim': args.hidden,
            'latent_dim': args.latent,
            'n_heads': args.n_heads,
            'k_neighbors': args.k_neighbors,
            'weight_decay': args.weight_decay
        },
        'best_epoch': int(best_epoch),
        'metrics': {
            'accuracy': float(acc), 
            'precision': float(prec), 
            'recall': float(rec), 
            'f1': float(f1),
            'best_val_acc': float(best_val_acc)
        }
    }
    with open(os.path.join(args.out_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("✓ Saved training summary")

    print()
    print("=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"All artifacts saved to: {args.out_dir}")
    print(f"  - best_model_checkpoint.pth")
    print(f"  - model.pt (TorchScript)")
    print(f"  - preproc.joblib")
    print(f"  - confusion_matrix.png")
    print(f"  - training_curves.png")
    print(f"  - classification_report.txt")
    print(f"  - training_summary.json")
    print("=" * 70)

def train_tgat():
    args = argparse.Namespace(
        input='C:/Users/HP PAVILION/Downloads/labelled_inside.csv',
        out_dir='./out_train',
        epochs=100,
        lr=1e-3,
        hidden=128,
        latent=64,
        k_neighbors=8,
        n_heads=2,
        weight_decay=1e-4,
        nrows=20000,  # 20k samples -> 16k train (under 20k limit) + 4k val
    )
    train(args)
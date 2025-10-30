#!/usr/bin/env python3
"""
infer_tgat.py

Load traced TorchScript model (model.pt) and preproc.joblib, predict on input CSV,
and write output CSV with Label, Confidence, Reason and plots.

No class definitions required to run.
"""

from fastapi import HTTPException
import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import json

# ---------- Config (must mirror training feature engineering) ----------
FEATURE_COLS = [
    'T', 'H', 'PMS1', 'PMS2_5', 'PMS10', 'CO2', 'NO2', 'CO', 'VoC', 'C2H5OH',
    'pm25_max_2min', 'pm25_mean_2min', 'co2_max_2min', 'co2_mean_1min',
    'co2_rise_1to2min', 'T_rise_1min', 'pm10_mean_30s', 'pm10_spike',
    'H_mean_2min', 'voc_mean_1min', 'ethanol_mean_1min'
]
DERIVED = ['co2_pm25_ratio', 'temp_humidity_ratio', 'voc_ethanol_ratio', 'co_rise_rate', 'pm25_co2_correlation']

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
    """
    Generate explanation for predicted label based on sensor values.
    If the label is 'normal', reason should default to 'no rule triggered; baseline normal'.
    """
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
    # os.makedirs(args.out_dir, exist_ok=True)

    # print("=" * 70)
    # print("ANOMALY DETECTION INFERENCE")
    # print("=" * 70)
    # print(f"Model: {args.model}")
    # print(f"Preprocessing: {args.preproc}")
    # print(f"Input data: {args.input}")
    # print(f"Output directory: {args.out_dir}")
    # print()

    #print("Loading model and preprocessing...")
    model = torch.jit.load(args.model, map_location='cpu')
    preproc = joblib.load(args.preproc)
    feature_list = preproc['feature_list']
    scaler = preproc['scaler']
    label_encoder = preproc['label_encoder']
    reason_rules = preproc.get('reason_rules', {})

    # print(f"✓ Model loaded")
    # print(f"✓ Preprocessing loaded ({len(feature_list)} features)")
    # print(f"✓ Classes: {label_encoder.classes_}")
    # print()

    #print("Reading input CSV...")
    #df = pd.read_csv(args.input, parse_dates=['ts'] if 'ts' in pd.read_csv(args.input, nrows=1).columns else None)

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
        # print(f"⚠ Large dataset detected ({total_samples} samples)")
        # print(f"⚠ Processing first {max_samples} samples to avoid memory issues")
        # print(f"⚠ To process all data, run inference on smaller chunks")
        df = df.head(max_samples)
        # print(f"✓ Using {len(df)} samples")
    print()
    
    #print("Engineering features...")
    df_proc = engineer(df)
    X = df_proc[feature_list].values.astype(np.float32)
    X = np.where(np.isinf(X), np.nan, X)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    #print(f"✓ Features engineered")

    # print("Scaling features...")
    Xs = scaler.transform(X)
    # print(f"✓ Features scaled")

    #print("Building graph structure...")
    adj = build_corr_knn_graph(Xs, k=args.k_neighbors)
    # fallback minimal connectivity if degenerate
    if adj.sum() == 0:
        n = Xs.shape[0]
        adj = np.zeros((n, n), dtype=np.float32)
        for i in range(max(1, n-1)):
            adj[i, i+1] = 1.0
            adj[i+1, i] = 1.0

    #print(f"✓ Graph built")
    #print()

    # to tensors
    x_t = torch.FloatTensor(Xs)
    adj_t = torch.FloatTensor(adj)

    #print("Running inference...")
    model.eval()
    with torch.no_grad():
        logits, conf, alphas = model(x_t, adj_t)  # traced wrapper returns logits, conf, alpha
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds_idx = np.argmax(probs, axis=1)
        conf_np = conf.cpu().numpy()
        alpha_np = alphas.cpu().numpy()  # (n, n)
    #print(f"✓ Predictions completed")
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

    # out_csv = os.path.join(args.out_dir, args.output_csv)
    # df_out.to_csv(out_csv, index=False)
    
    # Get counts for summary
    counts = df_out['Label'].value_counts().sort_values(ascending=False)
    
    # print("=" * 70)
    # print("PREDICTION RESULTS")
    # print("=" * 70)
    # print(f"Total samples: {len(df_out)}")
    print(f"\nGenerating anomalies")
    # print(f"\nPredicted label distribution:")
    for label, count in counts.items():
        pct = 100 * count / len(df_out)
        print(f"  {label:30s}: {count:6d} samples ({pct:5.1f}%)")
    # print()
    # print(f"✓ Saved predictions to: {out_csv}")
    # print("=" * 70)
    # print()

    # # ---------------- Visualizations ----------------
    # # 1) Count per label
    # plt.figure(figsize=(8, 4))
    # counts.plot(kind='bar')
    # plt.title('Predicted Label Counts')
    # plt.xlabel('Label')
    # plt.ylabel('Count')
    # plt.tight_layout()
    # plt.savefig(os.path.join(args.out_dir, 'label_counts.png'), dpi=200)
    # plt.close()
    #
    # # 2) Timeline plot for a key sensor (CO2 if present or first feature)
    # key_sensor = 'CO2' if 'CO2' in df_out.columns else feature_list[0]
    # ts = df_out['ts'] if 'ts' in df_out.columns else pd.RangeIndex(len(df_out))
    # plt.figure(figsize=(12, 4))
    # plt.plot(ts, df_out[key_sensor], label=key_sensor)
    # # horizontal normal line: median
    # med = df_out[key_sensor].median()
    # plt.axhline(med, color='green', linestyle='--', label='Normal median')
    # # mark anomalies (non-normal labels)
    # anomalies_mask = df_out['Label'] != 'normal'
    # plt.scatter(ts[anomalies_mask], df_out.loc[anomalies_mask, key_sensor], color='red', s=20, label='Anomaly')
    # plt.title(f'{key_sensor} timeline with detected anomalies')
    # plt.xlabel('ts')
    # plt.ylabel(key_sensor)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(args.out_dir, 'sensor_timeline_with_anomalies.png'), dpi=200)
    # plt.close()
    #
    #
    # anomaly_score = 1.0 - combined_conf
    # plt.figure(figsize=(12, 3))
    # plt.plot(ts, anomaly_score, linewidth=1.2)
    # plt.fill_between(ts, 0, anomaly_score, alpha=0.15)
    # plt.title('Anomaly score (1 - Confidence) — spikes indicate anomalies')
    # plt.xlabel('ts')
    # plt.ylabel('AnomalyScore')
    # plt.tight_layout()
    # plt.savefig(os.path.join(args.out_dir, 'anomaly_spikes.png'), dpi=200)
    # plt.close()
    #
    #
    # try:
    #     n = alpha_np.shape[0]
    #     cap = min(200, n)
    #     plt.figure(figsize=(8, 6))
    #     plt.imshow(alpha_np[:cap, :cap], aspect='auto', cmap='viridis')
    #     plt.colorbar(label='attention weight')
    #     plt.title('Attention matrix (top-left block)')
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(args.out_dir, 'attention_heatmap.png'), dpi=200)
    #     plt.close()
    # except Exception:
    #     pass
    #
    #
    # try:
    #     perfeat = np.abs(Xs - (Xs.mean(axis=0)))
    #     if perfeat.shape[0] > 10 and perfeat.shape[0] > perfeat.shape[1]:
    #         tsne = TSNE(n_components=2, perplexity=min(30, perfeat.shape[0] // 4), random_state=42)
    #         emb = tsne.fit_transform(perfeat)
    #         plt.figure(figsize=(7, 5))
    #         sc = plt.scatter(emb[:, 0], emb[:, 1], c=anomaly_score, cmap='plasma', alpha=0.8)
    #         plt.colorbar(sc, label='AnomalyScore')
    #         plt.title('t-SNE of per-feature deviation (proxy embedding)')
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(args.out_dir, 'perfeat_tsne.png'), dpi=200)
    #         plt.close()
    # except Exception:
    #     pass
    #
    # # Save inference summary
    # summary = {
    #     'n_samples': int(len(df_out)),
    #     'n_anomalies': int((df_out['Label'] != 'normal').sum()),
    #     'label_counts': df_out['Label'].value_counts().to_dict()
    # }
    # joblib.dump(summary, os.path.join(args.out_dir, 'inference_summary.joblib'))
    # with open(os.path.join(args.out_dir, 'inference_summary.json'), 'w') as f:
    #     json.dump(summary, f, indent=2)

    # print("Generating visualizations...")
    # print(f"✓ Saved label_counts.png")
    # print(f"✓ Saved sensor_timeline_with_anomalies.png")
    # print(f"✓ Saved anomaly_spikes.png")
    # print(f"✓ Saved attention_heatmap.png")
    # print(f"✓ Saved inference_summary.json")
    # print()
    # print("=" * 70)
    # print("INFERENCE COMPLETE!")
    # print("=" * 70)
    # print(f"All results saved to: {args.out_dir}")
    # print("=" * 70)


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
        # out_dir='./out_infer',
        k_neighbors=8,
        output_csv='anomaly_predictions.csv',
        max_samples=10000  # Process 10k samples at a time to avoid memory issues
    )

    anomaly_df = infer(args)

    return anomaly_df



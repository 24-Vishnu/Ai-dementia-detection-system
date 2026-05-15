import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Load models & config ──────────────────────────────────────────────────────
model     = joblib.load('models/ensemble_model.pkl')
scaler    = joblib.load('models/scaler.pkl')
config    = json.load(open('models/config.json'))

THRESHOLD       = config['threshold']
FEATURE_NAMES   = config['feature_names']
ACOUSTIC_FEATS  = config['acoustic_features']
LINGUISTIC_FEATS= config['linguistic_features']

print("=" * 60)
print("DEMENTIA DETECTION — INFERENCE")
print("=" * 60)
print(f"Threshold     : {THRESHOLD:.2f}")
print(f"Features used : {len(FEATURE_NAMES)} total")
print(f"  Acoustic    : {len(ACOUSTIC_FEATS)}")
print(f"  Linguistic  : {len(LINGUISTIC_FEATS)}")
print("=" * 60)

# ── Load pre-extracted features ───────────────────────────────────────────────
df_features = pd.read_csv('processed_features_enhanced.csv')

# FIX: tempo stored as string array
df_features['tempo'] = (df_features['tempo']
                         .astype(str)
                         .str.extract(r'([\d.]+)')
                         .astype(float))
df_features = df_features.replace([np.inf, -np.inf], 0).fillna(0)


def test_sample(participant_id, actual_label=None, verbose=True):
    """
    Run inference for a single participant.
    Returns dict with prediction, probabilities, and feature breakdown.
    """
    match = df_features[df_features['participant_id'] == participant_id]
    if len(match) == 0:
        print(f"❌  Participant not found: {participant_id}")
        return None

    # Use all tasks for this participant (average probabilities)
    results_per_task = []
    for _, row in match.iterrows():
        X = row[FEATURE_NAMES].values.reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = np.nan_to_num(scaler.transform(X), nan=0.0)
        prob = model.predict_proba(X_scaled)[0]
        results_per_task.append({
            'task':         row['task'],
            'prob_control': prob[0],
            'prob_dementia':prob[1],
        })

    # Average across tasks
    avg_dem_prob  = np.mean([r['prob_dementia'] for r in results_per_task])
    avg_ctrl_prob = 1 - avg_dem_prob
    prediction    = 1 if avg_dem_prob >= THRESHOLD else 0

    # Feature contribution (acoustic vs linguistic importance from RF)
    # Quick proxy: mean absolute z-score per group
    row0    = match.iloc[0]
    X_all   = row0[FEATURE_NAMES].values
    X_s     = np.nan_to_num(scaler.transform(X_all.reshape(1,-1)), nan=0)[0]
    feat_df = pd.Series(X_s, index=FEATURE_NAMES)
    acoustic_signal  = feat_df[ACOUSTIC_FEATS].abs().mean()
    linguistic_signal= feat_df[LINGUISTIC_FEATS].abs().mean()

    if verbose:
        result_str = "⚠️  DEMENTIA" if prediction == 1 else "✅  CONTROL"
        print(f"\n{'─'*50}")
        print(f"Participant : {participant_id}")
        print(f"Tasks found : {', '.join(match['task'].tolist())}")
        print(f"Result      : {result_str}")
        print(f"Probability : Control={avg_ctrl_prob:.1%}  Dementia={avg_dem_prob:.1%}")

        # Per-task breakdown
        if len(results_per_task) > 1:
            print(f"Per-task probabilities:")
            for r in results_per_task:
                indicator = '→ dementia' if r['prob_dementia'] >= THRESHOLD else '→ control'
                print(f"  [{r['task']:<10}] Dem={r['prob_dementia']:.1%}  {indicator}")

        # Feature group signal
        print(f"Feature signal (z-score magnitude):")
        print(f"  Acoustic   : {acoustic_signal:.3f}")
        print(f"  Linguistic : {linguistic_signal:.3f}")

        if actual_label is not None:
            actual_str = 'Dementia' if actual_label == 1 else 'Control'
            status     = '✅ CORRECT' if prediction == actual_label else '❌ WRONG'
            print(f"Actual      : {actual_str}  |  {status}")

    return {
        'participant_id': participant_id,
        'prediction':     prediction,
        'prob_dementia':  avg_dem_prob,
        'prob_control':   avg_ctrl_prob,
        'per_task':       results_per_task,
    }


def batch_test(csv_path='master_dataset_map.csv', n_control=3, n_dementia=3):
    """Run inference on n_control + n_dementia samples from the dataset map."""
    df_map = pd.read_csv(csv_path)

    # Deduplicate to one row per participant
    df_map_dedup = df_map.drop_duplicates(subset='participant_id')

    ctrl_rows = df_map_dedup[df_map_dedup['label'] == 0].head(n_control)
    dem_rows  = df_map_dedup[df_map_dedup['label'] == 1].head(n_dementia)

    print(f"\n{'='*60}")
    print(f"BATCH TEST — {n_control} control + {n_dementia} dementia samples")
    print(f"{'='*60}")

    print(f"\n🔵  CONTROL SAMPLES:")
    for _, row in ctrl_rows.iterrows():
        test_sample(row['participant_id'], actual_label=row['label'])

    print(f"\n🔴  DEMENTIA SAMPLES:")
    for _, row in dem_rows.iterrows():
        test_sample(row['participant_id'], actual_label=row['label'])


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import os
    if os.path.exists('master_dataset_map.csv'):
        batch_test(n_control=3, n_dementia=3)
    else:
        print("\n⚠️  master_dataset_map.csv not found.")
        print("   Call test_sample(participant_id, actual_label) directly.")
        print("\nExample:")
        print("  from ts import test_sample")
        print("  test_sample('092', actual_label=0)")
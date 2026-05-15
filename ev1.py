import joblib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, roc_auc_score,
                              roc_curve, precision_recall_curve, f1_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': '#f8f9fa',
    'axes.grid': True, 'grid.color': 'white', 'grid.linewidth': 1.2,
    'font.family': 'DejaVu Sans', 'axes.spines.top': False,
    'axes.spines.right': False,
})
BLUE = '#2563EB'; RED = '#DC2626'; GREEN = '#16A34A'
AMBER = '#D97706'; PURPLE = '#7C3AED'

print("=" * 70)
print("COMPREHENSIVE MODEL EVALUATION (participant-level test set)")
print("=" * 70)

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv('processed_features_enhanced.csv')
X  = df.drop(columns=['participant_id', 'task', 'label'])
y  = df['label']
groups = df['participant_id']

X['tempo'] = X['tempo'].astype(str).str.extract(r'([\d.]+)').astype(float)
X = X.replace([np.inf, -np.inf], 0).fillna(0)

# Same participant-level split as training
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_, test_idx = next(gss.split(X, y, groups))
X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

print(f"\nTest set: {len(X_test)} samples  "
      f"(Control={sum(y_test==0)}, Dementia={sum(y_test==1)})")
print(f"Unique participants in test: {df['participant_id'].iloc[test_idx].nunique()}")

# Load models & config
model  = joblib.load('models/ensemble_model.pkl')
scaler = joblib.load('models/scaler.pkl')
config = json.load(open('models/config.json'))

threshold = config['threshold']
print(f"Threshold loaded from config: {threshold:.2f}")

X_test_s = np.nan_to_num(scaler.transform(X_test), nan=0)
y_proba  = model.predict_proba(X_test_s)[:, 1]
y_true   = y_test.values
y_pred   = (y_proba >= threshold).astype(int)

# ── Metrics ───────────────────────────────────────────────────────────────────
cm  = confusion_matrix(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_proba)
ctrl_recall = cm[0,0] / cm[0].sum()
dem_recall  = cm[1,1] / cm[1].sum()
ctrl_prec   = cm[0,0] / (cm[0,0] + cm[1,0]) if (cm[0,0]+cm[1,0]) > 0 else 0
dem_prec    = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1]+cm[0,1]) > 0 else 0

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)
print(f"  Overall Accuracy   : {acc:.4f}")
print(f"  ROC-AUC            : {auc:.4f}")
print(f"  Threshold          : {threshold:.2f}")
print(f"\n  Control  — Recall={ctrl_recall:.4f}  Precision={ctrl_prec:.4f}")
print(f"  Dementia — Recall={dem_recall:.4f}  Precision={dem_prec:.4f}")
print(f"\n  Confusion Matrix:")
print(f"                 Predicted")
print(f"               Ctrl    Dem")
print(f"  Actual Ctrl   {cm[0,0]:<6}  {cm[0,1]:<6}")
print(f"  Actual Dem    {cm[1,0]:<6}  {cm[1,1]:<6}")
print(f"\n  False negatives (dementia missed): {cm[1,0]}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Control','Dementia'], digits=4))

# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n🎨  Generating evaluation plots...")

# ── Fig EV-1: Full evaluation summary dashboard ───────────────────────────────
fig = plt.figure(figsize=(20, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle('Dementia Detection — Full Evaluation Dashboard', fontsize=18,
             fontweight='bold', y=1.01)

# (0,0) Confusion matrix
ax_cm = fig.add_subplot(gs[0, 0])
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
labels = np.array([[f'{cm[i,j]}\n({cm_pct[i,j]:.1%})'
                    for j in range(2)] for i in range(2)])
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', ax=ax_cm,
            xticklabels=['Control','Dementia'],
            yticklabels=['Control','Dementia'],
            linewidths=1, linecolor='white',
            annot_kws={'size': 13, 'weight': 'bold'})
ax_cm.set_title('Confusion Matrix', fontweight='bold', fontsize=12)
ax_cm.set_ylabel('Actual'); ax_cm.set_xlabel('Predicted')

# (0,1) ROC curve
ax_roc = fig.add_subplot(gs[0, 1])
fpr, tpr, roc_thr = roc_curve(y_true, y_proba)
ax_roc.plot(fpr, tpr, color=PURPLE, lw=2.5, label=f'AUC = {auc:.3f}')
ax_roc.plot([0,1],[0,1], 'k--', lw=1)
thr_idx = np.argmin(np.abs(roc_thr - threshold))
ax_roc.scatter(fpr[thr_idx], tpr[thr_idx], s=120, color=RED, zorder=5,
               label=f'Threshold={threshold:.2f}')
ax_roc.fill_between(fpr, tpr, alpha=0.08, color=PURPLE)
ax_roc.set_xlabel('False Positive Rate'); ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC Curve', fontweight='bold', fontsize=12)
ax_roc.legend(fontsize=10)

# (0,2) Precision-Recall
ax_pr = fig.add_subplot(gs[0, 2])
prec, rec, pr_thr = precision_recall_curve(y_true, y_proba)
ax_pr.plot(rec, prec, color=AMBER, lw=2.5, label='PR curve')
ax_pr.axhline(sum(y_true)/len(y_true), color='gray', linestyle='--',
              lw=1.5, label='Baseline')
pr_idx = np.argmin(np.abs(pr_thr - threshold))
ax_pr.scatter(rec[pr_idx], prec[pr_idx], s=120, color=RED, zorder=5,
              label=f'Threshold={threshold:.2f}')
ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision')
ax_pr.set_title('Precision-Recall Curve', fontweight='bold', fontsize=12)
ax_pr.legend(fontsize=10)

# (1,0) Probability histogram
ax_hist = fig.add_subplot(gs[1, 0])
ax_hist.hist(y_proba[y_true==0], bins=30, color=BLUE, alpha=0.65,
             label='Control', density=True)
ax_hist.hist(y_proba[y_true==1], bins=30, color=RED, alpha=0.65,
             label='Dementia', density=True)
ax_hist.axvline(threshold, color='black', linestyle='--', lw=2,
                label=f'Threshold={threshold:.2f}')
ax_hist.set_xlabel('P(Dementia)'); ax_hist.set_ylabel('Density')
ax_hist.set_title('Prediction Probability Distribution', fontweight='bold', fontsize=12)
ax_hist.legend(fontsize=10)

# (1,1) Threshold sweep — F1 macro
ax_thr = fig.add_subplot(gs[1, 1])
thresholds_sweep = np.arange(0.10, 0.90, 0.01)
f1s, accs, dem_recs, ctrl_recs = [], [], [], []
for t in thresholds_sweep:
    yp_t = (y_proba >= t).astype(int)
    f1s.append(f1_score(y_true, yp_t, average='macro', zero_division=0))
    accs.append(accuracy_score(y_true, yp_t))
    cm_t = confusion_matrix(y_true, yp_t, labels=[0,1])
    dem_recs.append(cm_t[1,1]/max(cm_t[1].sum(),1))
    ctrl_recs.append(cm_t[0,0]/max(cm_t[0].sum(),1))
ax_thr.plot(thresholds_sweep, f1s,      color=PURPLE, lw=2, label='Macro F1')
ax_thr.plot(thresholds_sweep, accs,     color=BLUE,   lw=2, label='Accuracy')
ax_thr.plot(thresholds_sweep, dem_recs, color=RED,    lw=1.5, linestyle='--',
            label='Dementia recall')
ax_thr.plot(thresholds_sweep, ctrl_recs,color=GREEN,  lw=1.5, linestyle='--',
            label='Control recall')
ax_thr.axvline(threshold, color='black', linestyle=':', lw=2,
               label=f'Chosen={threshold:.2f}')
ax_thr.set_xlabel('Threshold'); ax_thr.set_ylabel('Score')
ax_thr.set_title('Threshold Sweep', fontweight='bold', fontsize=12)
ax_thr.legend(fontsize=8)

# (1,2) Per-task breakdown
ax_task = fig.add_subplot(gs[1, 2])
task_accs, task_drec, task_names2 = [], [], []
for task in df['task'].unique():
    tidx = df.index[df['task'] == task].tolist()
    tidx_test = [i for i in tidx if i in df.index[test_idx]]
    if len(tidx_test) < 5: continue
    pos = [list(df.index[test_idx]).index(i) for i in tidx_test]
    yt = y_true[pos]; yp = y_pred[pos]
    task_accs.append(accuracy_score(yt, yp))
    cm_t = confusion_matrix(yt, yp, labels=[0,1])
    task_drec.append(cm_t[1,1]/max(cm_t[1].sum(),1))
    task_names2.append(task)
x_t = np.arange(len(task_names2))
ax_task.bar(x_t - 0.2, task_accs, 0.38, color=BLUE,  label='Accuracy', zorder=3)
ax_task.bar(x_t + 0.2, task_drec, 0.38, color=RED,   label='Dem recall',zorder=3)
ax_task.set_xticks(x_t); ax_task.set_xticklabels(task_names2)
ax_task.set_ylim(0, 1.15); ax_task.set_title('Per-Task Performance', fontweight='bold', fontsize=12)
ax_task.legend(fontsize=9)

# (2, span) Metrics summary bar
ax_sum = fig.add_subplot(gs[2, :])
metric_names = ['Accuracy', 'ROC-AUC', 'Control Precision', 'Dementia Precision',
                'Control Recall', 'Dementia Recall', 'Macro F1']
metric_vals  = [acc, auc, ctrl_prec, dem_prec, ctrl_recall, dem_recall,
                f1_score(y_true, y_pred, average='macro')]
bar_colors   = [BLUE, PURPLE, BLUE, RED, BLUE, RED, GREEN]
bars = ax_sum.bar(metric_names, metric_vals, color=bar_colors, width=0.55, zorder=3)
ax_sum.set_ylim(0, 1.15)
ax_sum.set_title('Summary Metrics — Ensemble Model', fontweight='bold', fontsize=12)
for bar, val in zip(bars, metric_vals):
    ax_sum.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontweight='bold', fontsize=11)
ax_sum.tick_params(axis='x', rotation=15)

plt.savefig('eval_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ eval_dashboard.png")

# ── Fig EV-2: Normalised confusion matrix ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Confusion Matrix Analysis', fontsize=14, fontweight='bold')

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Control','Dementia'],
            yticklabels=['Control','Dementia'],
            linewidths=1, linecolor='white',
            annot_kws={'size': 18, 'weight': 'bold'})
axes[0].set_title('Raw Counts', fontweight='bold')
axes[0].set_ylabel('Actual'); axes[0].set_xlabel('Predicted')

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
            xticklabels=['Control','Dementia'],
            yticklabels=['Control','Dementia'],
            linewidths=1, linecolor='white',
            annot_kws={'size': 16, 'weight': 'bold'})
axes[1].set_title('Normalised (row %)', fontweight='bold')
axes[1].set_ylabel('Actual'); axes[1].set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('eval_confusion_detail.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ eval_confusion_detail.png")

print("\n" + "=" * 70)
print("✅  EVALUATION COMPLETE")
print("=" * 70)
print(f"  Accuracy    : {acc:.4f}")
print(f"  AUC         : {auc:.4f}")
print(f"  Ctrl recall : {ctrl_recall:.4f}  (false alarm rate: {1-ctrl_recall:.4f})")
print(f"  Dem  recall : {dem_recall:.4f}  (miss rate: {1-dem_recall:.4f})")
print(f"  Dem  missed : {cm[1,0]} patients predicted healthy but are dementia")
print("=" * 70)
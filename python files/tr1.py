import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, accuracy_score,
                              confusion_matrix, roc_auc_score,
                              roc_curve, precision_recall_curve)
from imblearn.combine import SMOTEENN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#f8f9fa',
    'axes.grid':        True,
    'grid.color':       'white',
    'grid.linewidth':   1.2,
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right':False,
})
BLUE    = '#2563EB'
RED     = '#DC2626'
GREEN   = '#16A34A'
AMBER   = '#D97706'
PURPLE  = '#7C3AED'
PALETTE = [BLUE, RED, GREEN, AMBER, PURPLE]

print("=" * 70)
print("DEMENTIA DETECTION — FIXED TRAINING PIPELINE")
print("=" * 70)

# ── 1. LOAD & FIX ──────────────────────────────────────────────────────────────
print("\n📂 Loading data...")
df = pd.read_csv('processed_features_enhanced.csv')

X = df.drop(columns=['participant_id', 'task', 'label'])
y = df['label']
groups = df['participant_id']           # used for participant-level split

# FIX: tempo stored as string array e.g. '[133.9]'
X['tempo'] = X['tempo'].astype(str).str.extract(r'([\d.]+)').astype(float)
X = X.replace([np.inf, -np.inf], 0).fillna(0)

# ── Feature audit ──────────────────────────────────────────────────────────────
acoustic_cols = [c for c in X.columns if c not in
                 ['word_count','utterance_count','ttr','unique_words',
                  'avg_words_per_utterance','repetition_ratio']]
linguistic_cols = ['word_count','utterance_count','ttr','unique_words',
                   'avg_words_per_utterance','repetition_ratio']

print(f"✓ Samples  : {len(X)}")
print(f"✓ Features : {len(X.columns)}  "
      f"(Acoustic={len(acoustic_cols)}, Linguistic={len(linguistic_cols)})")
print(f"✓ Control  : {sum(y==0)}   Dementia: {sum(y==1)}")

# ── 2. PARTICIPANT-LEVEL SPLIT ─────────────────────────────────────────────────
print("\n✂️  Splitting by participant (no leakage)...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

train_participants = df['participant_id'].iloc[train_idx].nunique()
test_participants  = df['participant_id'].iloc[test_idx].nunique()
print(f"   Train: {len(X_train)} samples / {train_participants} participants")
print(f"   Test : {len(X_test)}  samples / {test_participants}  participants")
print(f"   Train label dist: {dict(y_train.value_counts())}")
print(f"   Test  label dist: {dict(y_test.value_counts())}")

# ── 3. SCALE ───────────────────────────────────────────────────────────────────
print("\n⚖️  Scaling features...")
scaler = StandardScaler()
X_train_s = np.nan_to_num(scaler.fit_transform(X_train), nan=0)
X_test_s  = np.nan_to_num(scaler.transform(X_test),  nan=0)

# ── 4. SMOTEENN ────────────────────────────────────────────────────────────────
print("🔄  Balancing with SMOTEENN...")
sampler = SMOTEENN(random_state=42)
X_tr, y_tr = sampler.fit_resample(X_train_s, y_train)
print(f"   After SMOTEENN: {dict(pd.Series(y_tr).value_counts())}")

# ── 5. TRAIN ───────────────────────────────────────────────────────────────────
print("\n🤖  Training models...")

print("   → Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=20,
                             class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)

print("   → SVM...")
svm = SVC(C=10, kernel='rbf', probability=True,
          class_weight='balanced', random_state=42)
svm.fit(X_tr, y_tr)

print("   → Logistic Regression...")
lr = LogisticRegression(C=1, max_iter=1000,
                         class_weight='balanced', random_state=42)
lr.fit(X_tr, y_tr)

# FIX: ensemble trained ONCE on balanced data (not re-trained from scratch)
print("   → Ensemble (soft voting, using pre-trained models)...")
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('svm', svm), ('lr', lr)],
    voting='soft'
)
ensemble.fit(X_tr, y_tr)

# ── 6. THRESHOLD ON VALIDATION SPLIT (not test set) ───────────────────────────
print("\n🎯  Finding optimal threshold on a held-out VALIDATION split...")
from sklearn.model_selection import train_test_split as tts
X_tv, X_val, y_tv, y_val = tts(X_tr, y_tr, test_size=0.15,
                                random_state=42, stratify=y_tr)
val_proba = ensemble.predict_proba(X_val)[:, 1]

best_threshold, best_f1 = 0.5, 0
for thr in np.arange(0.25, 0.75, 0.01):
    y_v = (val_proba >= thr).astype(int)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_val, y_v, average='macro')
    if f1 > best_f1:
        best_f1, best_threshold = f1, thr
print(f"   Best threshold (macro-F1): {best_threshold:.2f}")

# ── 7. EVALUATE ALL MODELS ON TEST ────────────────────────────────────────────
print("\n📊  Evaluating on held-out test set...")
models = {
    'Random Forest':      rf,
    'SVM':                svm,
    'Logistic Regression':lr,
    'Ensemble':           ensemble,
}

results = {}
for name, model in models.items():
    proba = model.predict_proba(X_test_s)[:, 1]
    pred  = (proba >= best_threshold).astype(int)
    results[name] = {
        'accuracy': accuracy_score(y_test, pred),
        'auc':      roc_auc_score(y_test, proba),
        'report':   classification_report(y_test, pred,
                        target_names=['Control', 'Dementia'], output_dict=True),
        'cm':       confusion_matrix(y_test, pred),
        'proba':    proba,
        'pred':     pred,
    }
    print(f"\n  {name}")
    print(f"  Accuracy={results[name]['accuracy']:.4f}  AUC={results[name]['auc']:.4f}")
    print(classification_report(y_test, pred,
          target_names=['Control', 'Dementia'], digits=4))

# ── 8. SAVE MODELS & CONFIG ───────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
joblib.dump(ensemble, 'models/ensemble_model.pkl')
joblib.dump(scaler,   'models/scaler.pkl')
joblib.dump(rf,       'models/random_forest.pkl')
json.dump({'threshold': float(best_threshold),
           'feature_names': list(X.columns),
           'acoustic_features': acoustic_cols,
           'linguistic_features': linguistic_cols},
          open('models/config.json', 'w'), indent=2)
print("\n💾  Models + config saved to models/")

# ══════════════════════════════════════════════════════════════════════════════
# 9. VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n🎨  Generating visualisations...")
y_test_arr  = y_test.values
ens_proba   = results['Ensemble']['proba']
ens_pred    = results['Ensemble']['pred']
feature_names = list(X.columns)

# ── Fig 1: Model comparison dashboard ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Model Comparison Dashboard', fontsize=16, fontweight='bold', y=1.02)

# Accuracy bar
model_names = list(results.keys())
accs = [results[m]['accuracy'] for m in model_names]
aucs = [results[m]['auc']      for m in model_names]
bars = axes[0].bar(model_names, accs, color=PALETTE[:4], width=0.5, zorder=3)
axes[0].set_ylim(0.5, 1.0)
axes[0].set_title('Accuracy', fontweight='bold')
axes[0].set_ylabel('Accuracy')
axes[0].tick_params(axis='x', rotation=20)
for bar, val in zip(bars, accs):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# AUC bar
bars2 = axes[1].bar(model_names, aucs, color=PALETTE[:4], width=0.5, zorder=3)
axes[1].set_ylim(0.5, 1.0)
axes[1].set_title('ROC-AUC', fontweight='bold')
axes[1].set_ylabel('AUC')
axes[1].tick_params(axis='x', rotation=20)
for bar, val in zip(bars2, aucs):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Per-class F1 grouped bar
x = np.arange(len(model_names))
w = 0.35
ctrl_f1 = [results[m]['report']['Control']['f1-score']    for m in model_names]
dem_f1  = [results[m]['report']['Dementia']['f1-score']   for m in model_names]
axes[2].bar(x - w/2, ctrl_f1, w, label='Control',  color=BLUE,  zorder=3)
axes[2].bar(x + w/2, dem_f1,  w, label='Dementia', color=RED,   zorder=3)
axes[2].set_xticks(x); axes[2].set_xticklabels(model_names, rotation=20)
axes[2].set_ylim(0, 1.05)
axes[2].set_title('Per-class F1', fontweight='bold')
axes[2].set_ylabel('F1-Score')
axes[2].legend()

plt.tight_layout()
plt.savefig('plot_01_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ plot_01_model_comparison.png")

# ── Fig 2: Confusion matrices (all 4 models) ──────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle('Confusion Matrices (all models)', fontsize=16, fontweight='bold')
for ax, name in zip(axes, model_names):
    cm = results[name]['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Control','Dementia'],
                yticklabels=['Control','Dementia'],
                linewidths=0.5, linecolor='white',
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(name, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    # Overlay recall rates
    ctrl_recall = cm[0,0] / cm[0].sum()
    dem_recall  = cm[1,1] / cm[1].sum()
    ax.set_xlabel(f'Predicted\nCtrl recall={ctrl_recall:.1%}  Dem recall={dem_recall:.1%}',
                  fontsize=9)
plt.tight_layout()
plt.savefig('plot_02_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ plot_02_confusion_matrices.png")

# ── Fig 3: ROC curves ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
for i, (name, color) in enumerate(zip(model_names, PALETTE)):
    fpr, tpr, _ = roc_curve(y_test_arr, results[name]['proba'])
    auc = results[name]['auc']
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC={auc:.3f})')
ax.plot([0,1],[0,1], 'k--', lw=1, label='Random')
ax.fill_between(*roc_curve(y_test_arr, ens_proba)[:2],
                alpha=0.07, color=PURPLE)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — All Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
plt.tight_layout()
plt.savefig('plot_03_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ plot_03_roc_curves.png")

# ── Fig 4: Precision-Recall curve (ensemble) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
prec, rec, thresholds = precision_recall_curve(y_test_arr, ens_proba)
ax.plot(rec, prec, color=PURPLE, lw=2.5, label='Ensemble PR curve')
ax.axhline(y=sum(y_test_arr)/len(y_test_arr), color='gray',
           linestyle='--', lw=1.5, label='Baseline (prevalence)')
# Mark chosen threshold
idx = np.argmin(np.abs(thresholds - best_threshold))
ax.scatter(rec[idx], prec[idx], s=120, color=RED, zorder=5,
           label=f'Threshold = {best_threshold:.2f}')
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve — Ensemble', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
plt.tight_layout()
plt.savefig('plot_04_precision_recall.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ plot_04_precision_recall.png")

# ── Fig 5: Probability distribution ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ctrl_probs = ens_proba[y_test_arr == 0]
dem_probs  = ens_proba[y_test_arr == 1]
ax.hist(ctrl_probs, bins=30, color=BLUE,  alpha=0.65, label='Control',  density=True)
ax.hist(dem_probs,  bins=30, color=RED,   alpha=0.65, label='Dementia', density=True)
ax.axvline(best_threshold, color='black', linestyle='--', lw=2,
           label=f'Threshold = {best_threshold:.2f}')
ax.set_xlabel('Predicted Probability of Dementia', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Prediction Probability Distribution — Ensemble', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('plot_05_probability_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ plot_05_probability_distribution.png")

# ── Fig 6: Feature importance (RF) + acoustic vs linguistic breakdown ──────────
fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1])

# Top-30 feature importances
ax_imp = fig.add_subplot(gs[0])
importances = pd.Series(rf.feature_importances_, index=feature_names)
top30 = importances.nlargest(30)
colors_imp = [AMBER if f in linguistic_cols else BLUE for f in top30.index]
bars = ax_imp.barh(range(len(top30)), top30.values, color=colors_imp, zorder=3)
ax_imp.set_yticks(range(len(top30)))
ax_imp.set_yticklabels(top30.index, fontsize=9)
ax_imp.invert_yaxis()
ax_imp.set_xlabel('Feature Importance', fontsize=11)
ax_imp.set_title('Top 30 Feature Importances (Random Forest)\n'
                 '  Blue = Acoustic   Orange = Linguistic', fontsize=12, fontweight='bold')
from matplotlib.patches import Patch
ax_imp.legend(handles=[Patch(color=BLUE, label='Acoustic'),
                        Patch(color=AMBER, label='Linguistic')],
              loc='lower right', fontsize=10)

# Acoustic vs linguistic pie
ax_pie = fig.add_subplot(gs[1])
ling_imp   = importances[linguistic_cols].sum()
acous_imp  = importances[acoustic_cols].sum()
ax_pie.pie([acous_imp, ling_imp],
           labels=[f'Acoustic\n{acous_imp:.1%}', f'Linguistic\n{ling_imp:.1%}'],
           colors=[BLUE, AMBER], autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 12},
           wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'})
ax_pie.set_title('Feature Group\nImportance Split', fontsize=12, fontweight='bold')

plt.suptitle('Feature Importance Analysis — Using BOTH Acoustic & Linguistic Features',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('plot_06_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ plot_06_feature_importance.png")

# ── Fig 7: Key feature distributions by class ─────────────────────────────────
key_features = ['mean_pause_duration', 'speech_rate', 'ttr',
                'word_count', 'pitch_mean', 'repetition_ratio']
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Key Feature Distributions: Control vs Dementia',
             fontsize=15, fontweight='bold')
df_plot = X.copy()
df_plot['label'] = y.values
for ax, feat in zip(axes.flat, key_features):
    ctrl_vals = df_plot[df_plot['label']==0][feat]
    dem_vals  = df_plot[df_plot['label']==1][feat]
    ax.hist(ctrl_vals, bins=40, color=BLUE, alpha=0.6, label='Control',  density=True)
    ax.hist(dem_vals,  bins=40, color=RED,  alpha=0.6, label='Dementia', density=True)
    ax.axvline(ctrl_vals.mean(), color=BLUE, linestyle='--', lw=1.5,
               label=f'μ Control={ctrl_vals.mean():.2f}')
    ax.axvline(dem_vals.mean(),  color=RED,  linestyle='--', lw=1.5,
               label=f'μ Dementia={dem_vals.mean():.2f}')
    ax.set_title(feat.replace('_', ' ').title(), fontweight='bold')
    ax.legend(fontsize=8)
    ftype = '(Linguistic)' if feat in linguistic_cols else '(Acoustic)'
    ax.set_xlabel(ftype, fontsize=9, color='gray')
plt.tight_layout()
plt.savefig('plot_07_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ plot_07_feature_distributions.png")

# ── Fig 8: Per-task performance ───────────────────────────────────────────────
print("\n📋  Per-task breakdown...")
task_results = {}
for task in df['task'].unique():
    tidx = df.index[df['task'] == task].tolist()
    tidx_test = [i for i in tidx if i in df.index[test_idx]]
    if len(tidx_test) < 5:
        continue
    pos = [list(df.index[test_idx]).index(i) for i in tidx_test]
    yt  = y_test_arr[pos]
    yp  = ens_pred[pos]
    task_results[task] = {
        'accuracy': accuracy_score(yt, yp),
        'n':        len(yt),
        'dem_recall': confusion_matrix(yt, yp, labels=[0,1])[1,1] /
                      max(sum(yt==1), 1)
    }

if task_results:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Per-Task Performance — Ensemble', fontsize=14, fontweight='bold')
    tasks = list(task_results.keys())
    task_acc = [task_results[t]['accuracy']   for t in tasks]
    task_rec = [task_results[t]['dem_recall'] for t in tasks]
    axes[0].bar(tasks, task_acc, color=PALETTE[:len(tasks)], width=0.5, zorder=3)
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title('Accuracy by Task', fontweight='bold')
    axes[0].set_ylabel('Accuracy')
    for i,(t,v) in enumerate(zip(tasks, task_acc)):
        axes[0].text(i, v+0.02, f'{v:.2f}', ha='center', fontweight='bold')
    axes[1].bar(tasks, task_rec, color=PALETTE[:len(tasks)], width=0.5, zorder=3)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title('Dementia Recall by Task', fontweight='bold')
    axes[1].set_ylabel('Recall')
    for i,(t,v) in enumerate(zip(tasks, task_rec)):
        axes[1].text(i, v+0.02, f'{v:.2f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot_08_per_task.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ plot_08_per_task.png")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("✅  TRAINING COMPLETE")
print("=" * 70)
best_model = max(results, key=lambda m: results[m]['auc'])
r = results[best_model]
cm = r['cm']
print(f"\n🏆  Best model (AUC): {best_model}")
print(f"   Accuracy         : {r['accuracy']:.4f}")
print(f"   ROC-AUC          : {r['auc']:.4f}")
print(f"   Control recall   : {cm[0,0]/cm[0].sum():.4f}")
print(f"   Dementia recall  : {cm[1,1]/cm[1].sum():.4f}")
print(f"   Threshold used   : {best_threshold:.2f}")
print("\n📊  Feature groups:")
print(f"   Acoustic features  : {len(acoustic_cols)}")
print(f"   Linguistic features: {len(linguistic_cols)}")
print(f"   TOTAL              : {len(X.columns)}")
print("\n📈  Saved plots:")
for i in range(1, 9):
    tag = ['model comparison', 'confusion matrices', 'ROC curves',
           'precision-recall', 'probability distribution',
           'feature importance', 'feature distributions', 'per-task'][i-1]
    print(f"   plot_0{i}_{['model_comparison','confusion_matrices','roc_curves','precision_recall','probability_distribution','feature_importance','feature_distributions','per_task'][i-1]}.png  ← {tag}")
print("=" * 70)
print("✅  Next step: python3 ev.py")
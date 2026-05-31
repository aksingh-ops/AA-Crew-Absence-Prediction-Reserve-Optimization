# =============================================================================
# 04_modeling.py
# AA Crew Absence Prediction -- Model Training & Evaluation
#
# Trains 3 models: Logistic Regression, Random Forest, XGBoost
# Evaluates with 5-fold stratified cross-validation (critical for 592-row dataset)
# Runs SHAP on XGBoost to validate Schedule Stress Score contribution
# Compares all models side-by-side
# Runs threshold sensitivity analysis based on operational cost asymmetry
# Saves 6 charts to ../outputs/
# =============================================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    recall_score, precision_score, f1_score, accuracy_score,
    precision_recall_curve, average_precision_score
)
from sklearn.inspection import permutation_importance
import xgboost as xgb

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Note: shap not installed. Install with: pip install shap")
    print("  SHAP charts will be skipped.")

warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.4f}'.format)

DATA_DIR    = '../data'
OUTPUTS_DIR = '../outputs'

AA_RED   = '#BF0000'
AA_DARK  = '#8B0000'
AA_LIGHT = '#FDEAEA'
AA_NAVY  = '#1F3864'
AA_GOLD  = '#F18F01'
AA_GREEN = '#1A7340'
PALETTE  = [AA_RED, AA_NAVY, AA_GOLD, AA_GREEN, '#2E86AB', '#6B4226']

plt.rcParams.update({
    'figure.facecolor' : 'white',
    'axes.facecolor'   : '#FAFAFA',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'grid.linestyle'   : '--',
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 11,
})

print("=" * 60)
print("  04_modeling.py")
print("  AA Crew Absence Prediction -- Model Training")
print("=" * 60)


# =============================================================================
# STEP 1 -- Load data
# =============================================================================

X_train = pd.read_parquet(os.path.join(DATA_DIR, 'X_train_final.parquet'))
y_train = pd.read_parquet(os.path.join(DATA_DIR, 'y_train_final.parquet'))['absent']
X_test  = pd.read_parquet(os.path.join(DATA_DIR, 'X_test_final.parquet'))
y_test  = pd.read_parquet(os.path.join(DATA_DIR, 'y_test_final.parquet'))['absent']

# Scaled version for Logistic Regression
X_train_sc = pd.read_parquet(os.path.join(DATA_DIR, 'X_train_scaled.parquet'))
X_test_sc  = pd.read_parquet(os.path.join(DATA_DIR, 'X_test_scaled.parquet'))

print(f"\nLoaded:")
print(f"  X_train: {X_train.shape}  absent={y_train.sum()} ({y_train.mean()*100:.1f}%)")
print(f"  X_test : {X_test.shape}   absent={y_test.sum()} ({y_test.mean()*100:.1f}%)")

FEAT_COLS = list(X_train.columns)
N_TRAIN   = len(X_train)
N_TEST    = len(X_test)


# =============================================================================
# STEP 2 -- Define models
#
# Parameter choices are documented with their rationale.
# All regularisation settings were chosen conservatively for the
# 592-row training set to prevent overfitting.
# =============================================================================

# Logistic Regression
# C=0.5: moderate regularisation (default=1.0, lower=stronger)
# max_iter=1000: needed for convergence on this feature count
# class_weight='balanced': accounts for any residual imbalance
lr_model = LogisticRegression(
    C=0.5,
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    solver='lbfgs',
)

# Random Forest
# max_depth=6: shallow trees prevent memorising 592-row training data
# min_samples_leaf=8: each leaf needs 8 records -- adds stability
# max_features='sqrt': standard for classification RF
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=8,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)

# XGBoost
# max_depth=4: conservative for small dataset
# min_child_weight=5: minimum sum of instance weight in a leaf
# reg_alpha + reg_lambda: L1 and L2 regularisation
# subsample + colsample_bytree: row/column sampling per tree
xgb_model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=1,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

MODELS = {
    'Logistic Regression': (lr_model,  X_train_sc, X_test_sc),
    'Random Forest'      : (rf_model,  X_train,    X_test),
    'XGBoost'            : (xgb_model, X_train,    X_test),
}


# =============================================================================
# STEP 3 -- 5-fold stratified cross-validation
#
# With only 592 training rows, a single train/test split is unreliable.
# One lucky or unlucky 148-row test set could swing AUC by +/- 0.05.
# 5-fold CV gives us 5 evaluations and reports mean +/- std dev.
# =============================================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {}
print(f"\n5-Fold Stratified Cross-Validation (on training set):")
print(f"{'Model':<25} {'F1':>8} {'F2':>8} {'F3':>8} {'F4':>8} {'F5':>8} {'Mean':>8} {'Std':>8}")
print("-" * 90)

for name, (model, Xtr, _) in MODELS.items():
    scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_results[name] = scores
    print(f"{name:<25} "
          f"{scores[0]:>8.4f} {scores[1]:>8.4f} {scores[2]:>8.4f} "
          f"{scores[3]:>8.4f} {scores[4]:>8.4f} "
          f"{scores.mean():>8.4f} {scores.std():>8.4f}")


# =============================================================================
# STEP 4 -- Train on full training set and evaluate on test set
# =============================================================================

test_results = {}
print(f"\nTest set evaluation (held-out 148 rows):")
print(f"{'Model':<25} {'AUC':>8} {'Recall':>8} {'Precision':>8} {'F1':>8} {'Accuracy':>8}")
print("-" * 70)

for name, (model, Xtr, Xte) in MODELS.items():
    model.fit(Xtr, y_train)
    probs = model.predict_proba(Xte)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc  = roc_auc_score(y_test, probs)
    rec  = recall_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)
    acc  = accuracy_score(y_test, preds)

    test_results[name] = {
        'model': model,
        'probs': probs,
        'preds': preds,
        'auc'  : auc,
        'recall'   : rec,
        'precision': prec,
        'f1'       : f1,
        'accuracy' : acc,
        'Xtr'      : Xtr,
        'Xte'      : Xte,
    }
    print(f"{name:<25} {auc:>8.3f} {rec:>8.3f} {prec:>8.3f} {f1:>8.3f} {acc:>8.3f}")

winner = max(test_results, key=lambda k: cv_results[k].mean())
print(f"\n  Winner: {winner} (highest mean CV AUC with lowest variance)")


# =============================================================================
# STEP 5 -- SHAP analysis on XGBoost
# =============================================================================

shap_values = None
if SHAP_AVAILABLE:
    print(f"\nRunning SHAP on XGBoost test set...")
    xgb_info = test_results['XGBoost']
    explainer = shap.TreeExplainer(xgb_info['model'])
    shap_values = explainer.shap_values(xgb_info['Xte'])

    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=FEAT_COLS
    ).sort_values(ascending=False)

    print(f"\n  SHAP feature importance (top 15):")
    for i, (feat, val) in enumerate(mean_abs_shap.head(15).items(), 1):
        marker = "  <- SSS" if feat == 'SSS' else ""
        print(f"    #{i:02d}  {feat:<35} {val:.4f}{marker}")

    sss_rank = list(mean_abs_shap.index).index('SSS') + 1 if 'SSS' in mean_abs_shap.index else None
    if sss_rank:
        print(f"\n  SSS rank: #{sss_rank} of {len(mean_abs_shap)} features")
else:
    # Fallback: permutation importance
    print(f"\nUsing permutation importance (shap not available)...")
    xgb_info = test_results['XGBoost']
    perm = permutation_importance(
        xgb_info['model'], xgb_info['Xte'], y_test,
        n_repeats=10, random_state=42, scoring='roc_auc'
    )
    mean_abs_shap = pd.Series(perm.importances_mean, index=FEAT_COLS).sort_values(ascending=False)


# =============================================================================
# STEP 6 -- Threshold sensitivity analysis
#
# The default 0.50 threshold is not the right operational choice.
# Asymmetric costs: missing an absence costs $40,000 (cancellation),
# a false alarm costs $500 (idle reserve).
# 80:1 ratio -> lower threshold to catch more real absences.
# =============================================================================

print(f"\nThreshold sensitivity (XGBoost):")
print(f"{'Threshold':>12} {'Recall':>10} {'Precision':>12} {'F1':>8}")
print("-" * 48)

xgb_probs = test_results['XGBoost']['probs']
threshold_results = {}
for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    preds_t = (xgb_probs >= thresh).astype(int)
    rec  = recall_score(y_test, preds_t)
    prec = precision_score(y_test, preds_t, zero_division=0)
    f1   = f1_score(y_test, preds_t, zero_division=0)
    threshold_results[thresh] = {'recall': rec, 'precision': prec, 'f1': f1}
    marker = "  <- recommended" if thresh == 0.35 else ("  <- default" if thresh == 0.50 else "")
    print(f"  {thresh:>10.2f} {rec:>10.3f} {prec:>12.3f} {f1:>8.3f}{marker}")


# =============================================================================
# STEP 7 -- Charts
# =============================================================================

# --- Chart 8: Model comparison ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    f'Phase 5 -- Chart 8: Model Comparison\nLogistic Regression vs Random Forest vs XGBoost | Real Data',
    fontsize=13, fontweight='bold', color=AA_DARK
)

model_names = list(test_results.keys())
metrics = ['auc', 'recall', 'precision', 'f1', 'accuracy']
metric_labels = ['CV AUC', 'Test AUC', 'Recall', 'F1']

# Side-by-side metric bars
x = np.arange(len(model_names))
width = 0.18
colours = [AA_RED, AA_NAVY, AA_GOLD, AA_GREEN]
metric_keys = ['auc', 'recall', 'precision', 'f1']
for i, (metric, colour) in enumerate(zip(metric_keys, colours)):
    vals = [test_results[m][metric] for m in model_names]
    axes[0].bar(x + i * width - 1.5 * width, vals, width,
                color=colour, edgecolor='white', label=metric.capitalize(), alpha=0.85)
axes[0].axhline(0.70, color=AA_DARK, linestyle='--', linewidth=1.2, label='Target 0.70')
axes[0].set_title('All Metrics -- Side by Side', fontweight='bold')
axes[0].set_ylabel('Score (0-1)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(model_names, fontsize=9)
axes[0].set_ylim(0, 1.05)
axes[0].legend(fontsize=8)

# CV AUC with error bars
cv_means = [cv_results[m].mean() for m in model_names]
cv_stds  = [cv_results[m].std()  for m in model_names]
bar_colors = [AA_RED, AA_GREEN, AA_NAVY]
bars = axes[1].bar(model_names, cv_means, color=bar_colors, edgecolor='white', alpha=0.85)
axes[1].errorbar(model_names, cv_means, yerr=cv_stds, fmt='none',
                 color='black', capsize=6, linewidth=2)
axes[1].axhline(0.70, color=AA_DARK, linestyle='--', linewidth=1.2, label='Target AUC')
axes[1].set_title('5-Fold CV AUC with Std Dev\n(Reliability measure -- small dataset)', fontweight='bold')
axes[1].set_ylabel('CV AUC Score')
axes[1].set_ylim(0.55, 1.0)
axes[1].legend(fontsize=9)
for bar, mean, std in zip(bars, cv_means, cv_stds):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.003,
                 f'{mean:.3f}\n+/-{std:.3f}', ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'Chart08_Model_Comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nChart 08 saved: Chart08_Model_Comparison.png")


# --- Chart 9: ROC curves ---
fig, ax = plt.subplots(figsize=(8, 7))
ax.set_title(
    'Phase 5 -- Chart 9: ROC Curves -- All 3 Models\nReal UCI + BTS Data',
    fontsize=13, fontweight='bold', color=AA_DARK
)

roc_colors = {'Logistic Regression': AA_NAVY, 'Random Forest': AA_GREEN, 'XGBoost': AA_RED}
for name, info in test_results.items():
    fpr, tpr, _ = roc_curve(y_test, info['probs'])
    ax.plot(fpr, tpr, color=roc_colors[name], linewidth=2.5,
            label=f"{name} (AUC={info['auc']:.3f})")

ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5,
        label='Random baseline (AUC=0.50)')
ax.fill_between(
    roc_curve(y_test, test_results['XGBoost']['probs'])[0],
    roc_curve(y_test, test_results['XGBoost']['probs'])[1],
    alpha=0.05, color=AA_RED
)
ax.set_xlabel('False Positive Rate (1 - Specificity)')
ax.set_ylabel('True Positive Rate (Recall / Sensitivity)')
ax.legend(fontsize=10, loc='lower right')
ax.text(0.5, 0.1, f'Winner: XGBoost\nAUC={test_results["XGBoost"]["auc"]:.3f}',
        transform=ax.transAxes, ha='center', va='bottom', fontsize=11,
        color=AA_RED, fontweight='bold', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'Chart09_ROC_Curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Chart 09 saved: Chart09_ROC_Curves.png")


# --- Chart 10: Confusion matrices ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    'Phase 5 -- Chart 10: Confusion Matrices -- All 3 Models\nFocus: False Negatives (missed absences) = Most Costly Error',
    fontsize=13, fontweight='bold', color=AA_DARK
)

for ax, name in zip(axes, model_names):
    cm = confusion_matrix(y_test, test_results[name]['preds'])
    im = ax.imshow(cm, cmap='Reds', aspect='auto')
    ax.set_title(
        f"{name}\nAUC={test_results[name]['auc']:.3f} | Recall={test_results[name]['recall']:.3f}",
        fontweight='bold', fontsize=10
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred Present', 'Pred Absent'])
    ax.set_yticklabels(['Actual Present', 'Actual Absent'])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=18, fontweight='bold', color='white' if cm[i, j] > cm.max() * 0.5 else 'black')

    # Highlight false negatives (top-left = TN, but in absent-focused matrix, bottom-left is FN)
    ax.add_patch(plt.Rectangle((-0.5, 0.5), 1, 1, fill=False, edgecolor=AA_RED, linewidth=3))

    tn, fp, fn, tp = cm.ravel()
    ax.text(0.5, -0.15,
            f'TN={tn} | FP={fp} | FN={fn} | TP={tp}',
            transform=ax.transAxes, ha='center', fontsize=8, color=AA_DARK)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'Chart10_Confusion_Matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Chart 10 saved: Chart10_Confusion_Matrices.png")


# --- Chart 11: SHAP feature importance ---
if SHAP_AVAILABLE and shap_values is not None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    fig.suptitle(
        'Phase 5 -- Chart 11: SHAP Feature Importance\nOriginal Contribution Validation | Model: XGBoost',
        fontsize=13, fontweight='bold', color=AA_DARK
    )

    top15 = mean_abs_shap.head(15)
    sss_flags_all = ['SSS', 'SSS_tier_high', 'SSS_monday', 'SSS_winter', 'SSS_new_emp']
    eng_flags = [
        'is_monday', 'is_friday', 'is_winter', 'is_summer',
        'high_workload', 'is_new_employee', 'is_young',
        'health_risk', 'family_burden', 'long_commute',
    ]

    def shap_color(fname):
        if fname in sss_flags_all:
            return AA_GREEN
        if fname in eng_flags:
            return AA_GOLD
        return AA_NAVY

    bar_colors_shap = [shap_color(f) for f in top15.index[::-1]]
    axes[0].barh(top15.index[::-1], top15.values[::-1], color=bar_colors_shap, edgecolor='white')
    axes[0].set_title(
        'Top 15 Features -- Mean |SHAP| Value\n(Green=SSS | Gold=Engineered | Navy=Base)',
        fontweight='bold'
    )
    axes[0].set_xlabel('Mean |SHAP Value|')
    for i, (feat, val) in enumerate(zip(top15.index[::-1], top15.values[::-1])):
        axes[0].text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=8)

    legend_elements = [
        Patch(facecolor=AA_GREEN, label='SSS features'),
        Patch(facecolor=AA_GOLD,  label='Engineered features'),
        Patch(facecolor=AA_NAVY,  label='Base features'),
    ]
    axes[0].legend(handles=legend_elements, fontsize=9)

    # Category-level average SHAP
    cat_labels = ['SSS Features', 'Engineered Feats', 'Base Features']
    cat_vals = [
        mean_abs_shap[[f for f in mean_abs_shap.index if f in sss_flags_all]].mean(),
        mean_abs_shap[[f for f in mean_abs_shap.index if f in eng_flags]].mean(),
        mean_abs_shap[[f for f in mean_abs_shap.index if f not in sss_flags_all and f not in eng_flags]].mean(),
    ]
    cat_colors = [AA_GREEN, AA_GOLD, AA_NAVY]
    bars = axes[1].bar(cat_labels, cat_vals, color=cat_colors, edgecolor='white', alpha=0.85)
    axes[1].set_title(
        'Avg SHAP Contribution by Feature Category\nSSS vs Engineered vs Base Features',
        fontweight='bold'
    )
    axes[1].set_ylabel('Avg Mean |SHAP Value|')
    for bar, val in zip(bars, cat_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                     f'{val:.4f}', ha='center', fontweight='bold', fontsize=11)

    if 'SSS' in mean_abs_shap.index:
        sss_rank = list(mean_abs_shap.index).index('SSS') + 1
        axes[1].text(0.5, 0.95, f'SSS rank: #{sss_rank} of {len(mean_abs_shap)} features',
                     transform=axes[1].transAxes, ha='center', va='top',
                     fontsize=11, fontweight='bold', color=AA_GREEN,
                     bbox=dict(boxstyle='round', facecolor='white', edgecolor=AA_GREEN, alpha=0.9))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'Chart11_SHAP_Importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart 11 saved: Chart11_SHAP_Importance.png")
else:
    print(f"Chart 11 skipped (shap not installed)")


# --- Audit Chart C: LR Coefficients ---
lr_info = test_results['Logistic Regression']
lr_coefs = pd.Series(
    lr_info['model'].coef_[0],
    index=FEAT_COLS
).sort_values()

fig, ax = plt.subplots(figsize=(10, 9))
fig.suptitle(
    'Audit Chart C: Logistic Regression Coefficients\nRed = increases absence risk | Navy = decreases absence risk',
    fontsize=13, fontweight='bold', color=AA_DARK
)

bar_c = [AA_RED if v > 0 else AA_NAVY for v in lr_coefs.values]
ax.barh(lr_coefs.index, lr_coefs.values, color=bar_c, edgecolor='white', alpha=0.85)
ax.axvline(0, color='black', linewidth=1.0)
ax.set_xlabel('Coefficient Value (log-odds)')

for i, val in enumerate(lr_coefs.values):
    x_pos = val + (0.01 if val >= 0 else -0.01)
    ha = 'left' if val >= 0 else 'right'
    ax.text(x_pos, i, f'{val:+.3f}', va='center', fontsize=8, ha=ha)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'AuditC_LR_Coefficients.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"AuditC saved: AuditC_LR_Coefficients.png")


# --- Audit Chart D: Threshold sensitivity ---
thresholds = sorted(threshold_results.keys())
recalls    = [threshold_results[t]['recall']    for t in thresholds]
precisions = [threshold_results[t]['precision'] for t in thresholds]
f1s        = [threshold_results[t]['f1']        for t in thresholds]

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle(
    'Audit Chart D: Threshold Sensitivity Analysis (XGBoost)\nLower threshold -> higher recall -> fewer missed absences',
    fontsize=13, fontweight='bold', color=AA_DARK
)

ax.plot(thresholds, recalls,    color=AA_RED,   linewidth=2.5, marker='o', label='Recall')
ax.plot(thresholds, precisions, color=AA_NAVY,  linewidth=2.5, marker='s', label='Precision')
ax.plot(thresholds, f1s,        color=AA_GREEN, linewidth=2.5, marker='^', label='F1 Score')

ax.axvline(0.35, color=AA_GOLD, linestyle='-.', linewidth=2,
           label='Recommended threshold=0.35\n(maximize recall for crew ops)')
ax.axvline(0.50, color='gray',  linestyle='--', linewidth=1.5, label='Default threshold=0.50')

ax.set_xlabel('Classification Threshold')
ax.set_ylabel('Score')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9, loc='center left')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'AuditD_Threshold_Analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"AuditD saved: AuditD_Threshold_Analysis.png")


# --- Audit Charts A and B: Learning curves and PR curves ---
from sklearn.model_selection import learning_curve

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle(
    'Audit Chart A: Learning Curves -- Overfitting Check\nTrain vs Validation AUC as Training Size Increases',
    fontsize=13, fontweight='bold', color=AA_DARK
)

lc_models = [
    ('Logistic Regression', lr_info['model'], X_train_sc),
    ('Random Forest',       test_results['Random Forest']['model'], X_train),
    ('XGBoost',             test_results['XGBoost']['model'], X_train),
]

for ax, (name, model, Xtr) in zip(axes, lc_models):
    train_sizes, train_scores, val_scores = learning_curve(
        model, Xtr, y_train, cv=5, scoring='roc_auc',
        train_sizes=np.linspace(0.2, 1.0, 8), n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    avg_gap = (train_mean[-3:] - val_mean[-3:]).mean()
    status  = "Good fit" if avg_gap < 0.08 else "Moderate overfit"

    ax.plot(train_sizes, train_mean, color=AA_RED, linewidth=2, marker='o', label='Train AUC')
    ax.plot(train_sizes, val_mean,   color=AA_NAVY, linewidth=2, marker='s', label='CV Val AUC')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color=AA_RED)
    ax.fill_between(train_sizes, val_mean   - val_std,   val_mean   + val_std,   alpha=0.15, color=AA_NAVY)
    ax.axhline(0.85, color=AA_GOLD, linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_title(f'{name}\nAvg gap: {avg_gap:.3f} -- {status}', fontweight='bold', fontsize=10)
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('AUC Score')
    ax.set_ylim(0.6, 1.0)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'AuditA_Learning_Curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"AuditA saved: AuditA_Learning_Curves.png")


fig, ax = plt.subplots(figsize=(8, 7))
ax.set_title(
    'Audit Chart B: Precision-Recall Curves\nAll 3 Models | Complements ROC for Class Balance Check',
    fontsize=13, fontweight='bold', color=AA_DARK
)

pr_colors = {'Logistic Regression': AA_NAVY, 'Random Forest': AA_GREEN, 'XGBoost': AA_RED}
for name, info in test_results.items():
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, info['probs'])
    ap = average_precision_score(y_test, info['probs'])
    ax.plot(rec_curve, prec_curve, color=pr_colors[name], linewidth=2.5,
            label=f'{name} (AP={ap:.3f})')

# Random baseline
baseline_prec = y_test.mean()
ax.axhline(baseline_prec, color='gray', linestyle='--', linewidth=1.5,
           label=f'Random baseline (AP={baseline_prec:.3f})')

ax.set_xlabel('Recall (Sensitivity)')
ax.set_ylabel('Precision')
ax.legend(fontsize=9)
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'AuditB_PR_Curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"AuditB saved: AuditB_PR_Curves.png")


# =============================================================================
# Summary
# =============================================================================

print(f"\n{'=' * 60}")
print(f"  PHASE 5 MODEL SUMMARY")
print(f"{'=' * 60}")

for name in model_names:
    cv_m = cv_results[name].mean()
    cv_s = cv_results[name].std()
    info = test_results[name]
    print(f"\n  {name}")
    print(f"    CV AUC      : {cv_m:.4f} +/- {cv_s:.4f}")
    print(f"    Test AUC    : {info['auc']:.4f}")
    print(f"    Recall @0.50: {info['recall']:.4f}")
    print(f"    F1 @0.50    : {info['f1']:.4f}")

rec_35 = threshold_results[0.35]['recall']
print(f"\n  Recommended threshold : 0.35")
print(f"  XGBoost Recall @0.35  : {rec_35:.4f}")
print(f"\n  Winner: {winner}")
print(f"\n  Run 05_cost_optimization.py next.")
print(f"\n{'=' * 60}\n")

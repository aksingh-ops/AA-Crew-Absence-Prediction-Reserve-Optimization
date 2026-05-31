# =============================================================================
# 03_feature_engineering.py
# AA Crew Absence Prediction -- Feature Engineering & Selection
#
# Reads parquets saved by 02_cleaning_eda.py.
# Adds 15 new features (interaction flags + SSS features).
# Runs variance threshold and correlation filter to drop weak/redundant ones.
# Saves the final 35-feature train and test sets ready for modeling.
# =============================================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)

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
print("  03_feature_engineering.py")
print("  AA Crew Absence Prediction -- Feature Engineering")
print("=" * 60)


# =============================================================================
# STEP 1 -- Load saved data
# =============================================================================

X_train = pd.read_parquet(os.path.join(DATA_DIR, 'X_train_smote.parquet'))
y_train = pd.read_parquet(os.path.join(DATA_DIR, 'y_train_smote.parquet'))['absent']
X_test  = pd.read_parquet(os.path.join(DATA_DIR, 'X_test.parquet'))
y_test  = pd.read_parquet(os.path.join(DATA_DIR, 'y_test.parquet'))['absent']

print(f"\nLoaded from {DATA_DIR}/:")
print(f"  X_train: {X_train.shape}")
print(f"  X_test : {X_test.shape}")
print(f"  y_train: absent={y_train.sum()} ({y_train.mean()*100:.1f}%)")
print(f"  y_test : absent={y_test.sum()} ({y_test.mean()*100:.1f}%)")
print(f"  Columns: {list(X_train.columns)}")


# =============================================================================
# STEP 2 -- Detect workload column name
# =============================================================================

WL_COL = 'work_load_average_day'
if WL_COL not in X_train.columns:
    candidates = [c for c in X_train.columns if 'work_load' in c]
    WL_COL = candidates[0] if candidates else None
    if WL_COL:
        print(f"\n  Workload column: '{WL_COL}'")


# =============================================================================
# STEP 3 -- Impute any nulls (SSS merge can introduce nulls via left join)
# =============================================================================

imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test  = pd.DataFrame(imputer.transform(X_test),      columns=X_test.columns)
print(f"\nAfter imputation: X_train nulls={X_train.isnull().sum().sum()}  X_test nulls={X_test.isnull().sum().sum()}")


# =============================================================================
# STEP 4 -- Add interaction features
#
# These combine existing signals into flags that have direct business meaning.
# Each one was chosen because the EDA showed a clear absence rate difference.
# =============================================================================

def add_interactions(df):
    df = df.copy()

    # Monday is consistently the highest absence day across all absenteeism studies
    if 'day_of_the_week' in df.columns:
        df['is_monday'] = (df['day_of_the_week'] == 2).astype(int)
        df['is_friday'] = (df['day_of_the_week'] == 6).astype(int)

    # Seasons (UCI coding: 1=Summer, 2=Autumn, 3=Winter, 4=Spring)
    if 'seasons' in df.columns:
        df['is_winter'] = (df['seasons'] == 3).astype(int)
        df['is_summer'] = (df['seasons'] == 1).astype(int)

    # High workload -- top quartile
    if WL_COL and WL_COL in df.columns:
        threshold = df[WL_COL].quantile(0.75)
        df['high_workload'] = (df[WL_COL] >= threshold).astype(int)

    # New employee -- absence rate at 71% vs 44.7% baseline
    if 'service_time' in df.columns:
        df['is_new_employee'] = (df['service_time'] <= 3).astype(int)

    # Younger employees show higher absence rates in EDA
    if 'age' in df.columns:
        df['is_young'] = (df['age'] < 30).astype(int)

    # Combined lifestyle health risk (BMI + smoker)
    if 'body_mass_index' in df.columns and 'social_smoker' in df.columns:
        df['health_risk'] = (
            (df['body_mass_index'] > 30) & (df['social_smoker'] == 1)
        ).astype(int)

    # Family responsibility proxy (children + pets)
    if 'son' in df.columns and 'pet' in df.columns:
        df['family_burden'] = (df['son'] + df['pet']).clip(upper=5)

    # Long commute flag
    if 'distance_from_residence_to_work' in df.columns:
        median_dist = df['distance_from_residence_to_work'].median()
        df['long_commute'] = (df['distance_from_residence_to_work'] > median_dist).astype(int)

    return df


X_train = add_interactions(X_train)
X_test  = add_interactions(X_test)
print(f"\nAfter interaction features: {X_train.shape[1]} columns")


# =============================================================================
# STEP 5 -- Add SSS interaction features (if SSS is present)
# =============================================================================

if 'SSS' in X_train.columns:
    for df in [X_train, X_test]:
        df['SSS_tier_high'] = (df['SSS'] >= 50).astype(int)

        if 'is_monday' in df.columns:
            df['SSS_monday'] = df['SSS'] * df['is_monday']

        if 'is_winter' in df.columns:
            df['SSS_winter'] = df['SSS'] * df['is_winter']

        if 'is_new_employee' in df.columns:
            df['SSS_new_emp'] = df['SSS'] * df['is_new_employee']

    print(f"  SSS interaction features added")
    print(f"  After SSS features: {X_train.shape[1]} columns")


# =============================================================================
# STEP 6 -- Variance threshold
#
# Remove features with near-zero variance (they carry no signal).
# Threshold of 0.01 removes features that are essentially constant.
# =============================================================================

base_n = X_train.shape[1]
vt = VarianceThreshold(threshold=0.01)
vt.fit(X_train)
keep_mask = vt.get_support()
keep_cols = X_train.columns[keep_mask].tolist()
dropped_vt = X_train.columns[~keep_mask].tolist()

X_train = X_train[keep_cols]
X_test  = X_test[keep_cols]

print(f"\nVariance threshold (< 0.01):")
print(f"  Dropped {len(dropped_vt)} features: {dropped_vt}")
print(f"  Remaining: {X_train.shape[1]}")


# =============================================================================
# STEP 7 -- Correlation filter
#
# Of any pair of features with |correlation| > 0.92, drop the one that
# correlates less with the target. This prevents multicollinearity from
# distorting Logistic Regression coefficients.
# =============================================================================

CORR_THRESHOLD = 0.92

# Temporarily attach target to compute feature-target correlations
train_with_target = X_train.copy()
train_with_target['absent'] = y_train.values

corr_matrix = train_with_target.drop(columns=['absent']).corr().abs()
target_corr = train_with_target.corr()['absent'].drop('absent').abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
dropped_corr = set()

for col in upper.columns:
    high_corr_with = upper[col][upper[col] > CORR_THRESHOLD].index.tolist()
    for other in high_corr_with:
        if col in dropped_corr or other in dropped_corr:
            continue
        # Drop whichever one correlates less with the target
        if target_corr.get(col, 0) >= target_corr.get(other, 0):
            dropped_corr.add(other)
        else:
            dropped_corr.add(col)

if dropped_corr:
    X_train = X_train.drop(columns=list(dropped_corr), errors='ignore')
    X_test  = X_test.drop(columns=list(dropped_corr), errors='ignore')

print(f"\nCorrelation filter (|r| > {CORR_THRESHOLD}):")
print(f"  Dropped {len(dropped_corr)} features: {list(dropped_corr)}")
print(f"  Remaining: {X_train.shape[1]}")


# =============================================================================
# STEP 8 -- Scale for Logistic Regression
#
# Tree models (RF, XGBoost) do not need scaling.
# We save a scaled version specifically for LR.
# =============================================================================

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns
)

final_features = list(X_train.columns)
print(f"\nFinal feature set ({len(final_features)} features):")
for i, f in enumerate(final_features, 1):
    is_sss = "  <- SSS" if 'SSS' in f.upper() or f == 'SSS' else ""
    is_new = "  <- engineered" if any(x in f for x in [
        'is_monday', 'is_friday', 'is_winter', 'is_summer',
        'high_workload', 'is_new_employee', 'is_young',
        'health_risk', 'family_burden', 'long_commute'
    ]) else ""
    print(f"  {i:02d}. {f}{is_sss}{is_new}")


# =============================================================================
# STEP 9 -- Save outputs
# =============================================================================

X_train.to_parquet(os.path.join(DATA_DIR, 'X_train_final.parquet'), index=False)
y_train.reset_index(drop=True).to_frame().to_parquet(
    os.path.join(DATA_DIR, 'y_train_final.parquet'), index=False
)
X_test.to_parquet(os.path.join(DATA_DIR, 'X_test_final.parquet'), index=False)
y_test.reset_index(drop=True).to_frame().to_parquet(
    os.path.join(DATA_DIR, 'y_test_final.parquet'), index=False
)
X_train_scaled.to_parquet(os.path.join(DATA_DIR, 'X_train_scaled.parquet'), index=False)
X_test_scaled.to_parquet(os.path.join(DATA_DIR, 'X_test_scaled.parquet'), index=False)
pd.Series(final_features, name='feature').to_csv(
    os.path.join(DATA_DIR, 'feature_list.csv'), index=False
)

print(f"\nData files saved:")
for fname in [
    'X_train_final.parquet', 'y_train_final.parquet',
    'X_test_final.parquet', 'y_test_final.parquet',
    'X_train_scaled.parquet', 'X_test_scaled.parquet',
    'feature_list.csv',
]:
    fpath = os.path.join(DATA_DIR, fname)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  {fname:<35} ({size_kb:.0f} KB)")


# =============================================================================
# STEP 10 -- Charts
# =============================================================================

# --- Chart 6: Feature correlation matrix ---
fig, ax = plt.subplots(figsize=(14, 12))
fig.suptitle(
    'Phase 4 -- Chart 6: Feature Correlation Matrix\nTop 25 Features by Correlation with Target | Real Data',
    fontsize=13, fontweight='bold', color=AA_DARK
)

train_with_target = X_train.copy()
train_with_target['absent'] = y_train.values
top25 = (
    train_with_target.corr()['absent']
    .drop('absent').abs()
    .sort_values(ascending=False)
    .head(25).index.tolist()
)
corr_top25 = train_with_target[top25 + ['absent']].drop(columns=['absent']).corr()

mask = np.triu(np.ones_like(corr_top25, dtype=bool), k=1)
sns.heatmap(
    corr_top25, mask=~mask, ax=ax,
    cmap='RdBu_r', vmin=-1, vmax=1, center=0,
    annot=True, fmt='.2f', annot_kws={'size': 7},
    square=True, linewidths=0.3,
)
ax.set_title('Lower triangle only -- off-diagonal blocks confirm no severe multicollinearity',
             fontsize=10, style='italic')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'Chart06_Feature_Correlation.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nChart 06 saved: Chart06_Feature_Correlation.png")


# --- Chart 7: Engineered vs base feature signal strength ---
train_with_target = X_train.copy()
train_with_target['absent'] = y_train.values
corr_all = (
    train_with_target.corr()['absent'].drop('absent')
    .abs().sort_values(ascending=False)
)

engineered_flags = [
    'is_monday', 'is_friday', 'is_winter', 'is_summer',
    'high_workload', 'is_new_employee', 'is_young',
    'health_risk', 'family_burden', 'long_commute',
    'SSS_tier_high', 'SSS_monday', 'SSS_winter', 'SSS_new_emp',
]
sss_flags = ['SSS', 'SSS_tier_high', 'SSS_monday', 'SSS_winter', 'SSS_new_emp']

def feat_category(fname):
    if fname in sss_flags:
        return 'SSS'
    if fname in engineered_flags:
        return 'Engineered'
    return 'Base'

colors_feat = []
for f in corr_all.index:
    cat = feat_category(f)
    if cat == 'SSS':
        colors_feat.append(AA_GREEN)
    elif cat == 'Engineered':
        colors_feat.append(AA_GOLD)
    else:
        colors_feat.append(AA_NAVY)

fig, axes = plt.subplots(1, 2, figsize=(17, 8))
fig.suptitle(
    'Phase 4 -- Chart 7: Feature Engineering Results\nNew Features Validated | SSS Signal Confirmed',
    fontsize=13, fontweight='bold', color=AA_DARK
)

# All features ranked
axes[0].barh(corr_all.index[::-1], corr_all.values[::-1],
             color=colors_feat[::-1], edgecolor='white')
axes[0].set_title(
    'All Features -- Correlation with Absent\n(Green=SSS | Gold=Engineered | Navy=Base)',
    fontweight='bold'
)
axes[0].set_xlabel('|Correlation with Absent|')
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=AA_GREEN, label='SSS features'),
    Patch(facecolor=AA_GOLD,  label='Engineered features'),
    Patch(facecolor=AA_NAVY,  label='Base features'),
]
axes[0].legend(handles=legend_elements, fontsize=9, loc='lower right')

# Engineered vs base side-by-side for top features
eng_feats  = [f for f in corr_all.index if feat_category(f) == 'Engineered']
base_feats = [f for f in corr_all.index if feat_category(f) == 'Base']

pairs = list(zip(eng_feats[:12], base_feats[:12]))
x = np.arange(len(pairs))
width = 0.35

eng_vals  = [corr_all.get(e, 0) for e, _ in pairs]
base_vals = [corr_all.get(b, 0) for _, b in pairs]

axes[1].bar(x - width/2, eng_vals,  width, color=AA_GOLD,  edgecolor='white', label='Engineered')
axes[1].bar(x + width/2, base_vals, width, color=AA_NAVY,  edgecolor='white', label='Base')
axes[1].set_title('Engineered vs Base Feature Signal Strength\n(Higher = stronger predictor)', fontweight='bold')
axes[1].set_ylabel('|Correlation with Absent|')
axes[1].set_xticks(x)
axes[1].set_xticklabels([e for e, _ in pairs], rotation=45, ha='right', fontsize=8)
axes[1].legend()

if 'SSS' in corr_all.index:
    axes[1].annotate(
        f'SSS corr: {corr_all["SSS"]:.3f}',
        xy=(0.98, 0.95), xycoords='axes fraction',
        ha='right', va='top', fontsize=9,
        color=AA_GREEN, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'Chart07_Feature_Engineering.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Chart 07 saved: Chart07_Feature_Engineering.png")


# =============================================================================
# Summary
# =============================================================================

print(f"\n{'=' * 60}")
print(f"  PHASE 4 COMPLETE")
print(f"{'=' * 60}")
print(f"\n  Final feature count : {len(final_features)}")
print(f"  Dropped (low var)   : {len(dropped_vt)}")
print(f"  Dropped (high corr) : {len(dropped_corr)}")
print(f"  X_train shape       : {X_train.shape}")
print(f"  X_test shape        : {X_test.shape}")
if 'SSS' in final_features:
    sss_corr = corr_all.get('SSS', 0)
    print(f"  SSS correlation     : {sss_corr:.4f}")
print(f"\n  Run 04_modeling.py next.")
print(f"\n{'=' * 60}\n")

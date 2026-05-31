"""
03_feature_engineering.py
===========================
AA Crew Absence Prediction — Feature Engineering
Author: Akash Bhupesh Singh | MS Business Analytics, Iowa State University

PURPOSE: Builds 35 final features from 22 base features.
  - 10 interaction features (day/season/workload/demographic flags)
  - 5 SSS features (original contribution)
  - Feature selection: variance threshold + correlation filter
  - Scaled version for Logistic Regression

USAGE: python 03_feature_engineering.py
"""

import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection  import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute           import SimpleImputer
from imblearn.over_sampling   import SMOTE

os.makedirs('../data', exist_ok=True)

print("=" * 60)
print("  FEATURE ENGINEERING")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────────────────────
uci = pd.read_parquet('../data/uci_raw.parquet')
hub_day = pd.read_parquet('../data/hub_day_sss.parquet')
WL = 'work_load_average_day'

# Base processing
uci['absent'] = (uci['absenteeism_time_in_hours'] >= 4).astype(int)
REASON_MAP={0:'unjustified',1:'medical',2:'medical',3:'medical',4:'medical',
    5:'medical',6:'medical',7:'medical',8:'medical',9:'medical',
    10:'medical',11:'medical',12:'medical',13:'musculoskeletal',14:'medical',
    15:'preventive',16:'medical',17:'medical',18:'medical',19:'musculoskeletal',
    20:'preventive',21:'preventive',22:'preventive',23:'preventive',
    24:'preventive',25:'preventive',26:'unjustified',27:'preventive',28:'preventive'}
uci['reason_group'] = uci['reason_for_absence'].map(REASON_MAP)
rdummies = pd.get_dummies(uci['reason_group'], prefix='reason')
uci = pd.concat([uci, rdummies], axis=1)

# ═══════════════════════════════════════════════════════════
# INTERACTION FEATURES (10 new features)
# ═══════════════════════════════════════════════════════════
uci['is_monday']      = (uci['day_of_the_week']==2).astype(int)
uci['is_friday']      = (uci['day_of_the_week']==6).astype(int)
uci['is_winter']      = (uci['seasons']==3).astype(int)
uci['is_summer']      = (uci['seasons']==1).astype(int)
uci['high_workload']  = (uci[WL]>=uci[WL].quantile(0.75)).astype(int)
uci['is_new_employee']= (uci['service_time']<=3).astype(int)
uci['is_young']       = (uci['age']<30).astype(int)
uci['health_risk']    = ((uci['body_mass_index']>30).astype(int)+
                         (uci['social_smoker']==1).astype(int))
uci['family_burden']  = uci['son'] + uci['pet']
uci['long_commute']   = (uci['distance_from_residence_to_work']>
                          uci['distance_from_residence_to_work'].median()).astype(int)

# ═══════════════════════════════════════════════════════════
# SSS FEATURES (5 new features — ORIGINAL CONTRIBUTION)
# ═══════════════════════════════════════════════════════════
SSS_FEATURES = ['night_dep_pct','long_flight_pct','late_cascade_pct',
                'early_dep_pct','avg_dep_delay','cancel_rate','SSS','total_flights']
sss_dm = hub_day.groupby(['dow','month'])[SSS_FEATURES].mean().reset_index()
uci['dow_bts'] = uci['day_of_the_week'] - 2
uci = uci.merge(sss_dm, left_on=['dow_bts','month_of_absence'],
    right_on=['dow','month'], how='left')
uci = uci.drop(columns=['dow_bts','dow','month'], errors='ignore')

# SSS interaction features
uci['SSS_monday']    = uci['SSS'] * uci['is_monday']
uci['SSS_winter']    = uci['SSS'] * uci['is_winter']
uci['SSS_new_emp']   = uci['SSS'] * uci['is_new_employee']
uci['SSS_high_wl']   = uci['SSS'] * uci['high_workload']
uci['SSS_tier_high'] = (uci['SSS'] >= 50).astype(int)

# ═══════════════════════════════════════════════════════════
# FEATURE SELECTION
# ═══════════════════════════════════════════════════════════
DROP = ['id','reason_for_absence','reason_group','absenteeism_time_in_hours','SSS_raw']
df = uci.drop(columns=DROP, errors='ignore').copy()
df = df.drop(columns=df.select_dtypes(include=['object']).columns, errors='ignore')
feat_cols = [c for c in df.columns if c != 'absent']
X_full = df[feat_cols]; y_full = df['absent']

print(f"Features before selection: {len(feat_cols)}")

# Impute NaNs (from SSS left-join merge)
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X_full), columns=feat_cols)
print(f"NaN after impute: {X_imputed.isnull().sum().sum()}")

# Variance threshold
vt = VarianceThreshold(threshold=0.01)
vt.fit(X_imputed)
low_var = [c for c,k in zip(feat_cols,vt.get_support()) if not k]
print(f"Low-variance dropped: {low_var}")

# High-correlation filter (|r| > 0.92)
corr_m = X_imputed.corr().abs()
upper  = corr_m.where(np.triu(np.ones(corr_m.shape),k=1).astype(bool))
high_c = [col for col in upper.columns if any(upper[col]>0.92)]
print(f"High-correlation dropped: {high_c}")

drop_all    = list(set(low_var + high_c))
X_final     = X_imputed.drop(columns=drop_all, errors='ignore')
final_feat_cols = X_final.columns.tolist()
print(f"Final feature count: {len(final_feat_cols)}")

# ═══════════════════════════════════════════════════════════
# TRAIN/TEST SPLIT + SMOTE + SCALING
# ═══════════════════════════════════════════════════════════
X_tr, X_te, y_tr, y_te = train_test_split(
    X_final, y_full, test_size=0.20, random_state=42, stratify=y_full)

smote = SMOTE(random_state=42, k_neighbors=5)
X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)

scaler = StandardScaler()
X_tr_sc = pd.DataFrame(scaler.fit_transform(X_tr_sm), columns=final_feat_cols)
X_te_sc = pd.DataFrame(scaler.transform(X_te), columns=final_feat_cols)

# Save
X_tr_sm.to_parquet('../data/X_train_final.parquet', index=False)
pd.DataFrame({'absent':y_tr_sm}).to_parquet('../data/y_train_final.parquet', index=False)
X_te.reset_index(drop=True).to_parquet('../data/X_test_final.parquet', index=False)
pd.DataFrame({'absent':y_te.values}).to_parquet('../data/y_test_final.parquet', index=False)
X_tr_sc.to_parquet('../data/X_train_scaled.parquet', index=False)
X_te_sc.to_parquet('../data/X_test_scaled.parquet', index=False)
pd.Series(final_feat_cols).to_frame('feature').to_csv('../data/feature_list.csv', index=False)

print(f"\nTrain (SMOTE): {X_tr_sm.shape} | absent={y_tr_sm.sum()} ({y_tr_sm.mean()*100:.0f}%)")
print(f"Test         : {X_te.shape}  | absent={y_te.sum()} ({y_te.mean()*100:.1f}%)")
print("\n✅ All Phase 4 parquets saved")
print("   Next: Run 04_modeling.py")

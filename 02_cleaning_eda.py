"""
02_cleaning_eda.py
===================
AA Crew Absence Prediction — Data Cleaning & EDA
Author: Akash Bhupesh Singh | MS Business Analytics, Iowa State University

PURPOSE: Cleans both datasets, performs EDA, builds Schedule Stress Score (SSS).
  - UCI: binary target, reason grouping, SMOTE, train/test split
  - BTS: parse departure times, build SSS signals, hub-day aggregation

ORIGINAL CONTRIBUTION: Schedule Stress Score (SSS)
  Composite feature from 4 BTS signals:
    night_dep_pct (w=0.35) + long_flight_pct (w=0.30) +
    late_cascade_pct (w=0.20) + early_dep_pct (w=0.15)
  Weights derived from Springer Nature (2024) effect size rankings.

USAGE: python 02_cleaning_eda.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
os.makedirs('../outputs', exist_ok=True)

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ── Colors (AA brand) ─────────────────────────────────────────────────────────
AA_RED='#BF0000'; AA_DARK='#8B0000'; AA_LIGHT='#FDEAEA'
AA_NAVY='#1F3864'; AA_GOLD='#F18F01'; AA_GREEN='#1A7340'
PALETTE=[AA_RED,AA_NAVY,AA_GOLD,AA_GREEN,'#2E86AB','#6B4226']

plt.rcParams.update({
    'figure.facecolor':'white','axes.facecolor':'#FAFAFA',
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':0.3,'grid.linestyle':'--',
    'font.family':'sans-serif','font.size':11
})

print("=" * 60)
print("  DATA CLEANING & EDA")
print("=" * 60)

# ═══════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════
uci = pd.read_parquet('../data/uci_raw.parquet')
bts = pd.read_parquet('../data/bts_aa_dfw_clt_2022_2024.parquet')
bts['FlightDate'] = pd.to_datetime(bts['FlightDate'])
bts['date'] = bts['FlightDate'].dt.date
bts['dow']  = bts['FlightDate'].dt.dayofweek
WL = 'work_load_average_day'

print(f"UCI: {uci.shape} | Missing: {uci.isnull().sum().sum()}")
print(f"BTS: {bts.shape}")

# ═══════════════════════════════════════════════════════════
# UCI CLEANING
# ═══════════════════════════════════════════════════════════

# Binary target: absent = >= 4 hours (half-day = operationally significant)
# Rationale: 4-hour absence triggers reserve call-out in airline operations
ABSENT_THRESHOLD = 4
uci['absent'] = (uci['absenteeism_time_in_hours'] >= ABSENT_THRESHOLD).astype(int)

print(f"\nTarget distribution:")
print(f"  Absent (1): {uci['absent'].sum()} ({uci['absent'].mean()*100:.1f}%)")
print(f"  Present (0): {(uci['absent']==0).sum()} ({(1-uci['absent'].mean())*100:.1f}%)")

# Reason code grouping (28 ICD-10 codes → 4 operational groups)
# Based on aviation medical literature and JD requirement for root cause analysis
REASON_MAP = {
    0:'unjustified', 1:'medical', 2:'medical', 3:'medical', 4:'medical',
    5:'medical', 6:'medical', 7:'medical', 8:'medical', 9:'medical',
    10:'medical', 11:'medical', 12:'medical', 13:'musculoskeletal',
    14:'medical', 15:'preventive', 16:'medical', 17:'medical', 18:'medical',
    19:'musculoskeletal', 20:'preventive', 21:'preventive', 22:'preventive',
    23:'preventive', 24:'preventive', 25:'preventive', 26:'unjustified',
    27:'preventive', 28:'preventive'
}
uci['reason_group'] = uci['reason_for_absence'].map(REASON_MAP)
rdummies = pd.get_dummies(uci['reason_group'], prefix='reason')
uci = pd.concat([uci, rdummies], axis=1)

# Train/test split (BEFORE SMOTE — never leak test data)
DROP = ['id','reason_for_absence','reason_group','absenteeism_time_in_hours']
uci_model = uci.drop(columns=DROP, errors='ignore').copy()
feat_cols = [c for c in uci_model.columns if c != 'absent']
X = uci_model[feat_cols]; y = uci_model['absent']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

# SMOTE: training set only — NEVER apply to test set
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"\nTrain (before SMOTE): {len(X_train)} | absent={y_train.sum()} ({y_train.mean()*100:.1f}%)")
print(f"Train (after SMOTE) : {len(X_train_sm)} | absent={y_train_sm.sum()} ({y_train_sm.mean()*100:.1f}%)")
print(f"Test set            : {len(X_test)} | absent={y_test.sum()} ({y_test.mean()*100:.1f}%)")

# ═══════════════════════════════════════════════════════════
# BTS CLEANING & SSS CONSTRUCTION
# ═══════════════════════════════════════════════════════════

def parse_hour(val):
    """Convert HHMM integer (e.g. 1435) → hour integer (14)."""
    try:
        return int(str(int(val)).zfill(4)[:2])
    except:
        return np.nan

bts['dep_hour']       = bts['CRSDepTime'].apply(parse_hour)
bts['DepDelay']       = bts['DepDelay'].fillna(0)   # cancelled = no delay
bts['ArrDelay']       = bts['ArrDelay'].fillna(0)

# Four SSS signals (based on Springer 2024 paper findings)
bts['is_night_dep']   = ((bts['dep_hour']>=22)|(bts['dep_hour']<=2)).astype(int)
bts['is_long_flight'] = (bts['CRSElapsedTime']>240).astype(int)  # >4 hours
bts['is_late_cascade']= ((bts['DepDelay']>=30)&(bts['dep_hour']>=17)).astype(int)
bts['is_early_dep']   = ((bts['dep_hour']>=4)&(bts['dep_hour']<=6)).astype(int)

# Hub-day aggregation (ORIGINAL CONTRIBUTION: individual flight → operational level)
hub_day = bts.groupby(['Origin','date','month','dow','year']).agg(
    total_flights     =('Cancelled','count'),
    cancelled_flights =('Cancelled','sum'),
    night_dep_count   =('is_night_dep','sum'),
    long_flight_count =('is_long_flight','sum'),
    late_cascade_count=('is_late_cascade','sum'),
    early_dep_count   =('is_early_dep','sum'),
    avg_dep_delay     =('DepDelay','mean'),
    avg_arr_delay     =('ArrDelay','mean'),
    avg_flight_dur    =('CRSElapsedTime','mean'),
).reset_index()

# Rate-based signals
hub_day['night_dep_pct']    = hub_day['night_dep_count']   /hub_day['total_flights']
hub_day['long_flight_pct']  = hub_day['long_flight_count'] /hub_day['total_flights']
hub_day['late_cascade_pct'] = hub_day['late_cascade_count']/hub_day['total_flights']
hub_day['early_dep_pct']    = hub_day['early_dep_count']   /hub_day['total_flights']
hub_day['cancel_rate']      = hub_day['cancelled_flights'] /hub_day['total_flights']

# Schedule Stress Score (ORIGINAL CONTRIBUTION)
# Weights from Springer Nature (2024) effect size rankings
W = {
    'night_dep_pct'   : 0.35,  # Strongest — night shifts increase sick-call rate most
    'long_flight_pct' : 0.30,  # Doubles sick-call probability (4-sector days)
    'late_cascade_pct': 0.20,  # Rotation penalty proxy
    'early_dep_pct'   : 0.15,  # Recovery deficit proxy
}
hub_day['SSS_raw'] = sum(hub_day[sig]*w for sig,w in W.items())
smin,smax = hub_day['SSS_raw'].min(), hub_day['SSS_raw'].max()
hub_day['SSS'] = ((hub_day['SSS_raw']-smin)/(smax-smin)*100).round(2)
hub_day['SSS_tier'] = pd.cut(hub_day['SSS'],bins=[-1,25,50,75,101],
    labels=['Low (0-25)','Moderate (25-50)','High (50-75)','Critical (75-100)'])

print(f"\nHub-day records : {len(hub_day):,}")
print(f"SSS range       : {hub_day['SSS'].min():.1f} – {hub_day['SSS'].max():.1f}")
print(f"SSS mean        : {hub_day['SSS'].mean():.1f}")
for hub in ['DFW','CLT']:
    sub = hub_day[hub_day['Origin']==hub]
    print(f"  {hub}: mean={sub['SSS'].mean():.1f} | max={sub['SSS'].max():.1f} | "
          f"critical={(sub['SSS']>75).sum()} days")

# ── Save all Phase 3 outputs ──────────────────────────────────────────────────
hub_day.to_parquet('../data/hub_day_sss.parquet', index=False)
pd.DataFrame(X_train_sm, columns=feat_cols).to_parquet('../data/X_train_smote.parquet', index=False)
pd.DataFrame({'absent':y_train_sm}).to_parquet('../data/y_train_smote.parquet', index=False)
X_test.reset_index(drop=True).to_parquet('../data/X_test.parquet', index=False)
pd.DataFrame({'absent':y_test.values}).to_parquet('../data/y_test.parquet', index=False)

print("\n✅ All Phase 3 parquets saved")
print("   Next: Run 03_feature_engineering.py")

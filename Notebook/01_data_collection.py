# =============================================================================
# 01_data_collection.py
# AA Crew Absence Prediction -- Data Collection & Initial Profiling
#
# Datasets:
#   1. UCI Absenteeism at Work (local CSV, sep=';')
#      Source: archive.ics.uci.edu/dataset/445/absenteeism+at+work
#      License: CC BY 4.0
#
#   2. BTS On-Time Performance (local parquet)
#      Source: transtats.bts.gov
#      License: Public Domain, U.S. Federal Government
#
# Run this first. It verifies both datasets are present and readable,
# prints a full profile of each, and saves nothing -- just confirms
# the data is in good shape before any cleaning happens.
# =============================================================================

import os
import sys
import zipfile
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', '{:.3f}'.format)

# --------------------------------------------------------------------------
# Paths -- adjust if your folder layout is different
# --------------------------------------------------------------------------
DATA_DIR    = '../data'
OUTPUTS_DIR = '../outputs'
UCI_ZIP     = os.path.join(DATA_DIR, 'Absenteeism_at_work.zip')
UCI_CSV     = os.path.join(DATA_DIR, 'Absenteeism_at_work.csv')
BTS_PARQUET = os.path.join(DATA_DIR, 'bts_aa_dfw_clt_2022_2024.parquet')

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# AA brand colours -- used consistently across all notebooks
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
print("  01_data_collection.py")
print("  AA Crew Absence Prediction -- Data Verification")
print("=" * 60)


# =============================================================================
# SECTION 1 -- UCI Absenteeism at Work
# =============================================================================

def load_uci():
    """
    Load UCI dataset from local file.
    Tries ZIP first, then falls back to plain CSV.
    Returns a cleaned-column DataFrame.
    """
    if os.path.exists(UCI_ZIP):
        print(f"\nLoading UCI from ZIP: {UCI_ZIP}")
        with zipfile.ZipFile(UCI_ZIP) as z:
            with z.open('Absenteeism_at_work.csv') as f:
                df = pd.read_csv(f, sep=';')
    elif os.path.exists(UCI_CSV):
        print(f"\nLoading UCI from CSV: {UCI_CSV}")
        df = pd.read_csv(UCI_CSV, sep=';')
    else:
        print(f"\nERROR: UCI data not found.")
        print(f"  Expected ZIP : {UCI_ZIP}")
        print(f"  Expected CSV : {UCI_CSV}")
        print(f"  Download from: https://archive.ics.uci.edu/dataset/445/absenteeism+at+work")
        sys.exit(1)

    # Standardise column names
    df.columns = [
        c.strip().lower().replace(' ', '_').replace('/', '_')
        for c in df.columns
    ]
    return df


uci = load_uci()

print(f"\n  Rows     : {uci.shape[0]:,}")
print(f"  Columns  : {uci.shape[1]}")
print(f"  Missing  : {uci.isnull().sum().sum()}")

print(f"\n  All columns:")
for i, col in enumerate(uci.columns, 1):
    dtype  = str(uci[col].dtype)
    nuniq  = uci[col].nunique()
    n_null = uci[col].isnull().sum()
    print(f"    {i:02d}. {col:<42} dtype={dtype:<8}  unique={nuniq:<5}  nulls={n_null}")

print(f"\n  Target column: absenteeism_time_in_hours")
tgt = uci['absenteeism_time_in_hours']
print(f"    Min    : {tgt.min():.0f}")
print(f"    Max    : {tgt.max():.0f}")
print(f"    Mean   : {tgt.mean():.1f}")
print(f"    Median : {tgt.median():.1f}")
print(f"    Zeros  : {(tgt == 0).sum()}")

# Detect the workload column name -- real file has no trailing underscore
WL_COL = 'work_load_average_day'
if WL_COL not in uci.columns:
    # Some downloads have a trailing underscore
    candidates = [c for c in uci.columns if 'work_load' in c]
    if candidates:
        WL_COL = candidates[0]
        print(f"\n  Note: workload column found as '{WL_COL}'")
    else:
        print(f"\n  WARNING: workload column not found -- check column names above")

print(f"\n  First 5 rows:")
print(uci.head().to_string())


# =============================================================================
# SECTION 2 -- BTS On-Time Performance
# =============================================================================

def load_bts():
    if not os.path.exists(BTS_PARQUET):
        print(f"\nERROR: BTS parquet not found: {BTS_PARQUET}")
        print(f"  Download from: https://transtats.bts.gov/DL_SelectFields.aspx")
        print(f"  Filter: Carrier=AA, Origin=DFW or CLT, Years=2022-2024")
        print(f"  Fields: FlightDate, Reporting_Airline, Origin, CRSDepTime,")
        print(f"           DepDelay, ArrDelay, Cancelled, CRSElapsedTime, Distance")
        sys.exit(1)

    print(f"\nLoading BTS from parquet: {BTS_PARQUET}")
    df = pd.read_parquet(BTS_PARQUET)
    return df


bts = load_bts()
bts['FlightDate'] = pd.to_datetime(bts['FlightDate'])

print(f"\n  Rows     : {len(bts):,}")
print(f"  Columns  : {bts.shape[1]}")
print(f"  Columns  : {list(bts.columns)}")
print(f"\n  Flights by hub:")
for hub, count in bts['Origin'].value_counts().items():
    pct = count / len(bts) * 100
    print(f"    {hub}: {count:>7,}  ({pct:.1f}%)")

print(f"\n  Date range: {bts['FlightDate'].min().date()} to {bts['FlightDate'].max().date()}")

dep_nulls = bts['DepDelay'].isnull().sum()
print(f"\n  DepDelay nulls: {dep_nulls:,}  (these are cancelled flights -- correct)")
print(f"  Cancellation rate: {bts['Cancelled'].mean()*100:.2f}%")
print(f"  Avg dep delay (all): {bts['DepDelay'].mean():.1f} min")

print(f"\n  First 5 rows:")
print(bts.head().to_string())


# =============================================================================
# SECTION 3 -- Quick diagnostic charts
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    'Data Collection -- Dataset Profiles\nUCI Absenteeism + BTS On-Time Performance (AA)',
    fontsize=13, fontweight='bold', color=AA_DARK
)

# UCI absence hours distribution
axes[0].hist(
    uci['absenteeism_time_in_hours'].clip(upper=80),
    bins=35, color=AA_NAVY, alpha=0.85, edgecolor='white'
)
axes[0].axvline(4, color=AA_RED, linewidth=2.5, linestyle='--',
                label='Threshold = 4 hrs')
axes[0].set_title('UCI -- Absence Hours Distribution', fontweight='bold')
axes[0].set_xlabel('Hours Absent')
axes[0].set_ylabel('Frequency')
axes[0].legend(fontsize=9)

# BTS flights by hub
hub_counts = bts['Origin'].value_counts()
colors_hub = [AA_RED, AA_NAVY]
bars = axes[1].bar(hub_counts.index, hub_counts.values,
                   color=colors_hub, edgecolor='white')
axes[1].set_title('BTS -- American Airlines Flights by Hub\n(2022-2024)', fontweight='bold')
axes[1].set_ylabel('Flight Count')
axes[1].yaxis.set_major_formatter(
    mtick.FuncFormatter(lambda x, _: f'{x/1e6:.2f}M' if x >= 1e6 else f'{x/1e3:.0f}K')
)
for bar, val in zip(bars, hub_counts.values):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2000,
        f'{val:,}',
        ha='center', fontweight='bold', fontsize=10
    )

plt.tight_layout()
out_path = os.path.join(OUTPUTS_DIR, '00_data_profiles.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nChart saved: {out_path}")


# =============================================================================
# SECTION 4 -- Summary
# =============================================================================

print(f"\n{'=' * 60}")
print(f"  DATA COLLECTION SUMMARY")
print(f"{'=' * 60}")
print(f"\n  Dataset 1 -- UCI Absenteeism at Work")
print(f"    Rows    : {len(uci):,}")
print(f"    Columns : {uci.shape[1]}")
print(f"    Missing : {uci.isnull().sum().sum()}")
print(f"    Source  : CC BY 4.0, Martiniano & Ferreira (2012)")

print(f"\n  Dataset 2 -- BTS On-Time Performance")
print(f"    Rows    : {len(bts):,}")
print(f"    Hubs    : DFW + CLT")
print(f"    Years   : 2022 - 2024")
print(f"    Source  : Public Domain, U.S. Federal Government")

print(f"\n  Both datasets loaded successfully.")
print(f"  Run 02_cleaning_eda.py next.")
print(f"\n{'=' * 60}\n")

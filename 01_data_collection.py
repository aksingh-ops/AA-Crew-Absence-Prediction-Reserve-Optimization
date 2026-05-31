"""
01_data_collection.py
======================
AA Crew Absence Prediction — Data Collection
Author: Akash Bhupesh Singh | MS Business Analytics, Iowa State University

PURPOSE: Downloads and caches both datasets programmatically.
  - Dataset 1: UCI Absenteeism at Work (via ucimlrepo API)
  - Dataset 2: BTS On-Time Performance (via PREZIP endpoint)

DATASETS:
  UCI: archive.ics.uci.edu/dataset/445/absenteeism+at+work | License: CC BY 4.0
  BTS: transtats.bts.gov | U.S. Federal Government | Public Domain

RESEARCH PAPERS:
  [1] Homaie Shandizi (2014) — Pilot Absenteeism Prediction, Polytechnique Montreal
  [2] Springer Nature (2024) — Schedule Characteristics & Pilot Absenteeism
  [3] Atkin (2019) — Reserve Crew Scheduling with Probabilistic Absence Model
  [4] MDPI Applied Sciences (2024) — Employee Absence Prediction with ML
  [5] Transportation Science, INFORMS (2025) — Large-Scale Crew Recovery with ML

USAGE: python 01_data_collection.py
"""

import pandas as pd
import numpy as np
import requests
import zipfile
import io
import os
import warnings

warnings.filterwarnings('ignore')
os.makedirs('../data', exist_ok=True)

print("=" * 60)
print("  DATA COLLECTION — AA CREW ABSENCE PREDICTION")
print("=" * 60)

# ═══════════════════════════════════════════════════════════
# DATASET 1 — UCI Absenteeism at Work
# Source: archive.ics.uci.edu/dataset/445
# License: Creative Commons Attribution 4.0 (CC BY 4.0)
# Citation: Martiniano & Ferreira (2012)
# ═══════════════════════════════════════════════════════════
print("\n[1] Loading UCI Absenteeism at Work dataset...")

try:
    # Method 1: Official ucimlrepo API (recommended)
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=445)
    uci = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    uci.columns = [c.strip().lower().replace(' ', '_').replace('/', '_')
                   for c in uci.columns]
    print(f"  ✅ UCI loaded via API: {uci.shape[0]} rows | {uci.shape[1]} cols")
except Exception as e:
    print(f"  ⚠️  API failed ({e}). Download manually from:")
    print("  URL: archive.ics.uci.edu/dataset/445/absenteeism+at+work")
    print("  Place Absenteeism_at_work.csv in ../data/ and re-run.")
    raise SystemExit(1)

uci.to_parquet('../data/uci_raw.parquet', index=False)
print(f"  ✅ Saved: ../data/uci_raw.parquet")
print(f"  Missing values: {uci.isnull().sum().sum()}")
print(f"  Columns: {list(uci.columns)}")

# ═══════════════════════════════════════════════════════════
# DATASET 2 — BTS On-Time Performance
# Source: transtats.bts.gov
# License: Public Domain (U.S. Federal Government)
# ═══════════════════════════════════════════════════════════
CACHE_PATH = '../data/bts_aa_dfw_clt_2022_2024.parquet'

if os.path.exists(CACHE_PATH):
    print(f"\n[2] BTS cache found. Loading from: {CACHE_PATH}")
    bts = pd.read_parquet(CACHE_PATH)
    print(f"  ✅ BTS loaded from cache: {len(bts):,} rows")
else:
    print("\n[2] Downloading BTS On-Time Performance data...")
    print("  Fetching AA flights at DFW + CLT for 2022, 2023, 2024")
    print("  This will take approximately 5-10 minutes...")

    KEEP_COLS = [
        'FlightDate', 'Reporting_Airline', 'Origin', 'Dest',
        'CRSDepTime', 'DepTime', 'DepDelay', 'ArrDelay',
        'Cancelled', 'CRSElapsedTime', 'Distance'
    ]

    def fetch_bts_month(year: int, month: int) -> pd.DataFrame:
        """Download one month of BTS On-Time data for AA at DFW/CLT."""
        url = (
            f"https://transtats.bts.gov/PREZIP/"
            f"On_Time_Reporting_Carrier_On_Time_Performance_"
            f"1987_present_{year}_{month}.zip"
        )
        print(f"  Downloading {year}-{month:02d}...", end=' ', flush=True)
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                csv_name = [n for n in z.namelist() if n.endswith('.csv')][0]
                with z.open(csv_name) as f:
                    df = pd.read_csv(f, usecols=lambda c: c in KEEP_COLS,
                                     low_memory=False)
            df = df[
                (df['Reporting_Airline'] == 'AA') &
                (df['Origin'].isin(['DFW', 'CLT']))
            ].copy()
            df['year'] = year
            df['month'] = month
            print(f"✅  {len(df):,} AA flights at DFW/CLT")
            return df
        except Exception as e:
            print(f"❌  {e}")
            return pd.DataFrame()

    frames = []
    for year in [2022, 2023, 2024]:
        print(f"\n  Year {year}:")
        for month in range(1, 13):
            df_m = fetch_bts_month(year, month)
            if not df_m.empty:
                frames.append(df_m)

    bts = pd.concat(frames, ignore_index=True)
    bts['FlightDate'] = pd.to_datetime(bts['FlightDate'])
    bts.to_parquet(CACHE_PATH, index=False)
    print(f"\n  ✅ BTS cached: {len(bts):,} flights → {CACHE_PATH}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  DATA COLLECTION COMPLETE")
print("=" * 60)
print(f"  UCI rows       : {len(uci):,}")
print(f"  BTS rows       : {len(bts):,}")
print(f"  BTS DFW        : {(bts['Origin']=='DFW').sum():,}")
print(f"  BTS CLT        : {(bts['Origin']=='CLT').sum():,}")
print(f"  BTS date range : {bts['FlightDate'].min().date()} → {bts['FlightDate'].max().date()}")
print("\n  Next: Run 02_cleaning_eda.py")

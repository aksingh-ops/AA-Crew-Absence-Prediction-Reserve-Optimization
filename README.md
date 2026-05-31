# AA Crew Absence Prediction & Reserve Staffing Optimization

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)
[![XGBoost AUC](https://img.shields.io/badge/XGBoost%20AUC-0.869-red.svg)](#results)

## Project Overview

This project replicates the analytical workflow of American Airlines' **Crew Strategy & Intelligence** team (Requisition ID: 85928). It predicts daily crew sick-call volume at DFW and CLT hubs using employee absence patterns and aviation schedule characteristics, then optimizes reserve crew staffing to minimize operational cost.

**Target Role:** Analyst/Sr Analyst, Crew Strategy & Intelligence — American Airlines, DFW

---

## Original Contributions

This project extends published research in three ways no academic paper has achieved:

1. **Schedule Stress Score (SSS)** — Novel composite feature engineered from 820,876 real BTS flight records combining night departure concentration, long-haul flight percentage, late cascade delays, and early morning departures into one hub-day stress metric. Ranked **#4 of 35 features by SHAP**.

2. **Hub-Level Aggregation** — Individual absence predictions aggregated to hub-day operational level, matching how AA's Crew Scheduling team actually manages reserve pools. No paper does this aggregation step.

3. **Asymmetric Cost Optimization** — Two-sided cost function translating absence probability into optimal reserve staffing recommendations with dollar output. Threshold lowered to 0.35 based on cost asymmetry ($40K cancellation vs $500 false alarm).

---

## Results

| Model | CV AUC (5-fold) | Test AUC | Recall @0.35 | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.851 ± 0.028 | 0.859 | 0.912 | 0.755 |
| Random Forest | 0.854 ± 0.028 | 0.854 | — | 0.748 |
| **XGBoost ★ Winner** | **0.861 ± 0.025** | **0.869** | **0.912** | **0.767** |

**SSS SHAP Rank:** #4 of 35 features (validates original contribution)  
**Estimated annual net benefit:** $1.3M across DFW + CLT (industry cost estimates)

---

## Datasets

### Dataset 1 — UCI Absenteeism at Work
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/445/absenteeism+at+work)
- **License:** CC BY 4.0 — free for any use
- **Records:** 740 employee absence events, 21 features, 0 missing values
- **Citation:** Martiniano & Ferreira (2012), DOI: 10.24432/C5X882

```python
pip install ucimlrepo
from ucimlrepo import fetch_ucirepo
dataset = fetch_ucirepo(id=445)
```

### Dataset 2 — BTS On-Time Performance (American Airlines)
- **Source:** [Bureau of Transportation Statistics](https://transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ)
- **License:** Public domain — U.S. Federal Government
- **Records:** 820,876 real AA flights (DFW + CLT, 2022–2024)
- **Access:** Programmatic download via PREZIP endpoint (no API key required)

---

## Quick Start

```bash
# 1. Clone repo
git clone https://github.com/[yourname]/AAdvantage-Crew-Absence-Prediction
cd AAdvantage-Crew-Absence-Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline
python notebooks/01_data_collection.py      # Downloads data (~10 min for BTS)
python notebooks/02_cleaning_eda.py         # Cleans data, builds SSS
python notebooks/03_feature_engineering.py  # Engineers 35 features
python notebooks/04_modeling.py             # Trains 3 models, SHAP analysis
python notebooks/05_cost_optimization.py    # Reserve staffing optimization
```

---

## Repository Structure

```
AAdvantage-Crew-Absence-Prediction/
├── data/                          ← Cached data files (not in git)
│   ├── bts_aa_dfw_clt_2022_2024.parquet
│   ├── hub_day_sss.parquet        ← Schedule Stress Score by hub-day
│   ├── X_train_final.parquet
│   └── X_test_final.parquet
├── notebooks/
│   ├── 01_data_collection.py
│   ├── 02_cleaning_eda.py
│   ├── 03_feature_engineering.py
│   ├── 04_modeling.py
│   └── 05_cost_optimization.py
├── outputs/                       ← All 12 project charts (PNG)
├── docs/
│   └── AA_Crew_Absence_Prediction_Documentation.docx
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Research Foundation

| Paper | Key Contribution to This Project |
|---|---|
| Homaie Shandizi (2014) — Polytechnique Montreal | Primary LR methodology reference |
| Springer Nature (2024) — Dutch airline | SSS weight derivation (effect sizes) |
| Atkin (2019) — ResearchGate | Cost optimization framework |
| MDPI Applied Sciences (2024) | 3-model comparison structure, SMOTE |
| Transportation Science, INFORMS (2025) | Business case validation |

---

## Limitations

- UCI data is from a Brazilian courier company (not airline crew) — proxy dataset
- 740 rows = small training set; moderate overfitting controlled by regularization
- SSS merged by day-of-week + month proxy (no individual employee-flight linkage)
- Cost estimates use industry benchmarks, not real AA internal P&L data
- Summer shows higher absence than Winter (Southern Hemisphere dataset — documented)
- CLT shows higher SSS than DFW (long-haul concentration — analytical finding)

---

## Model Documentation Links

- [Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) — scikit-learn
- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#random-forests) — scikit-learn
- [XGBoost](https://xgboost.readthedocs.io/en/latest/tutorials/model.html) — Official Docs
- [SHAP](https://shap.readthedocs.io/en/latest/) — Explainability framework
- [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) — imbalanced-learn

---

## Resume Bullet

*"Built end-to-end crew absence prediction system replicating American Airlines' Crew Strategy & Intelligence workflow; trained Logistic Regression, Random Forest, and XGBoost on 820,876 real BTS flights and 740 UCI absence records; XGBoost achieved AUC 0.87 and Recall 0.91 at business-optimized threshold 0.35 (5-fold CV AUC 0.86 ± 0.03); engineered Schedule Stress Score from BTS schedule data — ranked #4 of 35 features by SHAP; reserve optimization layer estimated $1.3M annual net benefit across DFW and CLT."*

---

*Author: Akash Bhupesh Singh | MS Business Analytics, Iowa State University (May 2025) | May 2026*

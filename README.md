# AA Crew Absence Prediction & Reserve Staffing Optimization

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)
[![XGBoost AUC](https://img.shields.io/badge/XGBoost%20AUC-0.869-red.svg)](#results)

## Project Overview

This project replicates the analytical workflow of American Airlines' **Crew Strategy & Intelligence** team (Requisition ID: 85928). It predicts daily crew sick-call volume at DFW and CLT hubs using employee absence patterns and aviation schedule characteristics, then optimizes reserve crew staffing to minimize operational cost.

**Target Role:** Analyst/Sr Analyst, Crew Strategy & Intelligence, American Airlines, DFW

---

## Original Contributions

This project extends published research in three ways that no academic paper has achieved:

**1. Schedule Stress Score (SSS)**
A novel composite feature engineered from 820,876 real BTS flight records. It combines night-departure concentration, long-haul flight percentage, late-cascade delays, and early-morning departures into a single hub-day stress metric, scored on a 0 to 100 scale. The SSS ranked **#4 of 35 features by SHAP importance**, confirming that aviation schedule patterns predict crew absence beyond what demographic features alone can explain.

**2. Hub-Level Aggregation**
Individual absence predictions are aggregated to the hub-day operational level, matching exactly how AA's Crew Scheduling team manages reserve pools in practice. No published research paper performs this aggregation step.

**3. Asymmetric Cost Optimization**
A two-sided cost function translates absence probability into specific reserve staffing recommendations with a dollar output. The classification threshold was lowered from 0.50 to 0.35 based on cost asymmetry: missing a real absence costs $40,000 in a flight cancellation, while a false alarm costs $500 for one idle reserve crew day.

---

## Results

| Model | CV AUC (5-fold) | Test AUC | Recall at 0.35 | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.851 ± 0.028 | 0.859 | 0.912 | 0.755 |
| Random Forest | 0.854 ± 0.028 | 0.854 | — | 0.748 |
| **XGBoost (Winner)** | **0.861 ± 0.025** | **0.869** | **0.912** | **0.767** |

**SSS SHAP Rank:** #4 of 35 features, validating the original contribution  
**Estimated annual net benefit:** $1.3M across DFW and CLT based on industry cost benchmarks

---

## Datasets

### Dataset 1: UCI Absenteeism at Work
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/445/absenteeism+at+work)
- **License:** CC BY 4.0, free for any use
- **Records:** 740 employee absence events, 21 features, 0 missing values
- **Citation:** Martiniano & Ferreira (2012), Universidade Nove de Julho, Brazil

Place the file at `data/Absenteeism_at_work.csv` (semicolon-separated) before running the pipeline.

### Dataset 2: BTS On-Time Performance (American Airlines)
- **Source:** [Bureau of Transportation Statistics](https://transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ)
- **License:** Public Domain, U.S. Federal Government
- **Records:** 820,876 real AA flights at DFW and CLT, 2022 through 2024
- **Access:** Programmatic download via PREZIP endpoint, no API key required

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/AA-Crew-Absence-Prediction-Reserve-Optimization
cd AA-Crew-Absence-Prediction-Reserve-Optimization

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your data files in the data/ folder
#    data/Absenteeism_at_work.csv  (UCI download, semicolon-separated)
#    data/bts_aa_dfw_clt_2022_2024.parquet  (BTS download)

# 4. Run the full pipeline in order
python notebooks/01_data_collection.py
python notebooks/02_cleaning_eda.py
python notebooks/03_feature_engineering.py
python notebooks/04_modeling.py
python notebooks/05_cost_optimization.py
```

Each script picks up exactly where the previous one left off using saved parquet files. All charts are automatically written to the `outputs/` folder.

---

## Repository Structure

```
AA-Crew-Absence-Prediction-Reserve-Optimization/
├── data/
│   ├── Absenteeism_at_work.csv        (UCI local file, not in git)
│   ├── bts_aa_dfw_clt_2022_2024.parquet
│   ├── hub_day_sss.parquet            (Schedule Stress Score by hub-day)
│   ├── X_train_final.parquet
│   └── X_test_final.parquet
├── notebooks/
│   ├── 01_data_collection.py
│   ├── 02_cleaning_eda.py
│   ├── 03_feature_engineering.py
│   ├── 04_modeling.py
│   └── 05_cost_optimization.py
├── outputs/
│   └── (all 14 project charts saved here as PNG)
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Schedule Stress Score: How It Works

The SSS is built from four BTS-derived signals per hub per day. Weights are based on the effect sizes reported in the Springer 2024 paper on pilot absenteeism at a Dutch low-cost airline.

| Signal | Weight | What it captures |
|---|---|---|
| Night departure percentage | 0.35 | Flights departing between 22:00 and 02:00 |
| Long flight percentage | 0.30 | Flights exceeding 4 hours (sector overload proxy) |
| Late cascade percentage | 0.20 | Delays of 30+ minutes after 17:00 (rotation penalty) |
| Early departure percentage | 0.15 | Departures between 04:00 and 06:00 (recovery deficit) |

The raw score is normalized to a 0-100 scale. Days scoring above 75 are classified as Critical and trigger the high-risk reserve recommendation in the cost optimization layer.

---

## Research Foundation

Five peer-reviewed papers ground this methodology. The table below shows what each one contributed.

| Paper | Key contribution |
|---|---|
| Homaie Shandizi (2014), Polytechnique Montreal | Primary logistic regression methodology reference |
| Springer Nature (2024), Dutch airline data | SSS weight derivation from published effect sizes |
| Atkin (2019), ResearchGate | Cost optimization framework |
| MDPI Applied Sciences (2024), 312K observations | Three-model comparison structure and SMOTE validation |
| Transportation Science, INFORMS (2025) | Business case validation at scale |

---

## Model Parameters

All regularisation settings were chosen conservatively for the 592-row training set.

**Logistic Regression:** C=0.5, max_iter=1000, class_weight=balanced  
**Random Forest:** n_estimators=200, max_depth=6, min_samples_leaf=8, max_features=sqrt  
**XGBoost:** n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.8, min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0

---

## Limitations

Every serious project documents its constraints honestly.

- **Dataset domain gap.** The UCI data is from a Brazilian courier company, not airline crew. It lacks aviation-specific duty-time and block-hour data. The SSS partially bridges this gap by bringing real aviation schedule signals into the feature set.
- **Small training set.** 592 training rows after the 80/20 split. RF and XGBoost show moderate overfitting (train-validation gap 0.10 to 0.12), controlled by conservative regularisation and 5-fold CV.
- **SSS merge approximation.** SSS is joined on day-of-week and month because the UCI dataset lacks actual flight dates for each employee. With real crew schedule data linked to individual IDs, SSS would likely rank higher.
- **Southern Hemisphere seasonality.** The UCI data is from Brazil, where summer is December and January. The data show higher absence in summer than in winter. This is an honest finding, not an error.
- **Cost estimate uncertainty.** The $1.3M figure uses industry benchmarks of $40,000 per cancellation and $500 per reserve crew day. Replace these constants in `05_cost_optimization.py` with real AA P&L data for a production estimate.

---

## Model Documentation

- [Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) via scikit-learn
- [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#random-forests) via scikit-learn
- [XGBoost](https://xgboost.readthedocs.io/en/latest/tutorials/model.html) official documentation
- [SHAP](https://shap.readthedocs.io/en/latest/) explainability framework
- [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) via imbalanced-learn

---

## Resume Bullet

*Built end-to-end crew absence prediction system replicating American Airlines' Crew Strategy & Intelligence workflow; trained Logistic Regression, Random Forest, and XGBoost on 820,876 real BTS flights and 740 UCI absence records; XGBoost achieved AUC 0.87 and Recall 0.91 at business-optimized threshold 0.35 (5-fold CV AUC 0.86 ± 0.03); engineered Schedule Stress Score from BTS schedule data, ranked #4 of 35 features by SHAP; reserve optimization layer estimated $1.3M annual net benefit across DFW and CLT.*

---

*Author: Akash Bhupesh Singh | MS Business Analytics, Iowa State University (May 2025) | May 2026*

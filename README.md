# Heart‑Failure Patient‑Outcome Prediction

An end‑to‑end **machine‑learning and survival‑analysis** pipeline that:

1. **Ingests** the UCI *Heart Failure Clinical Records* dataset programmatically  
2. Performs leakage‑free **pre‑processing and feature engineering**  
3. Leverages **nested cross‑validation + Optuna** for robust model selection  
4. Evaluates reliability with **calibration curves & statistical tests**  
5. Adds longitudinal **survival analysis** (Kaplan–Meier & CoxPH)  
6. Provides global / local **explainability** via SHAP & LIME  
7. Delivers results through a **Dash + Bootstrap** interactive dashboard

---

## Repository layout

```
.
├── notebooks/
│   ├─ Heart_Failure_Clinical_Records_Dataset_Exploratory_Data_Analysis_(EDA).ipynb        
│   └─ Heart_Failure_Clinical_Records_Dataset_Predictive_Modeling.ipynb     
├── src/                             
│   ├─ __init__.py
│   ├─ config.py           
│   ├─ data_loader.py      
│   ├─ preprocessing.py    
│   ├─ modeling.py          
│   ├─ evaluation.py        
│   ├─ survival_analysis.py
│   ├─ explainability.py   
│   ├─ dashboard.py      
│   └─ main.py              #
├── requirements.txt
└── README.md
```

---

## Quick‑start

```bash
# 1. Clone
git clone https://github.com/your‑handle/Patient‑Outcome‑Prediction.git
cd Patient‑Outcome‑Prediction

# 2. (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline & launch dashboard
python -m src.main
# → open http://127.0.0.1:8050  🌐
```

---

## Key design choices

| Layer | Rationale |
|-------|-----------|
| **Programmatic data pull** | `ucimlrepo` guarantees the latest dataset without bundling raw CSVs. |
| **Pipeline‑based preprocessing** | `ColumnTransformer` prevents data‑leakage and keeps notebooks tidy. |
| **Nested CV + Optuna** | Outer 5‑fold measures generalisation; inner 3‑fold tunes hyper‑parameters via TPE. |
| **Calibration** | `CalibratedClassifierCV` + Brier Score ensure probabilities are decision‑grade. |
| **Survival modelling** | `lifelines` extends analysis beyond binary labels into time‑to‑event outcomes. |
| **Explainability** | SHAP (global) + LIME (local) build clinician trust and case‑level insight. |
| **Dash UI** | All‑Python, lightweight, easily deployable, Bootstrap‑styled for a professional look. |

---

## Reproducing study metrics

```bash
# Re‑run Optuna with 50 trials
python -m src.modeling --trials 50
# Evaluate on hold‑out set
python -m src.evaluation
```
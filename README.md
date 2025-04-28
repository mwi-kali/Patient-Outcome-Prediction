# Heart-Failure Patient Outcome Prediction

An end-to-end **machine-learning & survival-analysis** pipeline for predicting outcomes in heart-failure patients. 

---

## Repository Structure

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

## Quick Start

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-handle/Patient-Outcome-Prediction.git
   cd Patient-Outcome-Prediction
   ```

2. **(Optional) Set up a virtual environment**  
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the full pipeline & launch the dashboard**  
   ```bash
   python -m src.main
   ```
   Then open [http://127.0.0.1:8050](http://127.0.0.1:8050) in your browser.

---


## Reproducing Experiments

```bash
# 1. Re-run hyperparameter optimization (50 Optuna trials)
python -m src.modeling --trials 50

# 2. Evaluate final models on a held-out test set
python -m src.evaluation
```

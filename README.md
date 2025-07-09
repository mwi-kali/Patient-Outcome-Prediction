# Patient Outcome Prediction

An end-to-end **machine learning** and **survival analysis** pipeline for predicting clinical outcomes in heart failure patients using the UCI Heart Failure Clinical Records dataset.

---


## 📁 Project Structure

```bash
Patient-Outcome-Prediction/
├── data/
│   ├── raw/                          
│   └── processed/                 
│
├── models/                         
│
├── notebooks/                     
│   ├── Heart_Failure_Clinical_Records_Dataset_Exploratory_Data_Analysis_(EDA).ipynb
│   ├── Heart_Failure_Clinical_Records_Dataset_Feature_Engineering.ipynb
│   ├── Heart_Failure_Clinical_Records_Dataset_Hyperparameter_Optimization.ipynb
│   ├── Heart_Failure_Clinical_Records_Dataset_Model_Evaluation_and_Interpretation.ipynb
│   ├── Heart_Failure_Clinical_Records_Dataset_Cox_PH_and_Survival_Modeling.ipynb
│   └── Heart_Failure_Clinical_Records_Dataset_Results_Summary.ipynb
│
├── reports/                 
│
├── requirements.txt         
└── README.md       
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/mwi-kali/Patient-Outcome-Prediction.git
cd Patient-Outcome-Prediction
```

### 2. (Optional) Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter

```bash
jupyter lab                        # Or jupyter notebook
```

### 5. Execute the Notebooks

Run each notebook sequentially from **EDA → Feature Engineering → Modeling → Evaluation → Survival Analysis → Reporting**.


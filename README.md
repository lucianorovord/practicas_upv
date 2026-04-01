# 📊 Data Analysis, Machine Learning & Python Portfolio

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.x-013243?style=flat&logo=numpy&logoColor=white)

---

## 🚀 Overview

This repository documents my practical learning journey in **Python**, **Data Analysis** and **Machine Learning**, developed during my internship at **UPV Gandía** under the supervision of a PhD researcher in **Cybersecurity**.

The work covers the full **ML pipeline** — from raw data exploration to training and evaluating neural network models — applied to real-world datasets.

---

## 📂 Project Structure

```
PRACTICAS_UPV/
│
├── 📁 Heart_Failure_Prediction/        ← Main ML Project
│   ├── 📁 heart/
│   │   └── heart.csv                   ← Raw dataset (Kaggle)
│   ├── 📁 notebook/
│   │   └── Analisis_datos.ipynb        ← EDA + Preprocessing notebook
│   ├── hfp_pipeline.py                 ← Full ML pipeline script
│   ├── ml_pipeline.py                  ← Clean pipeline version
│   ├── xTrain.csv                      ← Preprocessed training set (70%)
│   ├── xVal.csv                        ← Preprocessed validation set (15%)
│   ├── xTest.csv                       ← Preprocessed test set (15%)
│   └── requirements.txt
│
├── 📁 Jupyter/
│   ├── 📁 Numpy/
│   │   └── numpy_practica.ipynb
│   └── 📁 Pandas/
│       └── pandas_practica.ipynb
│
├── 📁 Python/
│   ├── 📁 Ejercicios_Condicionales/
│   │   ├── Condicionales.py
│   │   └── Condicionales2.py
│   ├── BucleFor.py
│   ├── BucleWhile.py
│   └── AprendizajePY.py
│
└── README.md
```

---

## 🧠 Main Project — Heart Failure Prediction

**Dataset:** [Heart Failure Prediction — Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)  
**Goal:** Binary classification — predict whether a patient has heart disease (0 / 1)  
**Model:** MLP Classifier (Multi-Layer Perceptron)  
**Best result:** ~87% accuracy on validation set

### ML Pipeline followed

```
EDA → Pre-Processing → Data Split → Scaling → Training + Validation → Test
 1          2               3           4               5                6
```

#### 1. EDA — Exploratory Data Analysis
- Inspected dataset shape, data types and null values (`918 rows × 12 columns`)
- Identified statistically impossible values: **172 rows** with `Cholesterol = 0` and **1 row** with `RestingBP = 0`
- Computed descriptive statistics per column (min, max, mean, std)

#### 2. Pre-Processing

**2.1 Fixing Problems**
- Replaced zero values in `Cholesterol` with column mean excluding zeros → `244 mg/dL`
- Replaced zero value in `RestingBP` with column mean excluding zeros → `132 mmHg`

**2.2 Encoding**
- Converted categorical string columns to integers using label encoding:

| Column | Mapping |
|--------|---------|
| Sex | M→1, F→0 |
| ChestPainType | ATA→0, NAP→1, ASY→2, TA→3 |
| RestingECG | Normal→0, ST→1, LVH→2 |
| ExerciseAngina | N→0, Y→1 |
| ST_Slope | Flat→0, Up→1, Down→2 |

#### 3. Data Split — 70 / 15 / 15

| Set | Size | Purpose |
|-----|------|---------|
| Train | 642 rows (70%) | Model learns from these |
| Validation | 138 rows (15%) | Detects overfitting during training |
| Test | 138 rows (15%) | Final honest evaluation — used only once |

> **Key rule:** Validation is used to tune hyperparameters. Test is only touched once at the very end to avoid data leakage.

#### 4. Scaling — StandardScaler
- Applied `fit_transform()` on training set only
- Applied `transform()` on validation and test sets using training statistics
- Result: all columns normalized to mean ≈ 0, std ≈ 1

> **Why fit only on train?** Fitting on val/test would leak future information into the model — a practice known as **data leakage**.

#### 5. Training + Validation
- Architecture: `MLPClassifier` with hidden layers `(90, 90)`
- Epochs: `80` with early stopping (`n_iter_no_change=50`)
- Monitored three scenarios during experimentation:

| Scenario | Architecture | Epochs | Result |
|----------|-------------|--------|--------|
| Underfitting | (4, 4) | 10 | Low accuracy on both train and val |
| **Generalization** | **(90, 90)** | **80** | **Good accuracy on both ✅** |
| Overfitting | (600, 600) | 500 | High train accuracy, low val accuracy |

#### 6. Test
- Final evaluation on unseen test data
- Results compared across Train / Validation / Test sets

---

## 🐍 Python Fundamentals

| Topic | Files |
|-------|-------|
| Conditional structures | `Condicionales.py`, `Condicionales2.py` |
| For loops | `BucleFor.py` |
| While loops | `BucleWhile.py` |
| General Python practice | `AprendizajePY.py` |

---

## 🔢 NumPy & Pandas

- Array creation, vectorized operations and mathematical computations
- DataFrame manipulation, filtering, cleaning and basic analysis workflows

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11 | Main language |
| TensorFlow | 2.21 | Deep learning framework |
| scikit-learn | 1.8 | ML algorithms + preprocessing |
| Pandas | 3.x | Data manipulation |
| NumPy | 2.x | Numerical computing |
| Matplotlib | 3.x | Data visualization |
| Jupyter Notebook | — | Interactive analysis |

---

## 🎯 Next Steps

- [ ] Explore additional evaluation metrics — Precision, Recall, F1-Score, Confusion Matrix
- [ ] Implement **GridSearchCV** to automate hyperparameter tuning
- [ ] Start anomaly detection project on network traffic data (Cybersecurity)
- [ ] Learn SQL for data querying
- [ ] Build interactive dashboards with Power BI or Matplotlib

---

## 👨‍💻 Author

**Luciano Rovere Ordoñez**  
Junior Developer | Python · Data Analysis · Machine Learning  
📍 Valencia, Spain  

---

*Developed during internship at UPV Gandía · 2025*
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

The work covers the full **ML pipeline** — from raw data exploration to training and evaluating neural network models — applied to two real-world datasets: a medical classification problem and a **network intrusion detection system**.

---

## 📂 Project Structure

```
PRACTICAS_UPV/
│
├── 📁 Heart_Failure_Prediction/        ← ML Project 1
│   ├── 📁 heart/
│   │   └── heart.csv                   ← Raw dataset (Kaggle)
│   ├── 📁 notebook/
│   │   └── Analisis_datos.ipynb        ← EDA + Preprocessing notebook
│   ├── hfp_pipeline.py                 ← Full ML pipeline script
│   └── ml_pipeline.py                  ← Clean pipeline version
│
├── 📁 CyberSecurity/                   ← ML Project 2 (Final)
│   ├── 📁 notebook/
│   │   └── cibseg_eda.ipynb            ← EDA + Preprocessing notebook
│   └── cibseg_pipeline.py              ← Full ML pipeline script
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

> ⚠️ **Note:** Raw and preprocessed CSV files are not included in this repository due to file size constraints (some exceed 500MB). See the dataset links below to download them.

---

## 🧠 Project 1 — Heart Failure Prediction

**Dataset:** [Heart Failure Prediction — Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)  
**Goal:** Binary classification — predict whether a patient has heart disease (0 / 1)  
**Model:** MLP Classifier (Multi-Layer Perceptron)  
**Best result:** ~87% accuracy on validation set

### ML Pipeline

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

#### 6. Metrics evaluated
- Accuracy, Precision, Recall
- Confusion Matrix (TP, FP, TN, FN) across Train / Validation / Test

---

## 🔐 Project 2 — Network Intrusion Detection (Cybersecurity)

**Dataset:** [CNS2022 Network Intrusion Dataset — Distrinet Research](https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/)  
**Goal:** Binary classification — detect whether network traffic is benign or an attack (0 / 1)  
**Model:** MLP Classifier (Multi-Layer Perceptron)  
**Context:** Final project supervised by a PhD researcher in Cybersecurity at UPV Gandía

### Dataset characteristics

| Property | Value |
|----------|-------|
| Total records | 496.641 |
| Total columns (original) | 90+ |
| Columns used | 81 |
| BENIGN traffic | 319.120 (64%) |
| Attack traffic | 177.521 (36%) |
| Attack types | DoS Slowloris, DoS Slowhttptest, DoS Hulk, DoS GoldenEye, Heartbleed (+ Attempted variants) |

### ML Pipeline

```
EDA → Pre-Processing → Data Split → Scaling → Training + Validation → Test
 1          2               3           4               5                6
```

#### 1. EDA — Exploratory Data Analysis
- Inspected dataset shape: `496.641 rows × 82 columns`
- Verified: **0 null values**, **0 infinite values** detected after column filtering
- Checked for impossible negative values — only `ICMP Code` and `ICMP Type` had `-1` (valid in networking — means no ICMP protocol used)
- Verified all columns had `std > 0` — no zero-variance columns to remove
- Analyzed class balance: 64% BENIGN vs 36% attacks — **moderately imbalanced**

> **Key insight:** Unlike Heart Failure, zero values in network traffic are **valid** (e.g. `Flow Bytes/s = 0` means no data was transferred). They were NOT replaced.

#### 2. Pre-Processing

**2.1 Column Selection**
- Dropped non-informative columns: `id`, `Flow ID`, `Src IP`, `Src Port`, `Dst IP`, `Dst Port`, `Protocol`, `Timestamp`, `Attempted Category`

**2.2 Encoding**
- `Label` column encoded using `np.where()` — any value that is not `BENIGN` becomes an attack:

```python
df['Label'] = np.where(df['Label'].str.strip() == 'BENIGN', 0, 1)
```

> Used `str.strip()` to remove leading/trailing whitespace that caused encoding errors.

**2.3 Fixing Infinities**
- Infinite values appeared after the split (generated during feature computation — e.g. division by zero in `Flow Bytes/s`)
- Replaced with NaN and imputed using `x_train.mean()` for all three sets to avoid data leakage

#### 3. Data Split — 70 / 15 / 15

| Set | Size | Purpose |
|-----|------|---------|
| Train | 347.648 rows (70%) | Model learns from these |
| Validation | 74.496 rows (15%) | Hyperparameter tuning |
| Test | 74.497 rows (15%) | Final honest evaluation |

#### 4. Scaling — StandardScaler
- Same approach as Project 1 — `fit_transform` only on train, `transform` on val and test

#### 5. Training + Validation
- Architecture: `MLPClassifier` with hidden layers `(32, 32)`
- `verbose=True` enabled to monitor training progress on large dataset
- Evaluated Accuracy, Precision and Recall across Train / Validation / Test

#### 6. Metrics evaluated
- **Accuracy** — overall correctness
- **Precision** — minimizes false alarms (important in security to avoid alert fatigue)
- **Recall** — minimizes missed attacks (critical — missing a real attack is worse than a false alarm)
- **Confusion Matrix** — visualized for Train, Validation and Test

> **Why Recall matters most here:** In cybersecurity, classifying a real attack as benign (False Negative) is far more dangerous than triggering a false alarm (False Positive).

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

- [x] Implement Confusion Matrix, Precision and Recall metrics
- [ ] Implement **GridSearchCV** to automate hyperparameter tuning
- [ ] Learn SQL for data querying
- [ ] Build interactive dashboards with Power BI or Matplotlib

---

## 👨‍💻 Author

**Luciano Rovere Ordoñez**  
Junior Developer | Python · Data Analysis · Machine Learning  
📍 Valencia, Spain

---

*Developed during internship at UPV Gandía · 2025*
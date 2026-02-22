# Credit Card Fraud Detection — MLOps Lab 1

## Project Overview

This project implements a machine learning pipeline for detecting fraudulent credit card transactions using ensemble methods and MLflow experiment tracking.

The dataset is highly imbalanced (fraud cases represent less than 1% of all transactions), which makes this task particularly suitable for evaluation using Precision-Recall metrics rather than accuracy.

The goal of the project is to demonstrate:

* structured exploratory data analysis,
* model training and hyperparameter tuning,
* handling of severe class imbalance,
* experiment tracking with MLflow,
* reproducibility and clean project structure.

## Dataset

Source: Kaggle — Credit Card Fraud Detection

Target column: `Class`

* 0 — legitimate transaction
* 1 — fraudulent transaction

The dataset contains anonymized numerical features along with `Time` and `Amount`. The class distribution is extremely imbalanced.

## Project Structure

```
mlops_lab_1/
│
├── src/
│   └── train.py
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

* `01_eda.ipynb` — exploratory data analysis only.
* `train.py` — training pipeline, evaluation, MLflow logging.
* MLflow artifacts and virtual environment are excluded from version control.

## Exploratory Data Analysis

The EDA notebook includes:

* dataset inspection (`shape`, `info`, missing values),
* class imbalance visualization,
* distribution comparison of transaction amount across classes,
* time-based fraud analysis,
* top correlations with the target variable.

EDA does not include model training, SMOTE, SHAP, or threshold tuning. These steps are implemented in the training pipeline.

## Models Implemented

The following models were tested:

### Random Forest

Used as the primary baseline model.
Hyperparameter tuning was performed over `max_depth`.

### XGBoost

Gradient boosting model evaluated with and without SMOTE.

### LightGBM

Efficient gradient boosting implementation evaluated under the same framework.

## Handling Class Imbalance

Three approaches were evaluated:

* `none` — original class distribution,
* `SMOTE` — synthetic minority oversampling applied to the training set only,
* `class_weight` — weighted loss function.

SMOTE is applied exclusively to the training data in order to prevent data leakage and ensure that the test set reflects the true real-world distribution.

## Evaluation Metrics

Due to extreme class imbalance, the following metrics are emphasized:

* Precision
* Recall
* F1-score
* PR-AUC (primary metric)
* ROC-AUC

PR-AUC is considered the most informative metric for this task.

## Running the Project

Activate the virtual environment:

```
source venv/bin/activate
```

Run Random Forest:

```
python src/train.py --model rf --imbalance none --max_depth 9
```

Run XGBoost:

```
python src/train.py --model xgb --imbalance none --max_depth 6 --n_estimators 300
```

Run LightGBM:

```
python src/train.py --model lgbm --imbalance none --max_depth 6 --n_estimators 300
```

## MLflow Experiment Tracking

Start MLflow UI:

```
mlflow ui
```

Open in browser:

```
http://127.0.0.1:5000
```

Each run logs:

* hyperparameters,
* test metrics,
* trained model,
* confusion matrix,
* feature importance,
* SHAP visualizations (optional).

## Key Findings

* Random Forest with tuned depth provides a strong and stable baseline.
* SMOTE increases recall but may reduce precision.
* XGBoost achieves strong PR-AUC performance under SMOTE.
* LightGBM performance depends heavily on hyperparameter tuning.

## Reproducibility

All experiments:

* use fixed random seeds,
* are tracked via MLflow,
* are version-controlled via Git.


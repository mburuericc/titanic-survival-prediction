# Titanic Survival Prediction

A machine learning project that predicts passenger survival on the Titanic, built for the [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/overview) competition.

**Kaggle leaderboard score:** 0.76315 (accuracy)

## Overview

The project walks through a full classification workflow: data checks, preprocessing pipelines, comparison of five models with cross-validation, hyperparameter tuning of a Random Forest with three search strategies, and generation of a Kaggle submission file from the saved model.

## Project structure

```
titanic-survival-prediction/
├── data/
│   ├── train.csv               # Kaggle training data (891 passengers)
│   ├── test.csv                # Kaggle test data (418 passengers, no labels)
│   └── gender_submission.csv   # Kaggle's sample submission
├── notebook/
│   └── titanic.ipynb           # Full workflow: EDA, modelling, tuning, evaluation
├── Models/
│   └── titanic_model.joblib    # Trained preprocessing + Random Forest pipeline
├── Scripts/
│   └── predict.py              # Loads the model and writes a submission file
├── Submissions/
│   └── submission.csv          # Predictions for test.csv
├── requirements.txt
└── README.md
```

## Dataset

- Source: Kaggle Titanic competition (`train.csv`)
- 891 passengers, 12 original columns
- Target column: `Survived` (0 = did not survive, 1 = survived)
- Features used: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`
- Dropped: `Cabin` (too many missing values), plus `PassengerId`, `Name` and `Ticket`

### Missing values

| Column | % Missing |
|---|---|
| Cabin | 77.1% |
| Age | 19.9% |
| Embarked | 0.2% |

`Cabin` was dropped because of excessive missingness. `Age` and `Embarked` are imputed inside the preprocessing pipeline.

## Workflow

1. **Data checks:** duplicates, shape, missing values, data types, summary statistics
2. **EDA:** target distribution and feature types (numerical vs categorical)
3. **Preprocessing pipeline** (combined with `ColumnTransformer`)
   - Numerical (`Age`, `SibSp`, `Parch`, `Fare`): median imputation, then standard scaling
   - Categorical (`Pclass`, `Sex`, `Embarked`): most-frequent imputation, then one-hot encoding
4. **Model comparison:** 5 candidate models evaluated with 5-fold stratified cross-validation on an 80/20 stratified train/test split
5. **Hyperparameter tuning** of the Random Forest with Grid Search, Randomized Search and Optuna
6. **Final model training, evaluation and saving** (`joblib`)
7. **Kaggle submission** generated with `Scripts/predict.py`

## Results

### Model comparison (5-fold CV on the training split)

| Model | Mean F1 | Mean ROC-AUC |
|---|---|---|
| Logistic Regression | 0.730 | 0.853 |
| K-Nearest Neighbors | 0.719 | 0.848 |
| Random Forest | 0.735 | 0.865 |
| SVM (RBF) | 0.745 | 0.851 |
| Gradient Boosting | 0.705 | 0.849 |

Random Forest had the highest ROC-AUC, so it was chosen for hyperparameter tuning.

### Hyperparameter tuning (Random Forest, CV F1)

| Method | Best F1 | n_estimators | max_depth | min_samples_split | min_samples_leaf |
|---|---|---|---|---|---|
| Grid Search | 0.7598 | 100 | 10 | 5 | 1 |
| Randomized Search | 0.7476 | 200 | None | 3 | 3 |
| Optuna | 0.7578 | 292 | 19 | 7 | 2 |

All three methods landed at a similar score (about 0.75 to 0.76 F1), so the choice of search strategy made little difference.

### Final model (held-out 20% test split, 179 passengers)

The final Random Forest uses `n_estimators=501`, `max_depth=16`, `min_samples_split=6` and `min_samples_leaf=1`.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Did not survive (0) | 0.81 | 0.91 | 0.85 |
| Survived (1) | 0.82 | 0.65 | 0.73 |

Overall accuracy on the held-out split: **0.81**. The model is better at identifying passengers who did not survive than those who did.

## Getting started

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
git clone https://github.com/mburuericc/titanic-survival-prediction.git
cd titanic-survival-prediction
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter notebook notebook/titanic.ipynb
```

### Generate predictions

The script uses relative paths, so run it from inside the `Scripts` folder:

```bash
cd Scripts
python predict.py
```

This loads `Models/titanic_model.joblib`, predicts survival for `data/test.csv`, and writes `Submissions/submission.csv` in the format Kaggle expects (`PassengerId`, `Survived`).

### Predict for a single passenger

```python
import joblib
import pandas as pd

model = joblib.load("Models/titanic_model.joblib")

passenger = pd.DataFrame({
    "Pclass": [1],
    "Sex": ["female"],
    "Age": [12],
    "SibSp": [2],
    "Parch": [1],
    "Fare": [54.72],
    "Embarked": ["S"],
})

print(model.predict(passenger))              # [1]  -> survived
print(model.predict_proba(passenger)[:, 1])  # probability of survival
```

## Tech stack

- Python, pandas, NumPy
- scikit-learn (pipelines, preprocessing, model selection)
- XGBoost
- Optuna
- matplotlib, seaborn

## Future improvements

- Feature engineering: extract titles from `Name`, build a family-size feature from `SibSp` + `Parch`, and use the cabin deck where available
- Ensemble or stacking of the top-performing models
- Add `XGBClassifier` to the model comparison (it is imported but not yet evaluated)
- Tune the decision threshold to improve recall on survivors

## Author

Eric Mburu - [GitHub](https://github.com/mburuericc)

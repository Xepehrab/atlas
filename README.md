# Allstate Claims Severity

Machine learning solution for the [Kaggle Allstate Claims Severity](https://www.kaggle.com/competitions/allstate-claims-severity) competition.

## Project Overview

This project predicts the **severity of insurance claims** (`loss`) using anonymized tabular data. Each row represents a claim with **116 categorical** and **14 continuous** features. The model is evaluated with **Mean Absolute Error (MAE)** on the original dollar amount of the claim.

The workflow includes:

- Exploratory data analysis in Jupyter
- Data preparation in a reusable Python script
- CatBoost regression with native categorical handling
- A LightGBM baseline explored in the notebook

## Problem Summary

| Item | Detail |
|------|--------|
| Task | Regression |
| Target | `loss` (claim amount) |
| Metric | MAE |
| Train rows | 188,318 |
| Test rows | 125,546 |
| Features | 130 (`cat1`–`cat116`, `cont1`–`cont14`) |

The target is **highly right-skewed** (skew ≈ 3.8), so models are trained on:

```python
log_loss = np.log1p(loss)
```

Predictions are converted back with `np.expm1()` before scoring or submission.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- LightGBM
- CatBoost
- ydata-profiling

## Project Structure

```
allstate-claims-severity/
├── prepare.py              # Data loading and feature preparation
├── main.py                 # CatBoost training and submission
├── jupyter.ipynb           # EDA, LightGBM baseline, experiments
├── requirements.txt        # Python dependencies
├── train.csv               # Training data (not in repo)
├── test.csv                # Test data (not in repo)
├── sample_submission.csv   # Kaggle submission format
├── submission.csv          # Generated predictions
└── data_profile_report.html  # Optional EDA report
```

## Key EDA Findings

- No missing values in train or test
- Most categorical columns are low-cardinality; `cat116` has the highest cardinality (~326 levels)
- Some categorical levels appear in test but not in train (handled by CatBoost natively)
- Continuous features are already numeric and scaled roughly between 0 and 1
- `loss` mean ≈ 3037, median ≈ 2116, max ≈ 121,012

## Modeling Approach

### Data preparation (`prepare.py`)

- Load `train.csv` and `test.csv`
- Separate categorical and continuous features
- Check train/test category mismatches
- Create `log_loss = log1p(loss)`
- Return model-ready `X`, `y`, `X_test`, `test_ids`, and `cat_cols`

### Training (`main.py`)

- Split training data 80/20 for validation
- Train `CatBoostRegressor` on `log_loss`
- Pass categorical columns directly to CatBoost
- Evaluate validation MAE on the original `loss` scale
- Predict test claims and save `submission.csv`

### Notebook experiments (`jupyter.ipynb`)

- Target distribution and skew analysis
- Cardinality checks for categorical features
- Train vs test category mismatch report
- **LightGBM baseline** with `OrdinalEncoder` and 5-fold CV
- Optional CatBoost comparison

## Results

| Model | Validation / CV | Notes |
|-------|-----------------|-------|
| LightGBM | ~1148 OOF MAE | 5-fold CV, ordinal-encoded categoricals |
| CatBoost | ~1141 validation MAE | 80/20 split, raw categorical features |

These are strong first-pass results for this competition using a simple feature pipeline.

## Setup

1. Clone the repository

```bash
git clone <your-repo-url>
cd allstate-claims-severity
```

2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Download the competition data from Kaggle and place these files in the project folder:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

## Usage

Train CatBoost and generate a submission:

```bash
python main.py
```

For exploration and baseline models, open:

```bash
jupyter notebook jupyter.ipynb
```

## Submission Format

The output file must contain two columns:

```csv
id,loss
4,1528.90
6,2108.87
...
```

## Possible Improvements

- Target encoding for high-cardinality categoricals
- Feature interactions between top categorical variables
- Hyperparameter tuning for CatBoost / LightGBM
- Ensemble multiple models (LightGBM + CatBoost)
- Retrain final model on the full training set before submission

## Author

Sky

## License

This project is for educational and portfolio purposes. Dataset provided by Allstate via Kaggle.

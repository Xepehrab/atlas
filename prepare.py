import pandas as pd
import numpy as np


def prepare_data():
    """
    Load and prepare the Allstate Claims Severity dataset.

    Returns:
        X (DataFrame): Training features
        y (Series): Log-transformed target
        X_test (DataFrame): Test features
        test_ids (Series): Test IDs for submission
        cat_cols (list): Categorical feature names
    """

    # ==========================================================
    # Load Data
    # ==========================================================

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    print(f"Train shape: {train.shape}")
    print(f"Test shape : {test.shape}")

    # ==========================================================
    # Feature Lists
    # ==========================================================

    cont_cols = [c for c in train.columns if c.startswith("cont")]
    cat_cols = [c for c in train.columns if c.startswith("cat")]

    print(f"Continuous features : {len(cont_cols)}")
    print(f"Categorical features: {len(cat_cols)}")

    # ==========================================================
    # Check category mismatches
    # ==========================================================

    mismatch_report = []

    for col in cat_cols:

        train_vals = set(train[col].unique())
        test_vals = set(test[col].unique())

        only_in_test = test_vals - train_vals
        only_in_train = train_vals - test_vals

        if only_in_test or only_in_train:
            mismatch_report.append({
                "column": col,
                "only_in_test": len(only_in_test),
                "only_in_train": len(only_in_train)
            })

    print(f"Columns with category mismatches: {len(mismatch_report)}")

    # ==========================================================
    # Target Engineering
    # ==========================================================

    train["log_loss"] = np.log1p(train["loss"])

    # ==========================================================
    # Modeling Data
    # ==========================================================

    feature_cols = [
        c for c in train.columns
        if c not in ["id", "loss", "log_loss"]
    ]

    X = train[feature_cols].copy()
    y = train["log_loss"]

    X_test = test[feature_cols].copy()
    test_ids = test["id"]

    return X, y, X_test, test_ids, cat_cols
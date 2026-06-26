import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Import feature preparation from prepare.py
from prepare import prepare_data


print("Preparing data...")

# X = training features, y = log_loss target, X_test = test features
# test_ids = claim ids for submission, cat_cols = categorical column names
X, y, X_test, test_ids, cat_cols = prepare_data()

print("Creating train/validation split...")

# Hold out 20% of training data to estimate model performance
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training CatBoost...")

# CatBoost can handle categorical columns directly
model = CatBoostRegressor(
    iterations=500,          # number of boosting trees
    learning_rate=0.1,       # step size for each tree
    depth=6,                 # tree depth
    loss_function="RMSE",     # optimize RMSE on log_loss
    eval_metric="RMSE",       # metric shown during training
    random_seed=42,           # reproducibility
    verbose=100,              # print progress every 100 iterations
    thread_count=-1           # use all CPU cores
)

# Train on training split and monitor performance on validation split
model.fit(
    X_train,
    y_train,
    cat_features=cat_cols,           # tell CatBoost which columns are categorical
    eval_set=(X_valid, y_valid),       # validation set for early stopping / monitoring
    use_best_model=True              # keep the best iteration from eval_set
)

print("Evaluating model...")

# Predict on validation set (still in log scale)
pred_log = model.predict(X_valid)

# Convert predictions and true values back to original loss scale
# MAE is computed on real dollars, not log_loss
mae = mean_absolute_error(
    np.expm1(y_valid),
    np.expm1(pred_log)
)

print(f"Validation MAE: {mae:.4f}")

print("Predicting test set...")

# Predict test claims and convert back from log scale
test_pred_log = model.predict(X_test)
test_pred = np.expm1(test_pred_log)

# Build Kaggle submission file: id + predicted loss
submission = pd.DataFrame({
    "id": test_ids,
    "loss": test_pred
})

submission.to_csv("submission.csv", index=False)

print("Submission saved as submission.csv")
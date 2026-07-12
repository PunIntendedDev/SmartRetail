"""
Regression Module
-----------------
Trains and evaluates Linear Regression and MLPRegressor to predict customer future spending.
Uses customer features from Months 1-9 to forecast customer purchases in Months 10-12.
Compares models on the validation set, saves the best model, and outputs predictions,
metrics, and validation plots (Predicted vs Actual, Residuals).
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from typing import Tuple

from src.config import get_absolute_path, config_data
from utils.logging_utils import get_logger
from utils.plotting_utils import plot_regression_predicted_vs_actual, plot_regression_residuals

logger = get_logger(__name__)

def load_regression_data(split_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads raw scaled features and the continuous Future_Spend target for regression.

    Args:
        split_name: 'train', 'validation', or 'test'.

    Returns:
        X: Feature matrix.
        y: Future spend target series.
    """
    processed_dir = get_absolute_path("processed_data_dir")
    filename = f"{split_name}_raw_scaled.csv"
    filepath = os.path.join(processed_dir, filename)
    
    df = pd.read_csv(filepath)
    if "CustomerID" in df.columns:
        df = df.set_index("CustomerID")
        
    y = df["Future_Spend"]
    
    # Exclude targets from features to prevent leakage
    exclude_cols = ["High_Value_Customer", "Future_Spend"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    return X, y

def train_regression() -> None:
    """
    Executes the regression pipeline:
    - Loads Train, Val, and Test datasets.
    - Fits Linear Regression and MLPRegressor.
    - Evaluates performance using MSE, RMSE, and R2 on the validation set.
    - Selects the best performing model.
    - Evaluates the best model once on the test set.
    - Saves model binaries and regression diagnostics plots.
    """
    models_dir = get_absolute_path("models_dir")
    processed_dir = get_absolute_path("processed_data_dir")
    
    random_state = config_data["data_split"]["random_state"]
    mlp_hidden = tuple(config_data["regression"]["mlp_hidden_layers"])
    
    # Load splits
    X_train, y_train = load_regression_data("train")
    X_val, y_val = load_regression_data("validation")
    
    # Model 1: Linear Regression
    logger.info("Fitting Linear Regression model...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Save Model 1
    joblib.dump(lr, os.path.join(models_dir, "linear_regression.joblib"))
    
    # Evaluate Linear Regression
    val_preds_lr = lr.predict(X_val)
    val_mse_lr = mean_squared_error(y_val, val_preds_lr)
    val_r2_lr = r2_score(y_val, val_preds_lr)
    logger.info(f"Linear Regression Val R²: {val_r2_lr:.4f} | RMSE: {np.sqrt(val_mse_lr):.2f}")
    
    # Model 2: MLPRegressor
    logger.info("Fitting MLPRegressor model...")
    mlp = MLPRegressor(
        hidden_layer_sizes=mlp_hidden,
        activation="relu",
        random_state=random_state,
        max_iter=1000
    )
    mlp.fit(X_train, y_train)
    
    # Save Model 2
    joblib.dump(mlp, os.path.join(models_dir, "mlp_regressor.joblib"))
    
    # Evaluate MLPRegressor
    val_preds_mlp = mlp.predict(X_val)
    val_mse_mlp = mean_squared_error(y_val, val_preds_mlp)
    val_r2_mlp = r2_score(y_val, val_preds_mlp)
    logger.info(f"MLPRegressor Val R²: {val_r2_mlp:.4f} | RMSE: {np.sqrt(val_mse_mlp):.2f}")
    
    # Model Selection: Select the model with highest R² score on Validation Set
    best_model_name = "MLPRegressor" if val_r2_mlp > val_r2_lr else "Linear Regression"
    best_model = mlp if val_r2_mlp > val_r2_lr else lr
    
    logger.info("=" * 80)
    logger.info(f"BEST REGRESSOR SELECTED: {best_model_name}")
    logger.info(f"Validation R²: {max(val_r2_lr, val_r2_mlp):.4f}")
    logger.info("=" * 80)
    
    # Save the representative and selected best regressor
    joblib.dump(best_model, os.path.join(models_dir, "best_regressor.joblib"))
    
    # Evaluate once on Test Set
    X_test, y_test = load_regression_data("test")
    test_preds = best_model.predict(X_test)
    
    test_mse = mean_squared_error(y_test, test_preds)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, test_preds)
    
    logger.info("FINAL TEST SET PERFORMANCE (Best Model):")
    logger.info(f"  MSE:  {test_mse:.4f}")
    logger.info(f"  RMSE: {test_rmse:.4f}")
    logger.info(f"  R²:   {test_r2:.4f}")
    
    # Output metrics JSON
    results = {
        "validation_comparisons": {
            "Linear_Regression": {
                "mse": val_mse_lr,
                "rmse": np.sqrt(val_mse_lr),
                "r2_score": val_r2_lr
            },
            "MLP_Regressor": {
                "mse": val_mse_mlp,
                "rmse": np.sqrt(val_mse_mlp),
                "r2_score": val_r2_mlp
            }
        },
        "best_model_selected": best_model_name,
        "test_evaluation": {
            "mse": test_mse,
            "rmse": test_rmse,
            "r2_score": test_r2
        }
    }
    
    results_path = os.path.join(processed_dir, "regression_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Metrics saved to: {results_path}")
    
    # Save Predictions CSV
    predictions_df = pd.DataFrame({
        "CustomerID": X_test.index,
        "Actual_Future_Spend": y_test.values,
        "Predicted_Future_Spend": test_preds
    })
    predictions_path = os.path.join(processed_dir, "regression_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to: {predictions_path}")
    
    # Generate and Save Diagnostics plots on Test predictions
    logger.info("Generating regression diagnostic figures...")
    plot_regression_predicted_vs_actual(y_test.values, test_preds, best_model_name)
    plot_regression_residuals(y_test.values, test_preds, best_model_name)
    logger.info("Regression figures successfully saved under outputs/figures/.")

if __name__ == "__main__":
    train_regression()

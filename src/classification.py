"""
Classification Module
---------------------
Trains and evaluates machine learning classifiers (Logistic Regression and MLPClassifier)
across raw scaled, PCA-transformed, and LDA-projected feature spaces.
Performs automatic model selection on the validation set, saves the best model,
and outputs performance metrics and curves (Confusion Matrix, ROC, Precision-Recall).
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
)
from typing import Dict, Any, Tuple

from src.config import get_absolute_path, config_data
from utils.logging_utils import get_logger
from utils.plotting_utils import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

logger = get_logger(__name__)

def load_dataset_split(split_name: str, representation: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads a specific feature representation and split (train/validation/test).

    Args:
        split_name: 'train', 'validation', or 'test'.
        representation: 'raw_scaled', 'pca', or 'lda'.

    Returns:
        X: Feature DataFrame.
        y: Binary High-Value Customer labels.
        target_spend: Continuous Future_Spend values.
    """
    processed_dir = get_absolute_path("processed_data_dir")
    filename = f"{split_name}_{representation}.csv"
    filepath = os.path.join(processed_dir, filename)
    
    df = pd.read_csv(filepath)
    if "CustomerID" in df.columns:
        df = df.set_index("CustomerID")
        
    y = df["High_Value_Customer"].astype(int)
    target_spend = df["Future_Spend"]
    
    # Exclude targets from features to prevent data leakage
    exclude_cols = ["High_Value_Customer", "Future_Spend"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    return X, y, target_spend

def train_classification() -> None:
    """
    Executes the classification pipeline:
    - Loads Train, Val, and Test splits for Raw, PCA, and LDA.
    - Fits GridSearchCV for Logistic Regression.
    - Fits MLPClassifier.
    - Logs performance metrics on the validation set.
    - Selects the best overall model configuration.
    - Evaluates the chosen model on the test set.
    - Saves model binaries and plotting figures.
    """
    models_dir = get_absolute_path("models_dir")
    processed_dir = get_absolute_path("processed_data_dir")
    
    representations = ["raw_scaled", "pca", "lda"]
    cv_folds = config_data["classification"]["cv_folds"]
    lr_grid = config_data["classification"]["logistic_regression_grid"]
    mlp_hidden = tuple(config_data["classification"]["mlp_hidden_layers"])
    random_state = config_data["data_split"]["random_state"]
    
    results = {}
    fitted_models = {}
    val_probs = {}
    val_y_dict = {}
    
    # Fit and evaluate models across all 3 representations
    for rep in representations:
        logger.info(f"--- Training on '{rep}' representation ---")
        
        # Load datasets
        X_train, y_train, _ = load_dataset_split("train", rep)
        X_val, y_val, _ = load_dataset_split("validation", rep)
        
        # Model 1: Logistic Regression with GridSearchCV
        logger.info("Fitting Logistic Regression with GridSearchCV...")
        lr = LogisticRegression(random_state=random_state, max_iter=1000)
        grid_search = GridSearchCV(
            estimator=lr,
            param_grid=lr_grid,
            cv=cv_folds,
            scoring="f1",
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_lr = grid_search.best_estimator_
        
        # Save fit model
        lr_key = f"Logistic_Regression_{rep}"
        fitted_models[lr_key] = best_lr
        joblib.dump(best_lr, os.path.join(models_dir, f"logistic_regression_{rep}.joblib"))
        
        # Evaluate on Validation
        val_preds_lr = best_lr.predict(X_val)
        val_prob_lr = best_lr.predict_proba(X_val)[:, 1]
        
        f1_lr = f1_score(y_val, val_preds_lr)
        auc_lr = roc_auc_score(y_val, val_prob_lr)
        
        results[lr_key] = {
            "model_type": "Logistic_Regression",
            "representation": rep,
            "best_params": grid_search.best_params_,
            "val_accuracy": accuracy_score(y_val, val_preds_lr),
            "val_precision": precision_score(y_val, val_preds_lr, zero_division=0),
            "val_recall": recall_score(y_val, val_preds_lr),
            "val_f1": f1_lr,
            "val_auc": auc_lr
        }
        val_probs[lr_key] = val_prob_lr
        val_y_dict[lr_key] = y_val
        logger.info(f"LR {rep} Val F1: {f1_lr:.4f} | AUC: {auc_lr:.4f} | Best C: {grid_search.best_params_['C']}")
        
        # Model 2: MLPClassifier
        logger.info("Fitting MLPClassifier...")
        mlp = MLPClassifier(
            hidden_layer_sizes=mlp_hidden,
            activation="relu",
            random_state=random_state,
            max_iter=1000
        )
        mlp.fit(X_train, y_train)
        
        # Save fit model
        mlp_key = f"MLP_Classifier_{rep}"
        fitted_models[mlp_key] = mlp
        joblib.dump(mlp, os.path.join(models_dir, f"mlp_classifier_{rep}.joblib"))
        
        # Evaluate on Validation
        val_preds_mlp = mlp.predict(X_val)
        val_prob_mlp = mlp.predict_proba(X_val)[:, 1]
        
        f1_mlp = f1_score(y_val, val_preds_mlp)
        auc_mlp = roc_auc_score(y_val, val_prob_mlp)
        
        results[mlp_key] = {
            "model_type": "MLP_Classifier",
            "representation": rep,
            "val_accuracy": accuracy_score(y_val, val_preds_mlp),
            "val_precision": precision_score(y_val, val_preds_mlp, zero_division=0),
            "val_recall": recall_score(y_val, val_preds_mlp),
            "val_f1": f1_mlp,
            "val_auc": auc_mlp
        }
        val_probs[mlp_key] = val_prob_mlp
        val_y_dict[mlp_key] = y_val
        logger.info(f"MLP {rep} Val F1: {f1_mlp:.4f} | AUC: {auc_mlp:.4f}")
        
    # Model Selection: Find model with highest F1-Score on Validation set
    best_model_key = max(results, key=lambda k: results[k]["val_f1"])
    best_config = results[best_model_key]
    logger.info("=" * 80)
    logger.info(f"BEST MODEL SELECTED: {best_model_key}")
    logger.info(f"Validation F1: {best_config['val_f1']:.4f} | AUC: {best_config['val_auc']:.4f}")
    logger.info("=" * 80)
    
    # Save the generic models as requested in output file list
    # Let's save the representative Logistic Regression and MLP Classifier model binaries
    # (using the raw_scaled representation as the primary baseline binary if needed, 
    # but let's make sure joblib copies the chosen ones as requested)
    joblib.dump(fitted_models["Logistic_Regression_raw_scaled"], os.path.join(models_dir, "logistic_regression.joblib"))
    joblib.dump(fitted_models["MLP_Classifier_raw_scaled"], os.path.join(models_dir, "mlp_classifier.joblib"))
    
    # Save the selected best overall model to best_classifier.joblib
    best_model = fitted_models[best_model_key]
    best_model_path = os.path.join(models_dir, "best_classifier.joblib")
    joblib.dump(best_model, best_model_path)
    logger.info(f"Saved best classifier binary to: {best_model_path}")
    
    # Evaluate chosen best model on Test Set (leakage-free final evaluation)
    best_rep = best_config["representation"]
    X_test, y_test, _ = load_dataset_split("test", best_rep)
    
    test_preds = best_model.predict(X_test)
    test_probs = best_model.predict_proba(X_test)[:, 1]
    
    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds, zero_division=0)
    test_rec = recall_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_probs)
    test_cm = confusion_matrix(y_test, test_preds)
    
    logger.info("FINAL TEST SET PERFORMANCE (Best Model):")
    logger.info(f"  Accuracy:  {test_acc:.4f}")
    logger.info(f"  Precision: {test_prec:.4f}")
    logger.info(f"  Recall:    {test_rec:.4f}")
    logger.info(f"  F1-Score:  {test_f1:.4f}")
    logger.info(f"  ROC-AUC:   {test_auc:.4f}")
    
    # Output metrics JSON
    final_output = {
        "validation_comparisons": results,
        "best_model_selected": best_model_key,
        "best_model_representation": best_rep,
        "test_evaluation": {
            "accuracy": test_acc,
            "precision": test_prec,
            "recall": test_rec,
            "f1_score": test_f1,
            "roc_auc": test_auc,
            "confusion_matrix": test_cm.tolist()
        }
    }
    
    results_path = os.path.join(processed_dir, "classification_results.json")
    with open(results_path, "w") as f:
        json.dump(final_output, f, indent=4)
    logger.info(f"Metrics saved to: {results_path}")
    
    # Save Predictions CSV
    predictions_df = pd.DataFrame({
        "CustomerID": X_test.index,
        "True_Label": y_test.values,
        "Predicted_Label": test_preds,
        "Predicted_Probability": test_probs
    })
    predictions_path = os.path.join(processed_dir, "classification_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to: {predictions_path}")
    
    # Generate and Save Figures
    logger.info("Generating classification figures...")
    
    # 1. Confusion Matrix (Best model on Test Set)
    plot_confusion_matrix(test_cm, best_model_key)
    
    # Prepare data for ROC curves on Validation Set
    roc_curves_data = {}
    for key, probs in val_probs.items():
        fpr, tpr, _ = roc_curve(val_y_dict[key], probs)
        auc_val = results[key]["val_auc"]
        roc_curves_data[key] = (fpr, tpr, auc_val)
        
    # 2. ROC Curve
    plot_roc_curve(roc_curves_data)
    
    # Prepare data for Precision-Recall curves on Validation Set
    pr_curves_data = {}
    for key, probs in val_probs.items():
        precision, recall, _ = precision_recall_curve(val_y_dict[key], probs)
        avg_prec = average_precision_score(val_y_dict[key], probs)
        pr_curves_data[key] = (precision, recall, avg_prec)
        
    # 3. Precision-Recall Curve
    plot_precision_recall_curve(pr_curves_data)
    
    logger.info("Classification figures successfully saved under outputs/figures/.")

if __name__ == "__main__":
    train_classification()

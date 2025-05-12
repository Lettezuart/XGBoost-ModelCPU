""" Model training using xgb.train """
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import os
import matplotlib.pyplot as plt
import shap
import json
import joblib  
import shutil
import pandas as pd

def xgbTrain_model(X_train, Y_train, X_test, Y_test, patient_id, horizon):
    # 1. Scale the features
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    # We convert back to DataFrame for SHAP
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


    # 2. Flatten Y
    Y_train_array = Y_train.to_numpy().ravel()
    Y_test_array = Y_test.to_numpy().ravel()

    # 3. Convert to DMatrix
    dtrain = xgb.DMatrix(X_train_scaled, label=Y_train_array)
    dtest  = xgb.DMatrix(X_test_scaled,  label=Y_test_array)

    # 4. Define model parameters 
    params = {
        "tree_method": "hist",  # hist means histogram-based algorithm
        "objective": "reg:squarederror", # Regression task   
        "eval_metric": ["rmse", "mae"],  # Evaluation metrics
        "device": "cuda",   # Use GPU for training
        "learning_rate": 0.003,  # Learning rate
        "max_depth": 6, # Maximum depth of the tree
        "min_child_weight": 1, # Minimum sum of instance weight (hessian) needed in a child
        "gamma": 0.1,   # Minimum loss reduction required to make a further partition on a leaf node
        "subsample": 0.8,   # Subsample ratio of the training instance
        "colsample_bytree": 0.6,    # Subsample ratio of columns when constructing each tree
        "lambda": 1,    # L2 regularization term on weights
        "alpha": 0.2,   # L1 regularization term on weights
        "random_state": 42, # Random seed for reproducibility
    }


    # 5. Train with watchlist
    watchlist = [(dtrain, "train"), (dtest, "eval")]
    evals_result = {}

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=5000,
        evals=watchlist,
        early_stopping_rounds=50,
        verbose_eval=False,
        evals_result=evals_result,
    )

    # 6. Predictions
    Y_pred = model.predict(dtest)

    # 7. Directories per pacient i horitzó
    base_output_dir = os.path.join("Trained_model_outputs", f"patient_{patient_id}", f"horizon_{horizon}min")
    model_dir = os.path.join(base_output_dir, "Model_parameters")
    shap_dir  = os.path.join(base_output_dir, "Shap")

    # Neteja total de la carpeta base (ignora errors de permisos)
    if os.path.exists(base_output_dir):
        shutil.rmtree(base_output_dir, ignore_errors=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(shap_dir,  exist_ok=True)

    # 8. Salva model i scaler
    model_path  = os.path.join(model_dir, "model_param.json")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    model.save_model(model_path)
    joblib.dump(x_scaler, scaler_path)

    # 9. Guarda metadades
    rmse     = evals_result['eval']['rmse'][-1]
    mae      = evals_result['eval']['mae'][-1]
    r2       = r2_score(Y_test_array, Y_pred)

    metadata = {
        "rmse_train": evals_result['train']['rmse'][-1],
        "rmse_eval":  rmse,
        "mae_train":  evals_result['train']['mae'][-1],
        "mae_eval":   mae,
        "r2_eval":      r2,
        "num_boost_round": model.best_iteration,
        "model_path":     model_path,
        "scaler_path":    scaler_path,
        "params":         params,
    }

    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Model, scaler & metadata saved in {model_dir}")

    # 10. SHAP visualizations
    print("Generating SHAP plots...")
    explainer   = shap.Explainer(model, X_train_scaled)
    shap_values = explainer(X_test_scaled)

    # Summary plot
    summary_plot = os.path.join(shap_dir, "shap_summary_plot.png")
    shap.summary_plot(shap_values, X_test_scaled, show=False)
    plt.savefig(summary_plot, bbox_inches="tight")
    plt.clf()

    # Bar plot
    bar_plot = os.path.join(shap_dir, "shap_summary_bar_plot.png")
    shap.summary_plot(shap_values, X_test_scaled, plot_type="bar", show=False)
    plt.savefig(bar_plot, bbox_inches="tight")
    plt.clf()

    print(f"SHAP plots saved in {shap_dir}")

    return model, x_scaler, rmse, mae, r2, Y_pred









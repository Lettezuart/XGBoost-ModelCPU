""" Model training using xgb.train """
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import os
import matplotlib.pyplot as plt
import shap
import json
import joblib  # per guardar el scaler
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
        "predictor": "gpu_predictor",   # Use GPU for prediction
        "objective": "reg:squarederror", # Regression task   
        "eval_metric": ["rmse", "mae"],  # Evaluation metrics
        "device": "cuda",   # Use GPU for training
        "learning_rate": 0.01,  # Learning rate
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
        num_boost_round=10000,
        evals=watchlist,
        early_stopping_rounds=50,
        verbose_eval=True,
        evals_result=evals_result,
    )

    # 6. Predictions
    Y_pred = model.predict(dtest)

    # 7. Directories per pacient i horitzó
    base_output_dir = os.path.join("outputs", f"patient_{patient_id}", f"horizon_{horizon}min")
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






""" Model training using XGBRegressor """
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
# import xgboost as xgb
# import numpy as np
# import json

# def xgbRegressor_model(X_train, Y_train, X_test, Y_test, Search: str = "auto"):
#     from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
#     from sklearn.metrics import mean_squared_error
#     from sklearn.preprocessing import StandardScaler
#     import xgboost as xgb
#     import numpy as np

#     # Escalem les features
#     x_scaler = StandardScaler()
#     X_train_scaled = x_scaler.fit_transform(X_train)
#     X_test_scaled = x_scaler.transform(X_test)

#     # Aplanem Y
#     Y_train_array = Y_train.to_numpy().ravel()
#     Y_test_array = Y_test.to_numpy().ravel()

#     # Hiperparàmetres
#     auto_distribution = {
#         'n_estimators': [100, 200, 300],
#         'max_depth': [3, 5, 6],  # Menys profunditat per evitar sobreajustament
#         'learning_rate': [0.01, 0.05, 0.1],  # Incrementar una mica el learning_rate
#         'subsample': [0.6, 0.7, 0.8],  # Provar més valors per augmentar la robustesa
#         'colsample_bytree': [0.6, 0.7, 0.8],  # Augmentar per reduir l'overfitting
#         'min_child_weight': [1, 3, 5],  # Augmentar min_child_weight per fer-ho més robust
#         'gamma': [0, 0.05, 0.1],  # Provar valors més baixos per gamma
#     }

#     random_distribution = {
#         'n_estimators': np.arange(100, 300, 100),  # Reduït a la meitat
#         'learning_rate': np.logspace(-3, -1, 5),  # Reduït a 5 punts
#         'max_depth': np.arange(5, 13),  # Reduït de 5 a 12
#         'min_child_weight': [0.5, 1, 3],  # Reduït a 3 valors
#         'subsample': np.logspace(-1, 0, 5),  # Reduït a 5 punts
#         'colsample_bytree': np.logspace(-1, 0, 5),  # Reduït a 5 punts
#         'gamma': np.linspace(0, 1, 3),  # Reduït a 3 punts
#         'reg_alpha': np.linspace(0, 5, 3),  # Reduït a 3 punts
#         'reg_lambda': np.linspace(0.1, 3, 3),  # Reduït a 3 punts
#         'max_delta_step': [0, 1],  # Reduït a 2 valors
#         'scale_pos_weight': [1, 2, 5],  # Reduït a 3 valors
#         'booster': ['gbtree'],  # Reduït a un sol valor
#         'objective': ['reg:squarederror'],  # Reduït a un sol valor
#         'tree_method': ['auto', 'hist'],  # Reduït a 2 valors
#         'eval_metric': ['rmse'],  # Reduït a un sol valor
#         'max_bin': [256, 512],  # Reduït a 2 valors
#     }



#     # Model base
#     model = xgb.XGBRegressor(
#         objective='reg:squarederror',
#         random_state=42,
#         tree_method='hist',
#         booster='gbtree',
#         n_jobs=10,
#     )

#     # Decidim el mètode de cerca
#     Search = Search.lower()
#     if Search == "random":
#        search_method = RandomizedSearchCV(
#             estimator=model,
#             param_distributions=random_distribution,
#             n_iter=1000,
#             scoring='neg_mean_squared_error',
#             cv=5,  # Canviar de 3 a 5 per més folds
#             verbose=1,
#             random_state=42,
#             n_jobs=10
#         )
#     elif Search == "auto":
#         search_method = GridSearchCV(
#             estimator=model,
#             param_grid=auto_distribution,
#             scoring='neg_mean_squared_error',
#             cv=5,  # Canviar de 3 a 5 per més folds
#             verbose=1,
#             n_jobs=-1
#         )
#     else:
#         raise ValueError("Search ha de ser 'auto' o 'random'")

#     # Entrenament
#     print(f"Searching hyperparameters using search method as: {Search}")
#     search_method.fit(X_train_scaled, Y_train_array)
#     # Crear directori si no existeix
#     output_dir = "ResultatsTemporals/models_parameters"
#     os.makedirs(output_dir, exist_ok=True)

#     # Convertim a tipus natius per poder-ho guardar
#     best_params_clean = {
#         k: float(v) if isinstance(v, (np.float32, np.float64)) 
#         else int(v) if isinstance(v, (np.int32, np.int64)) 
#         else v 
#         for k, v in search_method.best_params_.items()
#     }

#     # Nom del fitxer amb ID del pacient
#     filename = f"best_params_.json"
#     filepath = os.path.join(output_dir, filename)

#     # Guardem
#     with open(filepath, "w") as f:
#         json.dump(best_params_clean, f, indent=4)

#     print(f"✅ Parameters saved at: {filepath}")
    
#     best_model = search_method.best_estimator_

#     # Predicció
#     Y_pred = best_model.predict(X_test_scaled)
#     rmse = np.sqrt(mean_squared_error(Y_test_array, Y_pred))

#     return best_model, x_scaler, rmse, Y_pred









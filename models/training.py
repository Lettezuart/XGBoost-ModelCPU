""" Model training using XGBRegressor """
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import xgboost as xgb
import numpy as np
import json

def xgbRegressor_model(X_train, Y_train, X_test, Y_test, Search: str = "auto"):
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    import numpy as np

    # Escalem les features
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    # Aplanem Y
    Y_train_array = Y_train.to_numpy().ravel()
    Y_test_array = Y_test.to_numpy().ravel()

    # Hiperparàmetres
    auto_distribution = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 6],  # Menys profunditat per evitar sobreajustament
        'learning_rate': [0.01, 0.05, 0.1],  # Incrementar una mica el learning_rate
        'subsample': [0.6, 0.7, 0.8],  # Provar més valors per augmentar la robustesa
        'colsample_bytree': [0.6, 0.7, 0.8],  # Augmentar per reduir l'overfitting
        'min_child_weight': [1, 3, 5],  # Augmentar min_child_weight per fer-ho més robust
        'gamma': [0, 0.05, 0.1],  # Provar valors més baixos per gamma
    }

    random_distribution = {
        'n_estimators': np.arange(100, 300, 100),  # Reduït a la meitat
        'learning_rate': np.logspace(-3, -1, 5),  # Reduït a 5 punts
        'max_depth': np.arange(5, 13),  # Reduït de 5 a 12
        'min_child_weight': [0.5, 1, 3],  # Reduït a 3 valors
        'subsample': np.logspace(-1, 0, 5),  # Reduït a 5 punts
        'colsample_bytree': np.logspace(-1, 0, 5),  # Reduït a 5 punts
        'gamma': np.linspace(0, 1, 3),  # Reduït a 3 punts
        'reg_alpha': np.linspace(0, 5, 3),  # Reduït a 3 punts
        'reg_lambda': np.linspace(0.1, 3, 3),  # Reduït a 3 punts
        'max_delta_step': [0, 1],  # Reduït a 2 valors
        'scale_pos_weight': [1, 2, 5],  # Reduït a 3 valors
        'booster': ['gbtree'],  # Reduït a un sol valor
        'objective': ['reg:squarederror'],  # Reduït a un sol valor
        'tree_method': ['auto', 'hist'],  # Reduït a 2 valors
        'eval_metric': ['rmse'],  # Reduït a un sol valor
        'max_bin': [256, 512],  # Reduït a 2 valors
    }



    # Model base
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        tree_method='hist',
        booster='gbtree',
        n_jobs=10,
    )

    # Decidim el mètode de cerca
    Search = Search.lower()
    if Search == "random":
       search_method = RandomizedSearchCV(
            estimator=model,
            param_distributions=random_distribution,
            n_iter=1000,
            scoring='neg_mean_squared_error',
            cv=5,  # Canviar de 3 a 5 per més folds
            verbose=1,
            random_state=42,
            n_jobs=10
        )
    elif Search == "auto":
        search_method = GridSearchCV(
            estimator=model,
            param_grid=auto_distribution,
            scoring='neg_mean_squared_error',
            cv=5,  # Canviar de 3 a 5 per més folds
            verbose=1,
            n_jobs=-1
        )
    else:
        raise ValueError("Search ha de ser 'auto' o 'random'")

    # Entrenament
    print(f"Searching hyperparameters using search method as: {Search}")
    search_method.fit(X_train_scaled, Y_train_array)
    # Crear directori si no existeix
    output_dir = "ResultatsTemporals/models_parameters"
    os.makedirs(output_dir, exist_ok=True)

    # Convertim a tipus natius per poder-ho guardar
    best_params_clean = {
        k: float(v) if isinstance(v, (np.float32, np.float64)) 
        else int(v) if isinstance(v, (np.int32, np.int64)) 
        else v 
        for k, v in search_method.best_params_.items()
    }

    # Nom del fitxer amb ID del pacient
    filename = f"best_params_.json"
    filepath = os.path.join(output_dir, filename)

    # Guardem
    with open(filepath, "w") as f:
        json.dump(best_params_clean, f, indent=4)

    print(f"✅ Parameters saved at: {filepath}")
    
    best_model = search_method.best_estimator_

    # Predicció
    Y_pred = best_model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(Y_test_array, Y_pred))

    return best_model, x_scaler, rmse, Y_pred




""" Model training using xgb.train """
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def xgbTrain_model(X_train, Y_train, X_test, Y_test):
    # Escalem les dades
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    Y_train_array = Y_train.to_numpy().ravel()
    Y_test_array = Y_test.to_numpy().ravel()

    # Convertim a DMatrix
    dtrain = xgb.DMatrix(X_train_scaled, label=Y_train_array)
    dtest = xgb.DMatrix(X_test_scaled, label=Y_test_array)

    # Paràmetres del model
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',  
        'eval_metric': 'rmse',
        'learning_rate': 0.01,
        'max_depth': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
    }

    # Entrenament amb early stopping
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=10000,  # nombre màxim d’iteracions
        evals=[(dtest, 'eval')],
        early_stopping_rounds=50,
        verbose_eval=False,  # ID de la GPU a utilitzar  
    )

    # Prediccions
    Y_pred = model.predict(dtest)
    rmse = np.sqrt(mean_squared_error(Y_test_array, Y_pred))

    # Create folder if it doesn't exist 
    output_dir = "ResultatsTemporals/models_parameters"
    os.makedirs(output_dir, exist_ok=True)

    # Save model 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"xgboost_model_{timestamp}.json"
    model_path = os.path.join(output_dir, model_filename)
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Plot feature importance 
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"importance_features_{timestamp}.png"))
    plt.close()

    # Save numeric importance to CSV 
    importances = model.get_score(importance_type='weight')
    importances_df = pd.DataFrame.from_dict(importances, orient='index', columns=['importance'])
    importances_df = importances_df.sort_values(by='importance', ascending=False)
    importances_df.to_csv(os.path.join(output_dir, "feature_importance.csv"))

    print(f"Feature importance saved to {output_dir}")
    return model, x_scaler, rmse, Y_pred




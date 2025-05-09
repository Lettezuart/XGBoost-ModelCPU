from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

import xgboost as xgb
import numpy as np

def train_model(X_train, Y_train, X_test, Y_test):
    # Escalem les features
    x_scaler = MinMaxScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    # Aplanem Y
    Y_train_array = Y_train.to_numpy().ravel()
    Y_test_array = Y_test.to_numpy().ravel()

    # Hiperparàmetres
    param_grid = {
        'n_estimators': [100,200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8],
        'min_child_weight': [1, 5, 10],
        'gamma': [0, 0.1],
        
    }


    # Model base
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        tree_method='hist',
        booster='gbtree',
        n_jobs=-1
    )

    # GridSearchCV automàtic
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    # RandomizedSearchCV automàtic
    # random_search = RandomizedSearchCV(
    #     estimator=model,
    #     param_distributions=param_grid,
    #     n_iter=300,  # Prova 100 combinacions aleatòries
    #     scoring='neg_mean_squared_error',
    #     cv=5,  # Més robust (abans era 3)
    #     verbose=1,
    #     random_state=42,
    #     n_jobs=-1
    # )
    grid_search.fit(X_train_scaled, Y_train_array)

    print(f"Millors paràmetres trobats: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_

    # Predicció
    Y_pred = best_model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(Y_test_array, Y_pred))

    return best_model, x_scaler, rmse, Y_pred



from sklearn.model_selection import ParameterSampler
import numpy as np
# Afegeix un print per veure les distribucions generades
param_distributions = {
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


param_sampler = ParameterSampler(param_distributions, n_iter=10, random_state=42)

# Verifica els valors generats
for params in param_sampler:
    print(params)

import os
import multiprocessing

print("Nuclis lògics disponibles (n_jobs=-1):", multiprocessing.cpu_count())

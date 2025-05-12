# Glucose Level Prediction for Patients

This project aims to predict glucose levels in patients based on time-series data using machine learning models. The project includes a complete pipeline, from data preprocessing to final validation, and prediction using XGBoost.

## 💻 Requirements

To use this project, you will need the following packages and tools:

- **Python 3.x** 
- **Libraries**:
  - `pandas`
  - `numpy`
  - `xgboost`
  - `sklearn`
  - `joblib`
  - `matplotlib`
  - `tabulate`
  - `datetime`
  - `os`
  - `shutil`

Dependencies can be installed using `pip` with the following command:
```bash
pip install -r requirements.txt
```

## 🛠 How the Project Works

The project follows a pipeline with several stages:

1. **Data Preparation**:
   - The raw patient data is loaded from `.csv` files in the `data/raw` folder.
   - Data cleaning and preprocessing is performed. This includes applying Kalman filters or exponential moving averages (EMA) on glucose data to improve the quality of the data.
   
2. **Offline Model Training**:
   - The model is trained using XGBoost to predict glucose levels at two different time horizons (30 and 60 minutes).
   - The trained model is saved, along with its parameters and data scaler.
   - Performance metrics (RMSE, MAE, R²) are calculated for each patient and horizon.

3. **Final Validation**:
   - The trained model and scaler are loaded for making predictions on the held-out test set. 
   - Results are calculated for each patient and horizon (30 and 60 minutes). Graphs are generated showing glucose time-series predictions and Clarke error plots.
   - Performance metrics (RMSE, MAE, R²) are calculated for each patient and aggregated by horizon.

4. **Result Tables**:
   - All results (training and final validation) are printed to the console and saved in `.txt` files.
   - Tables include metrics for each patient and horizon, along with average results.

## 🔄 **Detailed Pipeline**

1. **Data Preprocessing**:
   - Raw data is loaded from `data/raw`.
   - Kalman filter or EMA is applied to glucose data.

2. **Offline Model Training**:
   - Models are trained for 30-minute and 60-minute horizons using the preprocessed data.
   - Performance metrics are calculated for each patient and horizon.

3. **Final Validation**:
   - The model is loaded and predictions are made on the test set.
   - The results are evaluated by comparing predictions to true values using RMSE, MAE, and R².
   - Clarke error plots are generated for each patient and horizon.

4. **Generated Results**:
   - Results tables with metrics for each patient are saved to `.txt` files in the `Training_metrics` (for training) and `Validation_metrics` (for final validation) directories.
   - Glucose time-series prediction graphs and Clarke error plots are saved in the `Validation_plots` directory.

## 📂 **Folder Structure**

The project follows this folder structure:

```
project/
│
├── data/
│   └── raw/                     # Raw data files of patients
│
├── models/
│   └── offline_model.py          # Code for offline model training
│
├── utils/
│   ├── final_validation.py      # Functions for final validation
│   ├── filter.py                # Functions for Kalman and EMA filtering
│   ├── evaluation.py            # Functions for generating validation plots
│   ├── clarke.py                # Functions for generating Clarke errors
│   └── data_loader.py           # Functions for loading and preparing data
│
├── Training_metrics/            # Training results
├── Validation_metrics/          # Final validation results
├── Validation_plots/            # Validation plots
└── main.py                      # Main script to execute the entire pipeline
```

## 📊 **Generated Outputs**

The outputs of the project include:

1. **Result Tables**:
   - **Text Files**: `.txt` files are generated for both training and final validation results. These files contain RMSE, MAE, and R² metrics for each patient and horizon.
   
   Example:
   ```plaintext
   📊 Offline results:
   +-------------+----------+----------+------+
   | Patient ID | RMSE 30  | MAE 30   | R²   |
   +-------------+----------+----------+------+
   | 101         | 2.14     | 1.34     | 0.98 |
   | 102         | 1.97     | 1.25     | 0.97 |
   +-------------+----------+----------+------+
   ```

2. **Result Graphs**:
   - **Time-Series Graphs**: These graphs show glucose predictions compared to true values for each patient.
   - **Clarke Error Graphs**: These graphs show Clarke error for each patient at different horizons.

3. **Model Files**:
   - **`.json` and `.pkl` Files**: The trained models and scalers are saved in a specific directory (`Trained_model_outputs`) for each patient and horizon.

## 📌 **How to Use the Project**

1. **Training**:
   To train the models offline and generate results, run the following command:
   ```bash
   python main.py
   ```

2. **Final Validation**:
   Once the models are trained, the final validation process will be automatically executed within the `main.py` script.

3. **Visualization**:
   The graphs and tables will be automatically generated and saved in the appropriate directories (`Validation_plots`,`Training_plots`, `Training_metrics`, `Validation_metrics`).
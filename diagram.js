flowchart TD
    subgraph "Data Layer"
        DS["Raw CSV Data<br/>(Data/Raw/&lt;patient_id&gt;)"]:::datastore
    end

    subgraph "Compute Layer"
        subgraph "Data Ingestion & Preprocessing"
            DL["utils/data_loader.py"]:::code_module
            DP["utils/data_processing.py"]:::code_module
            FT["utils/features.py"]:::code_module
            KF["utils/filter.py<br/>(Kalman filter)"]:::code_module
        end
        subgraph "Dataset Abstraction"
            DA["utils/datasets.py"]:::code_module
        end
        subgraph "Offline Model Training"
            TM["models/training.py"]:::process
            OM["models/offline_model.py"]:::process
        end
        Artifacts["Trained_model_outputs/"]:::datastore
    end

    subgraph "Validation & Evaluation"
        FV["utils/final_validation.py"]:::process
        EV["utils/evaluation.py"]:::process
        CL["utils/clarke.py"]:::process
    end

    subgraph "Storage & Reports"
        TMET["Training_metrics/"]:::datastore
        VMET["Validation_metrics/"]:::datastore
        TP["Training_plots/"]:::datastore
        VP["Validation_plots/"]:::datastore
    end

    Main["main.py"]:::process

    DS -->|"1 raw time-series"| DL
    DL -->|"2 clean data"| DP
    DP -->|"3 extract features"| FT
    FT -->|"4 apply Kalman filter"| KF
    KF -->|"5 features"| DA
    DA -->|"6 train splits"| TM
    TM -->|"7 train XGBoost"| OM
    OM -->|"8 model artifacts"| Artifacts
    Artifacts -->|"9 load models"| FV
    FV -->|"10 predictions"| EV
    EV -->|"11 metrics"| CL
    CL -->|"12 metrics output"| TMET
    CL -->|"12 metrics output"| VMET
    CL -->|"13 plots"| TP
    CL -->|"13 plots"| VP
    Main -->|"orchestrates pipeline"| DL
    Main -->|"orchestrates pipeline"| TM
    Main -->|"orchestrates pipeline"| FV
    Main -->|"orchestrates pipeline"| CL

    click DS "https://github.com/lettezuart/xgboost-modelgpu/tree/OnlyGPU/Data/Raw/<patient_id>"
    click DL "https://github.com/lettezuart/xgboost-modelgpu/blob/OnlyGPU/utils/data_loader.py"
    click DP "https://github.com/lettezuart/xgboost-modelgpu/blob/OnlyGPU/utils/data_processing.py"
    click FT "https://github.com/lettezuart/xgboost-modelgpu/blob/OnlyGPU/utils/features.py"
    click KF "https://github.com/lettezuart/xgboost-modelgpu/blob/OnlyGPU/utils/filter.py"
    click DA "https://github.com/lettezuart/xgboost-modelgpu/blob/OnlyGPU/utils/datasets.py"
    click TM "https://github.com/lettezuart/xgboost-modelgpu/blob/OnlyGPU/models/training.py"
    click OM "https://github.com/lettezuart/xgboost-modelgpu/blob/OnlyGPU/models/offline_model.py"
    click Artifacts "https://github.com/lettezuart/xgboost-modelgpu/tree/OnlyGPU/Trained_model_outputs/"
    click FV "https://github.com/lettezuart/xgboost-modelgpu/blob/OnlyGPU/utils/final_validation.py"
    click EV "https://github.com/lettezuart/xgboost-modelgpu/blob/OnlyGPU/utils/evaluation.py"
    click CL "https://github.com/lettezuart/xgboost-modelgpu/blob/OnlyGPU/utils/clarke.py"
    click TMET "https://github.com/lettezuart/xgboost-modelgpu/tree/OnlyGPU/Training_metrics/"
    click VMET "https://github.com/lettezuart/xgboost-modelgpu/tree/OnlyGPU/Validation_metrics/"
    click TP "https://github.com/lettezuart/xgboost-modelgpu/tree/OnlyGPU/Training_plots/"
    click VP "https://github.com/lettezuart/xgboost-modelgpu/tree/OnlyGPU/Validation_plots/"
    click Main "https://github.com/lettezuart/xgboost-modelgpu/blob/OnlyGPU/main.py"

    classDef code_module fill:#D6EAF8,stroke:#1B4F72,color:#154360
    classDef process fill:#FAD7A0,stroke:#B9770E,color:#7D6608
    classDef datastore fill:#ABEBC6,stroke:#196F3D,color:#145A32

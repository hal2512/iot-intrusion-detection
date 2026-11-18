# Dataset

The data is not included in this repository. Download **RT-IoT2022** from either source:

- UCI Machine Learning Repository (original): https://archive.ics.uci.edu/dataset/942/rt-iot2022
- Kaggle mirror: https://www.kaggle.com/datasets/supplejade/rt-iot2022real-time-internet-of-things

Citation: B. S. and R. Nagapadma (2023). *RT-IoT2022* [Data set]. UCI Machine Learning Repository.

Place the CSV here so the scripts can find it:

```
datasets/
└── raw/
    └── RT_IOT2022.csv
```

The scripts create everything else under `datasets/`:

| Path | Created by | Contents |
|---|---|---|
| `datasets/split/` | `data_split.py` | Stratified 60/20/20 train/val/test CSVs |
| `datasets/preprocessed/` | `data_preprocessing.py` | Encoded, scaled and SMOTE-resampled NumPy arrays |
| `datasets/*.json` | `data_preprocessing.py` | Label mapping, class weights, feature names |

RT-IoT2022: 123,117 network flows, 83 features (2 categorical, 81 numerical), 12 classes (9 attacks, 3 normal device behaviours).

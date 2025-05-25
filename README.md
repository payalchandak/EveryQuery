# EveryQuery

```
python src/train.py experiment=experiment.yaml paths.data_dir=$PATHS_DATA_DIR paths.meds_cohort_dir=$PATHS_MEDS_COHORT_DIR paths.output_dir=$PATHS_OUTPUT_DIR "hydra.searchpath=[$MEDS_TORCH_CONFIG_DIR,$CONFIG_DIR]"
```

```
python src/eval.py $PATHS_KWARGS hydra.searchpath=[/home/pac4279/EveryQuery/meds-torch/src/meds_torch/configs,$CONFIG_DIR]
```


Sample .env on Narsil 

PROJECT_DIR="/storage2/payal/EveryQuery"

# data
# RAW="${PROJECT_DIR}/data/mimic/raw"
RAW="/storage2/payal/dropbox/private/data/meds_ecg_pt/"
PROCESSED="${PROJECT_DIR}/data/mgb_ecg_pt/processed"
MEDS_DIR="${RAW}"
MODEL_DIR="${PROCESSED}"
PATHS_DATA_DIR="${PROCESSED}"
PATHS_MEDS_COHORT_DIR="${RAW}"
PATHS_OUTPUT_DIR="${PROJECT_DIR}/results"
PATHS_KWARGS="paths.data_dir=$PATHS_DATA_DIR paths.meds_cohort_dir=$PATHS_MEDS_COHORT_DIR paths.output_dir=$PATHS_OUTPUT_DIR"

# config
CONFIG_DIR="${PROJECT_DIR}/src/configs" 
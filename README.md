# EveryQuery

```
python src/train.py experiment=experiment.yaml paths.data_dir=$PATHS_DATA_DIR paths.meds_cohort_dir=$PATHS_MEDS_COHORT_DIR paths.output_dir=$PATHS_OUTPUT_DIR "hydra.searchpath=[$MEDS_TORCH_CONFIG_DIR,$CONFIG_DIR]"
```

```
python src/eval.py paths.data_dir=$PATHS_DATA_DIR paths.meds_cohort_dir=$PATHS_MEDS_COHORT_DIR paths.output_dir=$PATHS_OUTPUT_DIR "hydra.searchpath=[$MEDS_TORCH_CONFIG_DIR,$CONFIG_DIR]"
```

```
python src/estimate_query_prevalence.py paths.data_dir=$PATHS_DATA_DIR paths.meds_cohort_dir=$PATHS_MEDS_COHORT_DIR paths.output_dir=$PATHS_OUTPUT_DIR "hydra.searchpath=[$MEDS_TORCH_CONFIG_DIR,$CONFIG_DIR]"
```

Sample .env on O2

PROJECT_DIR="/home/pac4279/EveryQuery"

# data

DATA_DIR="/n/data1/hms/dbmi/zaklab/payal/mimic"
RAW="${DATA_DIR}/raw"
PROCESSED="${DATA_DIR}/processed"
MEDS_DIR="${RAW}"
MODEL_DIR="${PROCESSED}"
PATHS_DATA_DIR="${PROCESSED}"
PATHS_MEDS_COHORT_DIR="${RAW}"
PATHS_OUTPUT_DIR="${PROJECT_DIR}/results"
PATHS_KWARGS="paths.data_dir=$PATHS_DATA_DIR paths.meds_cohort_dir=$PATHS_MEDS_COHORT_DIR paths.output_dir=$PATHS_OUTPUT_DIR"

# config

CONFIG_DIR="${PROJECT_DIR}/src/configs"
MEDS_TORCH_CONFIG_DIR="${PROJECT_DIR}/meds-torch/src/meds_torch/configs"

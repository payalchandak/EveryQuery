# EveryQuery

.env

```
PROJECT_DIR="/home/pac4279/EveryQuery"
DATA_DIR="/n/data1/hms/dbmi/zaklab/payal/mimic"
SAVE_DIR="/n/data1/hms/dbmi/zaklab/payal/EveryQuery/results"
RAW="${DATA_DIR}/raw"
INTERMEDIATE="${DATA_DIR}/intermediate"
PROCESSED="${DATA_DIR}/processed"
```

run for processing data
```
MEICAR_process_data input_dir="$RAW" intermediate_dir="$INTERMEDIATE" output_dir="$PROCESSED"
```

run for training
```
source .env; set -a; . ./.env; set +a; python source/train.py
```

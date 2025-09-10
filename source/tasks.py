import ipdb, os
import polars as pl 

duration = { "minutes": 0, "hours": 0, "days": 30, "weeks": 0 }
min_context_per_subject = 50
query_codes = ["MEDS_DEATH","ED_OUT"]

read_dir = "/Users/payal/Desktop/EveryQuery/mimic/MEDS_intermediate/"
write_dir = "/Users/payal/Desktop/EveryQuery/mimic/MEDS_tasks/"

df = []
for file in os.listdir(f"{read_dir}/data/train"):
    df_shard = pl.read_parquet(
        f"{read_dir}/data/train/{file}"
    ).select(
        ['subject_id','time','code']
    ).unique(
    ).sort(
        ["subject_id", "time"]
    )
    df.append(df_shard)
df = pl.concat(df)

task_df = df.with_columns(
    pl.col("time").cum_count().over("subject_id").alias("context_cumsum")
).filter(
    pl.col("context_cumsum") >= min_context_per_subject # we want at least min_context_per_subject context events
).select(
    ["subject_id","time"]
).unique(
).rename(
    {"time": "prediction_time"}
).join(
    df.group_by(
        ["subject_id"]
    ).agg(
        pl.col("time").last().alias("record_end_time")
    ),
    on="subject_id",
    how="left"
).with_columns(
    (pl.col("record_end_time") - pl.col("prediction_time")).alias("future_duration")
).with_columns(
    (pl.col("future_duration") < pl.duration(**duration)).alias("censored")
).select(
    ["subject_id","prediction_time","censored"]
)

for query in query_codes:

    query_occurs = task_df.join(
        df.filter(
            pl.col("code") == query
        ).drop(
            "code",
        ).rename(
            {'time': f'{query}_time'}
        ),
        on="subject_id",
        how="left"
    ).filter(
        pl.col(f"{query}_time") > pl.col("prediction_time")
    ).filter(
        pl.col(f"{query}_time") < (pl.col("prediction_time") + pl.duration(**duration))
    ).select(
        ["subject_id","prediction_time"]
    ).unique(
    ).with_columns(
        pl.lit(1).alias(query)
    )

    task_df = task_df.join(
        query_occurs,
        on=["subject_id","prediction_time"],
        how="left"
    ).with_columns(
        # if not censored and query occurred then retain 1
        # if not censored and query did not occur (NULL) then set to 0
        # if censored then retain NULL
        pl.when(pl.col("censored")==False & pl.col(query).is_null()
        ).then(0).otherwise(pl.col(query)).alias(query)
    )

final = []
for query in query_codes:
    x = task_df.select(
        ["subject_id","prediction_time","censored",query]
    ).with_columns(
        pl.lit(query).alias('query'),
    ).rename(
        {query:'occurs'}
    )
    final.append(x)
final = pl.concat(final).sample(fraction=1,shuffle=True)
final = final.rename({'censored':'boolean_value'})
final = final.with_columns(pl.col('occurs').fill_null(-1)) # -1 for censored to avoid null errors 
os.makedirs(write_dir, exist_ok=True)
final.write_parquet(f"{write_dir}/task_df.parquet")

# ipdb.set_trace()
import ipdb, os
import polars as pl 
import time

min_context_per_subject = 50

read_dir = "/Users/payal/Desktop/EveryQuery/mimic/MEDS_intermediate"
write_dir = "/Users/payal/Desktop/EveryQuery/mimic/MEDS_tasks/"

df = []
for file in os.listdir(f"{read_dir}/data/train"):
    df_shard = (
        pl.read_parquet(f"{read_dir}/data/train/{file}")
        .select(['subject_id','time','code'])
        .unique()
        .sort(["subject_id", "time"])
    )
    df.append(df_shard)
df = pl.concat(df) # subject, time, code

duration = { "minutes": 0, "hours": 0, "days": 30, "weeks": 0 }

'''
functions, so you can parellize over shards
'''

censor_df = (
    df.with_columns(pl.col("time").cum_count().over("subject_id").alias("context_cumsum"))
    .filter(pl.col("context_cumsum") >= min_context_per_subject) # we want at least min_context_per_subject context tokens)
    .select(["subject_id","time"])
    .unique()
    .rename({"time": "prediction_time"})
    .join(df.group_by(["subject_id"]).agg(pl.col("time").last().alias("record_end_time")), on="subject_id", how="left",)
    .with_columns((pl.col("record_end_time") - pl.col("prediction_time")).alias("future_duration"))
    .with_columns((pl.col("future_duration") < pl.duration(**duration)).alias("censored"))
    .select(["subject_id","prediction_time","censored"])
)

code_metadata_df = pl.read_parquet(f"{read_dir}/metadata/codes.parquet")
query_codes = code_metadata_df.select("code").unique().to_series().to_list() 

censor_true = (
    censor_df.filter(pl.col("censored") == True)
    .with_columns([pl.lit(None).alias(query).cast(pl.Boolean) for query in query_codes])
)

censor_false = censor_df.filter(pl.col("censored") == False)

query_occurs_dfs = [censor_false]

censor_false_time = censor_false.drop("censored").with_row_index()
censor_false_index = censor_false_time.select("index")

start_time = time.time()
x = (
    censor_false.drop("censored")
    .join(df.rename({'time': f'query_time'}), on="subject_id", how="left")
    .filter((pl.col(f"query_time") > pl.col("prediction_time")) & (pl.col(f"query_time") < (pl.col("prediction_time") + pl.duration(**duration))))
    .drop("query_time")
    .unique(['subject_id', 'prediction_time', 'code'])
    .with_columns(pl.lit(1).alias('occurs').cast(pl.Boolean))
    .pivot(on='code', index=['subject_id', 'prediction_time'], values='occurs')
)
end_time = time.time()
print(f"Timing for x computation: {end_time - start_time:.2f} seconds")
# single join works but 30x slower than the loop! 

start_loop_time = time.time()
for query in query_codes:
    # do one big join here and then loop over the query codes
    # or group by query code and then check whether it occurs in time window
    query_occurs = (
        censor_false_time.join(df.filter(pl.col("code") == query).drop("code",).rename({'time': f'{query}_time'}), on="subject_id", how="left")
        .filter((pl.col(f"{query}_time") > pl.col("prediction_time")) & (pl.col(f"{query}_time") < (pl.col("prediction_time") + pl.duration(**duration))))
        .select(["index"])
        .unique()
        .with_columns(pl.lit(True).alias(query))
        .join(censor_false_index, on="index", how="right",)
        .with_columns(pl.col(query).fill_null(False))
        .select([query])
    )
    query_occurs_dfs.append(query_occurs)
end_loop_time = time.time()
print(f"Timing for query_codes loop: {end_loop_time - start_loop_time:.2f} seconds")

censor_false = pl.concat(query_occurs_dfs, how='horizontal')
assert sum(censor_false.null_count()).item() == 0

task_df = pl.concat([censor_true, censor_false], how='vertical')
task_df.write_parquet(f"/Users/payal/Desktop/EveryQuery/mimic/MEDS_all_tasks.parquet")

query_codes = ["MEDS_DEATH","ED_OUT"]
final = (
    task_df.select(["subject_id","prediction_time","censored"] + query_codes)
    .unpivot(index=['subject_id', 'prediction_time', 'censored'], variable_name="query", value_name="occurs")
    .rename({'censored':'boolean_value'})
    .with_columns(pl.col('occurs').fill_null(False)) 
)
# sample N times per subject 
os.makedirs(write_dir, exist_ok=True)
final.write_parquet(f"{write_dir}/task_df.parquet")
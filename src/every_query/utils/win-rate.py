from itertools import combinations
import pandas as pd

df = pd.read_parquet("tuning/eval_aucs_tuning_20260407_002344.parquet")

# Label each model with its (max_seq_len, num_layers) config
df["model_name"] = (
    "seq" + df["max_seq_len"].astype(str) + "_L" + df["num_layers"].astype(str)
)

# Sanity check: each raw model id should map to exactly one config label
assert df.groupby("model")["model_name"].nunique().eq(1).all()
print("Model config mapping:")
print(df[["model", "num_layers", "max_seq_len", "model_name"]].drop_duplicates().sort_values(["max_seq_len", "num_layers"]).to_string(index=False))

for duration in sorted(df["duration_days"].unique()):
    dfd = df[df["duration_days"] == duration]
    pivot = dfd.pivot_table(index="code", columns="model_name", values="occurs_auc")
    models = pivot.columns.tolist()
    n_models = len(models)
    n_codes = len(pivot)

    # Pairwise win matrix
    win_matrix = pd.DataFrame(0.0, index=models, columns=models)
    for m_a, m_b in combinations(models, 2):
        wins_a = (pivot[m_a] > pivot[m_b]).sum() + 0.5 * (pivot[m_a] == pivot[m_b]).sum()
        wins_b = n_codes - wins_a
        win_matrix.loc[m_a, m_b] = wins_a / n_codes
        win_matrix.loc[m_b, m_a] = wins_b / n_codes

    # Overall win rate per model
    total_wins = win_matrix.sum(axis=1)
    total_comparisons = n_models - 1  # each model compared against n-1 others
    win_rate = (total_wins / total_comparisons).sort_values(ascending=False)

    print(f"\n{'='*60}")
    print(f"Duration: {duration} days")
    print(f"{'='*60}")

    print(f"\nPairwise win fraction matrix (row beats column):")
    display(win_matrix.round(3))

    print(f"\nOverall win rate ranking:")
    ranking = pd.DataFrame({"win_rate": win_rate}).reset_index()
    ranking.columns = ["model", "win_rate"]
    ranking["rank"] = range(1, len(ranking) + 1)
    display(ranking.round(3))

    best = win_rate.idxmax()
    print(f"\n>>> Best model for {duration}-day horizon: {best} (win rate: {win_rate[best]:.3f})")

# Eval suite

Four-stage pipeline to evaluate a trained EveryQuery model across an
arbitrary set of (code, duration) tasks. Run in this order, each as an
installed console script (see the top-level [README](../../../README.md) for
the full pipeline):

1. `EQ_gen_eval_index` — sample prediction-time `(subject, time)` tuples
    into a deterministic eval index. Config:
    [`conf/gen_index_times_config.yaml`](conf/gen_index_times_config.yaml).
2. `EQ_gen_eval_tasks` — slice a pre-computed exhaustive task matrix by
    `(code, duration)` using that index. Config:
    [`conf/gen_tasks_config.yaml`](conf/gen_tasks_config.yaml).
3. `EQ_evaluate` — run a trained checkpoint against each sliced task,
    write per-code AUCs. Config:
    [`conf/eval_config.yaml`](conf/eval_config.yaml).
4. `EQ_select_model` — rank models by pairwise win rate over the (code,
    duration) pairs. Config:
    [`conf/select_model_config.yaml`](conf/select_model_config.yaml).

Note: step 2 depends on having already run `EQ_generate_tasks_exhaustive` to
produce the task matrix it slices.

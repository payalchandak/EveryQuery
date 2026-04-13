# Instructions

- Run gen_index_times.py to generate prediction times indices
- Run gen_task.py to generate EQ task df's for each code with index times generated before
- Run eval.py (configured by eval_config.yaml via hydra) to eval a model on a set of codes

- Run select_model.py to perform model selection based on task pairwise auc winrate across models listed in conf/select_model_config.yaml

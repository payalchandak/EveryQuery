"""Query-code sampling utilities for the EveryQuery paper experiments.

Three argparse CLIs — see each module's docstring for details:

- ``sample_train_codes`` — sample N-code YAMLs for PT runs.
- ``sample_eval_codes`` — sample paired ID/OOD eval YAMLs given an existing train YAML.
- ``sample_embedding_codes`` — sample codes for post-hoc embedding-visualization figures.

All three load their code universe from ``{metadata_dir}/codes.parquet`` passed on the CLI —
no hardcoded paths.  ``_common.py`` holds the shared filtering + hashing helpers.
"""

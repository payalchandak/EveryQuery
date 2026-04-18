"""Sample paired ID/OOD eval-code YAMLs from a MEDS cohort + an existing train-codes YAML.

Given a previously-generated train-codes YAML (produced by ``sample_train_codes.py``),
partition the cohort's code universe into ID (codes used during training) and OOD (codes
held out).  Then draw ``--num-id-codes`` / ``--num-ood-codes`` without replacement from each
and write the pair into a single YAML in the shape the eval configs expect
(``{id: [...], ood: [...]}``).

Paper-experiments only — ID/OOD held-out splits are a generalization-research construct, not
a deployment pattern.
"""

import argparse
import random
from pathlib import Path

import yaml

from every_query.paper_experiments.sample_codes._common import load_filtered_codes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        required=True,
        help="Path to a MEDS metadata directory containing codes.parquet.",
    )
    parser.add_argument(
        "--train-codes-yaml",
        type=Path,
        required=True,
        help="Path to a previously-sampled train-codes YAML (``{codes: [...]}``).",
    )
    parser.add_argument(
        "--out-fp",
        type=Path,
        required=True,
        help="Output YAML path (parent dir created if needed).",
    )
    parser.add_argument("--num-id-codes", type=int, default=5000)
    parser.add_argument("--num-ood-codes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-pattern",
        default="TIME",
        help="Drop codes containing this substring before partitioning.  Pass empty string to disable.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    exclude = args.exclude_pattern or None
    all_codes = set(load_filtered_codes(args.metadata_dir, exclude_pattern=exclude))

    with open(args.train_codes_yaml) as f:
        id_universe = set(yaml.safe_load(f)["codes"])
    # Tighten the ID universe to codes present in the current cohort — the train YAML may
    # have been generated against a different dataset version or `--exclude-pattern`, which
    # would otherwise leak phantom codes into `id_sampled`.
    id_universe &= all_codes
    ood_universe = all_codes - id_universe
    id_universe = sorted(id_universe)
    ood_universe = sorted(ood_universe)

    if len(id_universe) < args.num_id_codes:
        parser.error(
            f"Requested --num-id-codes={args.num_id_codes} but only {len(id_universe)} codes "
            f"are in the ID universe (the train_codes YAML)."
        )
    if len(ood_universe) < args.num_ood_codes:
        parser.error(
            f"Requested --num-ood-codes={args.num_ood_codes} but only {len(ood_universe)} codes "
            f"are in the OOD universe (metadata minus the train_codes YAML)."
        )

    id_sampled = random.sample(id_universe, args.num_id_codes)
    ood_sampled = random.sample(ood_universe, args.num_ood_codes)

    args.out_fp.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_fp, "w") as f:
        yaml.safe_dump({"id": id_sampled, "ood": ood_sampled}, f)

    print(f"wrote {args.out_fp} ({len(id_sampled)} ID + {len(ood_sampled)} OOD)")


if __name__ == "__main__":
    main()

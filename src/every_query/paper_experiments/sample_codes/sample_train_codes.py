"""Sample training-code YAML files for the EveryQuery paper's PT runs.

For each of ``--n-repeats`` repeats, draw ``--n-samples`` codes without replacement from a
MEDS cohort's code universe (after optional TIME-token filtering) and write them to a
content-hashed YAML file suitable for Hydra's ``train_codes`` compose group.

This is paper-experiments code: in normal EQ usage you train on a code list of your choosing
rather than sampling one.
"""

import argparse
import random
from pathlib import Path

from every_query.paper_experiments.sample_codes._common import load_filtered_codes, stable_hash_list


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        required=True,
        help="Path to a MEDS metadata directory containing codes.parquet.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write the sampled YAML files into (created if it doesn't exist).",
    )
    parser.add_argument("--n-samples", type=int, default=10000, help="Codes per repeat.")
    parser.add_argument("--n-repeats", type=int, default=5, help="How many YAML files to write.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed — governs all repeats.")
    parser.add_argument(
        "--exclude-pattern",
        default="TIME",
        help="Drop codes containing this substring.  Pass empty string to disable.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    exclude = args.exclude_pattern or None
    codes = load_filtered_codes(args.metadata_dir, exclude_pattern=exclude)
    print(f"loaded {len(codes)} codes from {args.metadata_dir}/codes.parquet (exclude={exclude!r})")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(args.n_repeats):
        id_codes = random.sample(codes, args.n_samples)
        id_hash = stable_hash_list(id_codes)
        out_fp = args.out_dir / f"{args.n_samples}_ID__{id_hash}.yaml"
        with open(out_fp, "x") as f:
            f.write("codes:\n")
            for code in id_codes:
                f.write(f'  - "{code}"\n')
        print(f"wrote {out_fp}")

    print(f"done — {args.n_repeats} x {args.n_samples}-code YAMLs in {args.out_dir}")


if __name__ == "__main__":
    main()

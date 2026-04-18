"""Sample query codes for post-hoc embedding visualization.

Given a MEDS cohort + an exclusion pattern, draw ``--n-samples`` codes without replacement
from the filtered universe and write a content-hashed YAML list.  These get fed to the
embedding-plot scripts in paper_experiments to produce UMAP-style figures over query
vocabulary.
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
        help="Directory to write the sampled YAML into (created if it doesn't exist).",
    )
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-pattern",
        default="TIME",
        help="Drop codes containing this substring.  Pass empty string to disable.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    exclude = args.exclude_pattern or None
    codes = load_filtered_codes(args.metadata_dir, exclude_pattern=exclude)
    print(f"loaded {len(codes)} codes (exclude={exclude!r})")

    sampled = random.sample(codes, args.n_samples)
    hash_str = stable_hash_list(sampled)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = args.out_dir / f"embed_{args.n_samples}_{hash_str}.yaml"
    with open(out_fp, "x") as f:
        for code in sampled:
            f.write(f'- "{code}"\n')
    print(f"wrote {out_fp}")


if __name__ == "__main__":
    main()

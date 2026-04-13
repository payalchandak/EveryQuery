"""Upload best model checkpoints + configs from a list of run directories to GCS."""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("upload_models")

REQUIRED_FILES = ("best_model.ckpt",)
OPTIONAL_FILES = ("config.yaml", "resolved_config.yaml")


def parse_runs_file(runs_file: Path) -> list[Path]:
    runs: list[Path] = []
    for raw in runs_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        runs.append(Path(line))
    return runs


def resolve_destination_names(run_dirs: list[Path]) -> dict[Path, str]:
    name_to_runs: dict[str, list[Path]] = {}
    for run in run_dirs:
        name_to_runs.setdefault(run.name, []).append(run)

    dest_names: dict[Path, str] = {}
    for name, runs in name_to_runs.items():
        if len(runs) == 1:
            dest_names[runs[0]] = name
        else:
            logger.warning(
                "Multiple runs share basename %r; disambiguating with a path hash.", name
            )
            for run in runs:
                short = hashlib.sha1(str(run.resolve()).encode()).hexdigest()[:8]
                dest_names[run] = f"{name}-{short}"
    return dest_names


def gsutil_cp(src: Path, dst: str, *, dry_run: bool) -> bool:
    cmd = ["gsutil", "-m", "cp", str(src), dst]
    if dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return True
    logger.info("Uploading %s -> %s", src, dst)
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        logger.error("`gsutil` not found on PATH. Install the Google Cloud SDK (gsutil) first.")
        raise
    except subprocess.CalledProcessError as e:
        logger.error("gsutil cp failed (exit %s) for %s", e.returncode, src)
        return False
    return True


def upload_run(run_dir: Path, dest_name: str, bucket: str, *, dry_run: bool) -> str:
    """Return 'uploaded', 'skipped', or 'failed' for one run directory."""
    if not run_dir.is_dir():
        logger.error("Run directory does not exist: %s", run_dir)
        return "skipped"

    missing_required = [f for f in REQUIRED_FILES if not (run_dir / f).is_file()]
    if missing_required:
        logger.error(
            "Skipping %s: missing required file(s) %s", run_dir, missing_required
        )
        return "skipped"

    dest_prefix = f"{bucket.rstrip('/')}/{dest_name}"

    all_ok = True
    for filename in REQUIRED_FILES:
        ok = gsutil_cp(run_dir / filename, f"{dest_prefix}/{filename}", dry_run=dry_run)
        all_ok = all_ok and ok

    for filename in OPTIONAL_FILES:
        src = run_dir / filename
        if not src.is_file():
            logger.warning("Optional file missing, skipping: %s", src)
            continue
        ok = gsutil_cp(src, f"{dest_prefix}/{filename}", dry_run=dry_run)
        all_ok = all_ok and ok

    return "uploaded" if all_ok else "failed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-file",
        type=Path,
        required=True,
        help="Text file with one run directory path per line. '#' comments and blank lines are ignored.",
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="Destination GCS prefix, e.g. gs://my-bucket/everyquery-models",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the gsutil commands that would run without executing them.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.bucket.startswith("gs://"):
        logger.error("--bucket must start with gs:// (got %r)", args.bucket)
        return 2

    if not args.runs_file.is_file():
        logger.error("--runs-file does not exist: %s", args.runs_file)
        return 2

    run_dirs = parse_runs_file(args.runs_file)
    if not run_dirs:
        logger.error("No run directories parsed from %s", args.runs_file)
        return 2

    dest_names = resolve_destination_names(run_dirs)

    counts = {"uploaded": 0, "skipped": 0, "failed": 0}
    failures: list[Path] = []
    for run in run_dirs:
        result = upload_run(run, dest_names[run], args.bucket, dry_run=args.dry_run)
        counts[result] += 1
        if result == "failed":
            failures.append(run)

    print(
        f"\nDone. uploaded={counts['uploaded']} "
        f"skipped={counts['skipped']} failed={counts['failed']}"
    )
    if failures:
        print("Failed runs:")
        for run in failures:
            print(f"  {run}")

    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

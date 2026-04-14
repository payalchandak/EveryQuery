"""Upload selected run dirs (ckpt + configs + eval outputs) to GCS.

Bucket and path layout are read from ``scripts/sync_config.yaml`` so uploads
land at the same paths ``scripts/sync.py`` would use for the ``outputs``
mapping. Each listed run is expected to live under ``$OUTPUT_DIR/outputs/``.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

logger = logging.getLogger("upload_models")

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNC_CONFIG_PATH = REPO_ROOT / "scripts" / "sync_config.yaml"

REQUIRED_FILES = ("best_model.ckpt",)
OPTIONAL_FILES = ("config.yaml", "resolved_config.yaml")
EVAL_SUBDIR = "eval"


def parse_runs_file(runs_file: Path) -> list[Path]:
    runs: list[Path] = []
    for raw in runs_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        runs.append(Path(line))
    return runs


def load_sync_outputs_mapping() -> tuple[str, str, Path]:
    """Return (bucket, gcs_prefix, server_root) from sync_config.yaml + env."""
    with open(SYNC_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    bucket = cfg["bucket"]
    mapping = next((m for m in cfg["mappings"] if m["name"] == "outputs"), None)
    if mapping is None:
        raise RuntimeError(f"No 'outputs' mapping found in {SYNC_CONFIG_PATH}")

    env_var = mapping["server_root_env"]
    base = os.environ.get(env_var)
    if not base:
        raise RuntimeError(
            f"Environment variable {env_var} is not set. Set it in your .env file or export it."
        )
    server_root = Path(base) / mapping.get("server_root_suffix", "")
    return bucket, mapping["gcs_prefix"], server_root


def _run_gsutil(cmd: list[str], *, dry_run: bool) -> bool:
    if dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return True
    logger.info("Running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        logger.error("`gsutil` not found on PATH. Install the Google Cloud SDK first.")
        raise
    except subprocess.CalledProcessError as e:
        logger.error("gsutil failed (exit %s): %s", e.returncode, " ".join(cmd))
        return False
    return True


def gsutil_cp(src: Path, dst: str, *, dry_run: bool) -> bool:
    return _run_gsutil(["gsutil", "-m", "cp", str(src), dst], dry_run=dry_run)


def gsutil_rsync(src: Path, dst: str, *, dry_run: bool) -> bool:
    return _run_gsutil(["gsutil", "-m", "rsync", "-r", str(src), dst], dry_run=dry_run)


def upload_run(
    run_dir: Path,
    *,
    bucket: str,
    gcs_prefix: str,
    server_root: Path,
    dry_run: bool,
) -> str:
    """Return 'uploaded', 'skipped', or 'failed' for one run directory."""
    if not run_dir.is_dir():
        logger.error("Run directory does not exist: %s", run_dir)
        return "skipped"

    try:
        rel = run_dir.resolve().relative_to(server_root.resolve())
    except ValueError:
        logger.error(
            "Run %s is not under the sync outputs root %s; skipping.",
            run_dir,
            server_root,
        )
        return "skipped"

    missing_required = [f for f in REQUIRED_FILES if not (run_dir / f).is_file()]
    if missing_required:
        logger.error("Skipping %s: missing required file(s) %s", run_dir, missing_required)
        return "skipped"

    dest_prefix = f"{bucket.rstrip('/')}/{gcs_prefix.strip('/')}/{rel.as_posix()}"

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

    eval_dir = run_dir / EVAL_SUBDIR
    if eval_dir.is_dir() and any(eval_dir.iterdir()):
        ok = gsutil_rsync(eval_dir, f"{dest_prefix}/{EVAL_SUBDIR}", dry_run=dry_run)
        all_ok = all_ok and ok
    else:
        logger.warning("No eval outputs to upload for %s (missing or empty %s/)", run_dir, EVAL_SUBDIR)

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
        "--dry-run",
        action="store_true",
        help="Print the gsutil commands that would run without executing them.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv(REPO_ROOT / ".env")

    if not args.runs_file.is_file():
        logger.error("--runs-file does not exist: %s", args.runs_file)
        return 2

    try:
        bucket, gcs_prefix, server_root = load_sync_outputs_mapping()
    except (RuntimeError, FileNotFoundError, KeyError) as e:
        logger.error("Failed to load sync config: %s", e)
        return 2

    if not bucket.startswith("gs://"):
        logger.error("bucket in sync_config.yaml must start with gs:// (got %r)", bucket)
        return 2

    run_dirs = parse_runs_file(args.runs_file)
    if not run_dirs:
        logger.error("No run directories parsed from %s", args.runs_file)
        return 2

    counts = {"uploaded": 0, "skipped": 0, "failed": 0}
    failures: list[Path] = []
    for run in run_dirs:
        result = upload_run(
            run,
            bucket=bucket,
            gcs_prefix=gcs_prefix,
            server_root=server_root,
            dry_run=args.dry_run,
        )
        counts[result] += 1
        if result == "failed":
            failures.append(run)

    print(f"\nDone. uploaded={counts['uploaded']} skipped={counts['skipped']} failed={counts['failed']}")
    if failures:
        print("Failed runs:")
        for run in failures:
            print(f"  {run}")

    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

"""Bidirectional GCS sync for EveryQuery data.

Usage:
    python scripts/sync.py upload              # push all mappings to GCS
    python scripts/sync.py download            # pull all mappings from GCS
    python scripts/sync.py upload --mapping outputs   # sync one mapping
    python scripts/sync.py download --dry-run         # preview commands
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

logger = logging.getLogger("sync")

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "sync_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def resolve_server_root(mapping: dict) -> Path:
    """Build the server source path from env var + optional suffix."""
    env_var = mapping["server_root_env"]
    base = os.environ.get(env_var)
    if not base:
        logger.error(
            "Environment variable %s is not set. Set it in your .env file or export it.",
            env_var,
        )
        sys.exit(1)
    suffix = mapping.get("server_root_suffix", "")
    return Path(base) / suffix


def gsutil_rsync(src: str, dst: str, *, dry_run: bool, delete: bool = False) -> bool:
    cmd = ["gsutil", "-m", "rsync", "-r"]
    if delete:
        cmd.append("-d")
    cmd.extend([src, dst])

    if dry_run:
        print(f"DRY-RUN: {' '.join(cmd)}")
        return True

    logger.info("Running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        logger.error("`gsutil` not found on PATH. Install the Google Cloud SDK first.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error("gsutil rsync failed (exit %s)", e.returncode)
        return False
    return True


def cmd_upload(mappings: list[dict], bucket: str, *, dry_run: bool) -> int:
    failures = 0
    for m in mappings:
        if "server_root_env" not in m:
            logger.info("Skipping [%s]: no server_root_env (download-only mapping).", m["name"])
            continue
        server_root = resolve_server_root(m)
        gcs_dst = f"{bucket.rstrip('/')}/{m['gcs_prefix'].strip('/')}"

        if not dry_run and not server_root.is_dir():
            logger.error("Server root does not exist: %s", server_root)
            failures += 1
            continue

        print(f"\n--- Uploading [{m['name']}]: {server_root} -> {gcs_dst}")
        ok = gsutil_rsync(str(server_root), gcs_dst, dry_run=dry_run)
        if not ok:
            failures += 1
    return failures


def cmd_download(mappings: list[dict], bucket: str, *, dry_run: bool) -> int:
    failures = 0
    for m in mappings:
        prefix = m["gcs_prefix"].strip("/")
        gcs_src = f"{bucket.rstrip('/')}/{prefix}" if prefix else bucket.rstrip("/")
        local_dst = REPO_ROOT / m["local_root"]

        if not dry_run:
            local_dst.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Downloading [{m['name']}]: {gcs_src} -> {local_dst}")
        ok = gsutil_rsync(gcs_src, str(local_dst), dry_run=dry_run)
        if not ok:
            failures += 1
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["upload", "download"],
        help="Sync direction: upload (server -> GCS) or download (GCS -> local).",
    )
    parser.add_argument(
        "--mapping",
        help="Sync only the named mapping (e.g. 'outputs', 'tasks', 'meds').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print gsutil commands without executing them.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv(REPO_ROOT / ".env")

    cfg = load_config()
    bucket = cfg["bucket"]
    mappings = cfg["mappings"]

    if args.mapping:
        mappings = [m for m in mappings if m["name"] == args.mapping]
        if not mappings:
            logger.error(
                "No mapping named %r. Available: %s",
                args.mapping,
                ", ".join(m["name"] for m in cfg["mappings"]),
            )
            return 2

    if args.command == "upload":
        failures = cmd_upload(mappings, bucket, dry_run=args.dry_run)
    else:
        failures = cmd_download(mappings, bucket, dry_run=args.dry_run)

    if failures:
        print(f"\n{failures} mapping(s) failed.")
        return 1

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

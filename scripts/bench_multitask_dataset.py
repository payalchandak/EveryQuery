"""Time the multitask dataset against real sampler output: init, __getitem__, collate, DataLoader.

    uv run python scripts/bench_multitask_dataset.py \
        --cohort $FINAL_DATA_DIR --labels $TASK_DIR/multitask --split train \
        --batch-size 64 --num-batches 50 --num-workers 8 [--shuffle]

The collate breakdown separates the packed-label path (gather from memmap + unpackbits) from the
upstream MEDS collate, which is what tells you whether the *label format* is the slow part.
"""

import argparse
import time

import numpy as np
import torch
from meds_torchdata.config import MEDSTorchDataConfig
from torch.utils.data import DataLoader

from every_query.data.multitask_dataset import MultitaskBoundaryPytorchDataset


def timed(fn, n=1):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-batches", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--shuffle", action="store_true", help="random rows (cold memmap pages) vs sequential")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = MEDSTorchDataConfig(
        tensorized_cohort_dir=args.cohort,
        task_labels_dir=args.labels,
        max_seq_len=args.max_seq_len,
        seq_sampling_strategy="to_end",
        static_inclusion_mode="omit",
        batch_mode="SM",
    )
    t0 = time.perf_counter()
    ds = MultitaskBoundaryPytorchDataset(cfg, split=args.split)
    print(f"init: {time.perf_counter() - t0:.2f}s  rows={len(ds)}  K={ds.num_bounds}  V={ds.vocab_size}")
    packed_b, dense_b = ds.num_bounds * ds.packed_width, ds.num_bounds * ds.vocab_size
    print(f"labels per row: {packed_b} B packed -> {dense_b} B dense")

    rng = np.random.default_rng(args.seed)
    n = args.batch_size * args.num_batches
    idx = rng.permutation(len(ds))[:n] if args.shuffle else np.arange(min(n, len(ds)))
    batches = [idx[i : i + args.batch_size].tolist() for i in range(0, len(idx), args.batch_size)]

    # 1. __getitem__ only (no labels touched)
    t_item = timed(lambda: [ds[i] for i in batches[0]]) / len(batches[0])
    print(f"__getitem__: {t_item * 1e3:.2f} ms/item")

    # 2. collate breakdown on one batch
    items = [ds[i] for i in batches[0]]
    t_base = timed(lambda: super(MultitaskBoundaryPytorchDataset, ds).collate(items), 3)
    t_gather = timed(lambda: ds.gather_packed(items), 3)
    packed = ds.gather_packed(items)
    t_unpack = timed(lambda: ds.unpack_targets(packed), 3)
    t_full = timed(lambda: ds.collate(items), 3)
    print(
        f"collate (B={len(items)}): base MEDS {t_base * 1e3:.1f} ms | gather packed {t_gather * 1e3:.1f} ms "
        f"| unpack {t_unpack * 1e3:.1f} ms | full {t_full * 1e3:.1f} ms"
    )

    # 3. end-to-end: single process over all batches (gather hits cold pages if --shuffle)
    t0 = time.perf_counter()
    for b in batches:
        ds.collate([ds[i] for i in b])
    dt = time.perf_counter() - t0
    print(f"main-process loop: {len(batches) / dt:.1f} batches/s  ({dt / len(batches) * 1e3:.1f} ms/batch)")

    # 4. real DataLoader with workers
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        collate_fn=ds.collate,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    it = iter(loader)
    next(it)  # warm-up / worker spawn
    t0 = time.perf_counter()
    k = 0
    for _ in range(args.num_batches - 1):
        batch = next(it, None)
        if batch is None:
            break
        k += 1
    dt = time.perf_counter() - t0
    print(f"DataLoader (workers={args.num_workers}): {k / dt:.1f} batches/s")
    print(f"  targets shape: {tuple(batch.targets.shape)}")
    print(f"  targets bytes/batch: {batch.targets.numel() / 1e6:.1f} MB (bool) -> loss casts to float: x4")
    print(f"  torch threads={torch.get_num_threads()}")


if __name__ == "__main__":
    main()

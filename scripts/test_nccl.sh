#!/bin/bash
#SBATCH --job-name=nccl_test
#SBATCH --output=logs/nccl_test_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:10:00

set -euo pipefail
mkdir -p logs

echo "=== NCCL Smoke Test ==="
echo "Host: $(hostname) | Date: $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export NCCL_P2P_DISABLE=1
export NCCL_SHM_USE_CUDA_MEMCPY=1

UVENV="$HOME/eq_stuff/eq"

srun $UVENV/bin/python -c "
import os
import time
import torch
import torch.distributed as dist

rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', 0)))
world = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS', 1)))
local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID', 0)))

torch.cuda.set_device(local_rank)
device = torch.device(f'cuda:{local_rank}')

print(f'[Rank {rank}] GPU {local_rank}: {torch.cuda.get_device_name(device)}', flush=True)
print(f'[Rank {rank}] CUDA {torch.version.cuda} | PyTorch {torch.__version__} | NCCL {torch.cuda.nccl.version()}', flush=True)

# Init process group
dist.init_process_group(backend='nccl', rank=rank, world_size=world)
print(f'[Rank {rank}] Process group initialized', flush=True)

# Test 1: all-reduce (small)
t = torch.tensor([rank + 1.0], device=device)
dist.all_reduce(t)
assert t.item() == world * (world + 1) / 2, f'all-reduce failed: got {t.item()}'
print(f'[Rank {rank}] PASS: all-reduce (small)', flush=True)

# Test 2: all-reduce (large — ~200MB, similar to gradient sync)
big = torch.randn(50_000_000, device=device)
dist.barrier()
t0 = time.time()
dist.all_reduce(big)
torch.cuda.synchronize()
dt = time.time() - t0
gb = big.numel() * 4 / 1e9
print(f'[Rank {rank}] PASS: all-reduce (200MB) in {dt:.3f}s  ({gb/dt:.1f} GB/s)', flush=True)

# Test 3: broadcast
src = torch.tensor([42.0], device=device) if rank == 0 else torch.zeros(1, device=device)
dist.broadcast(src, src=0)
assert src.item() == 42.0, f'broadcast failed: got {src.item()}'
print(f'[Rank {rank}] PASS: broadcast', flush=True)

# Test 4: barrier timing
dist.barrier()
t0 = time.time()
for _ in range(100):
    dist.barrier()
torch.cuda.synchronize()
dt = time.time() - t0
print(f'[Rank {rank}] PASS: 100 barriers in {dt:.3f}s ({dt/100*1000:.1f}ms each)', flush=True)

# Test 5: sustained all-reduce (simulate gradient sync over many steps)
dist.barrier()
t0 = time.time()
for i in range(20):
    grad = torch.randn(25_000_000, device=device)
    dist.all_reduce(grad)
torch.cuda.synchronize()
dt = time.time() - t0
print(f'[Rank {rank}] PASS: 20x all-reduce (100MB each) in {dt:.3f}s', flush=True)

dist.destroy_process_group()
print(f'[Rank {rank}] ALL TESTS PASSED', flush=True)
"

echo "=== Done ==="

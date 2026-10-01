#!/bin/bash
# JOSS paper benchmark — all architectures, back-to-back on one exclusive GPU node.
#
# Runs the benchmark suite (benchmark/manual_benchmarking.jl) for cpu, gpu,
# reactant-cpu and reactant-gpu in sequence, so every architecture sees identical
# machine conditions and the GPU numbers are not contaminated by shared-GPU
# contention on the login node. Each run merges its results into
# benchmark/assets/benchmark_results.json (keyed by architecture) and records the
# Terrarium git revision it ran at; docs/joss/figures.jl reads that store for
# Figure 2.
#
# By default every suite runs, keeping the benchmark README complete. Pass a
# comma-separated suite list as the first argument to restrict the run — for the
# JOSS figure alone (the :soil_heat SoilModel sweep) use `bench202`. A partial run
# replaces each architecture's stored results with just the suites it ran.
#
# Submit from the repo root:  sbatch docs/joss/joss_benchmark.slurm
# Restrict suites / mode via positional args:  sbatch docs/joss/joss_benchmark.slurm bench202 default
#
#SBATCH --job-name=joss-bench
#SBATCH --qos=gpushort
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/home/briangr/lab/repos/julia/Terrarium.jl/scratch/joss-benchmark-%j.out
#SBATCH --error=/home/briangr/lab/repos/julia/Terrarium.jl/scratch/joss-benchmark-%j.err

set -uo pipefail

# Pin the Julia that built the precompile depot (1.10.12); the login PATH's bare
# `julia` is 1.12.7, which crashes precompiling this Manifest's stdlib.
JULIA=/home/briangr/.julia/juliaup/julia-1.10.12+0.x64.linux.gnu/bin/julia

# Isolated depot: avoids the shared ~/.julia/compiled corruption race and is
# already warm (packages + v1.10 compile cache, incl. the CUDA and Reactant chains).
export JULIA_DEPOT_PATH=/home/briangr/tmp/julia

MAIN=/home/briangr/lab/repos/julia/Terrarium.jl
# Optional positional args: comma-separated suites (empty = all suites), then mode.
SUITES=${1:-}
MODE=${2:-default}

cd "$MAIN"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv

for arch in cpu gpu reactant-cpu reactant-gpu; do
    echo "=== $arch ($MODE${SUITES:+, suites=$SUITES}) ==="
    if [[ -n "$SUITES" ]]; then
        "$JULIA" --project=benchmark benchmark/manual_benchmarking.jl "$arch" "$MODE" --suites "$SUITES"
    else
        "$JULIA" --project=benchmark benchmark/manual_benchmarking.jl "$arch" "$MODE"
    fi
    echo "=== $arch done (exit $?) ==="
done

echo "=== git revision recorded in the store ==="
grep -o '"git_revision": *"[^"]*"' benchmark/assets/benchmark_results.json | sort -u

#!/bin/bash
# Run the Terrarium benchmark suite (manual_benchmarking.jl) as a Slurm batch job.
#
# Usage, from the repository root or from benchmark/:
#
#     sbatch [cluster options] benchmark/run_benchmarks.sbatch <arch> [mode]
#
# <arch> is one of cpu, gpu, reactant-cpu, reactant-gpu, and [mode] one of quick, default (the default),
# long, or a numeric timestep multiplier; see `julia --project=. manual_benchmarking.jl --help`.
#
# Cluster-specific options are deliberately not set in this script: pass the partition, QoS, account,
# time limit, and GPU request to sbatch, for example
#
#     sbatch --partition=<partition> --qos=<qos> --time=<hh:mm:ss> benchmark/run_benchmarks.sbatch cpu
#     sbatch --partition=<partition> --qos=<qos> --gres=gpu:1 --time=<hh:mm:ss> benchmark/run_benchmarks.sbatch gpu
#
# Request a whole node (--exclusive) for numbers worth quoting, so that other jobs do not share the
# memory bandwidth. For reactant-cpu, request the cores XLA should use (--exclusive or --cpus-per-task);
# a job that is restricted to one core is stored under the separate `reactant-cpu-*-1core` label.
#
# The job runs `Pkg.instantiate()` in benchmark/ before benchmarking. If the compute nodes have no
# internet access, instantiate the environment on a login node first, with the same Julia version as the
# job, so that the job finds every package and artifact in the depot:
#
#     cd benchmark && julia --project=. -e 'using Pkg; Pkg.instantiate()'
#
# Environment variables:
#
#     JULIA          Julia command, which may include a juliaup channel, e.g. `julia +1.12`
#                    (default: `julia` on PATH). Use the same Julia version as the results you
#                    compare against; an existing benchmark/Manifest.toml must match it.
#     TERRARIUM_DIR  repository root (default: the directory sbatch was called from)
#     SINGLE_CORE=1  pin the run to the first CPU core allocated to the job, for a single-core
#                    reactant-cpu run
#
#SBATCH --job-name=terrarium-benchmarks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=%x-%j.out

set -euo pipefail

ARCH="${1:-cpu}"
MODE="${2:-default}"
read -r -a JULIA_CMD <<< "${JULIA:-julia}"

ROOT="${TERRARIUM_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -f "$ROOT/benchmark/manual_benchmarking.jl" ] && [ -f "$ROOT/manual_benchmarking.jl" ]; then
    ROOT="$ROOT/.."
fi
if [ ! -f "$ROOT/benchmark/manual_benchmarking.jl" ]; then
    echo "Cannot find benchmark/manual_benchmarking.jl under $ROOT; set TERRARIUM_DIR to the repository root." >&2
    exit 1
fi
cd "$ROOT/benchmark"

PREFIX=()
if [ "${SINGLE_CORE:-0}" = "1" ]; then
    CORE=$(taskset -cp $$ | sed 's/.*: //' | cut -d, -f1 | cut -d- -f1)
    PREFIX=(taskset -c "$CORE")
fi

echo "Terrarium benchmarks: arch=$ARCH mode=$MODE julia=${JULIA_CMD[*]} dir=$PWD host=$(hostname) job=${SLURM_JOB_ID:-none}"
"${JULIA_CMD[@]}" --version
"${JULIA_CMD[@]}" --project=. -e 'using Pkg; Pkg.instantiate()'
${PREFIX[@]+"${PREFIX[@]}"} "${JULIA_CMD[@]}" --project=. --threads=1 manual_benchmarking.jl "$ARCH" "$MODE"

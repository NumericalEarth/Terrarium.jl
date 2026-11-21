#!/bin/bash
# Two-year coupled Terrarium–SpeedyWeather simulation for Figure 3 of the JOSS paper
# (docs/joss/run_terrarium_speedy_simulation.jl). Runs on a GPU if one is available, otherwise on the CPU.
# Request a GPU at submission with the options of the cluster, e.g. `sbatch --gres=gpu:1 ...`.
# The output is written to a new run directory in docs/joss/outputs.
#
# Submit from the repository root:  sbatch docs/joss/run_terrarium_speedy_simulation.sbatch
#
#SBATCH --job-name=joss-simulation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=joss-simulation-%j.out

set -euo pipefail

# the docs/joss environment is resolved and precompiled with Julia 1.10 (+lts)
julia --project=docs/joss docs/joss/run_terrarium_speedy_simulation.jl

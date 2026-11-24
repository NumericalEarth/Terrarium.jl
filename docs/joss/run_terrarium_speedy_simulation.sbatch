#!/bin/bash
# Coupled Terrarium–SpeedyWeather simulation for Figure 3 of the JOSS paper
# (docs/joss/run_terrarium_speedy_simulation.jl). Runs on a GPU if one is available, otherwise on the CPU.
# Request a GPU at submission with the options of the cluster, e.g. `sbatch --gres=gpu:1 ...`.
# The output is written to a new run directory in docs/joss/outputs.
#
# Submit from the repository root:  sbatch docs/joss/run_terrarium_speedy_simulation.sbatch
# Options of the script (see `--help`) are passed on, e.g.  sbatch docs/joss/run_terrarium_speedy_simulation.sbatch --period 2y
#
#SBATCH --job-name=joss-simulation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=joss-simulation-%j.out

set -euo pipefail

# First instantiate and precompile the environment
julia --project=docs/joss -e "import Pkg; Pkg.instantiate()"

# Run the simulation
julia --project=docs/joss docs/joss/run_terrarium_speedy_simulation.jl "$@"

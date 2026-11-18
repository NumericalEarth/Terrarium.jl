#!/bin/bash
# Adjoint sensitivity of the land latent heat flux to the initial atmospheric humidity
# (docs/joss/run_coupled_sensitivity.jl). CPU only.
#
#SBATCH --job-name=joss-sensitivity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=joss-sensitivity-%j.out

set -euo pipefail

# vanilla Enzyme needs Julia 1.10 or 1.11; here we select +lts (1.10) with juliaup
julia +lts --project=docs/joss docs/joss/run_coupled_sensitivity.jl

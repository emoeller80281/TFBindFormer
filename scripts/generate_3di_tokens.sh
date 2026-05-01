#!/bin/bash -l
#SBATCH --job-name=generate_3di_tokens
#SBATCH --output=LOGS/generate_tokens/%x_%A.log
#SBATCH --error=LOGS/generate_tokens/%x_%A.err
#SBATCH --time=12:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 12
#SBATCH --mem=64G

# Generate 3Di tokens from TF PDB structures using Foldseek

source activate tfbindformer

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2026.TFBINDFORMER.MOELLER/TFBindFormer"
cd $PROJECT_DIR

set -e  # exit on error (recommended)

PDB_DIR="${PROJECT_DIR}/data/tf_data/tf_structure"
OUT_DIR="${PROJECT_DIR}/data/tf_data/3di_out_test"

mkdir -p "${OUT_DIR}"

foldseek createdb -v 3 "${PDB_DIR}" "${OUT_DIR}/pdb_3Di"
foldseek lndb "${OUT_DIR}/pdb_3Di_h" "${OUT_DIR}/pdb_3Di_ss_h"
foldseek convert2fasta "${OUT_DIR}/pdb_3Di_ss" "${OUT_DIR}/pdb_3Di_ss.fasta"

echo "3Di FASTA written to ${OUT_DIR}/pdb_3Di_ss.fasta"

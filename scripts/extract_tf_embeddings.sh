#!/bin/bash
#SBATCH --job-name=extract_tf_embeddings
#SBATCH --output=LOGS/extract_tf_embeddings/%x_%j.log
#SBATCH --error=LOGS/extract_tf_embeddings/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --mem=128G

set -eo pipefail

source activate tfbindformer

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2026.TFBINDFORMER.MOELLER/TFBindFormer"

cd ${PROJECT_DIR}/scripts

python extract_tf_embeddings.py \
  --aa_dir ${PROJECT_DIR}/data/tf_data/tf_sequence \
  --di_fasta ${PROJECT_DIR}/data/tf_data/3di_out_test/pdb_3Di_ss.fasta \
  --out_dir ${PROJECT_DIR}/data/tf_data/tf_embeddings_test/
#!/bin/bash -l
#SBATCH --job-name=train_tfbindformer
#SBATCH --output=LOGS/train_tfbindformer/%x_%j.log
#SBATCH --error=LOGS/train_tfbindformer/%x_%j.err
#SBATCH --time=36:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=4
#SBATCH -c 16
#SBATCH --mem=64G

set -eo pipefail

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2026.TFBINDFORMER.MOELLER/TFBindFormer"
cd $PROJECT_DIR

echo "Activating conda environment and starting training..."
source activate tfbindformer


# --- Memory + math ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:32
export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=1

# --- Threading: set to match SLURM CPUs per task ---
THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS
export BLIS_NUM_THREADS=$THREADS
export KMP_AFFINITY=granularity=fine,compact,1,0

# --- NCCL / networking overrides ---
# Dynamically find the interface with 10.90.29.* network
export IFACE=$(ip -o -4 addr show | grep "10.90.29." | awk '{print $2}')

if [ -z "$IFACE" ]; then
    echo "[ERROR] Could not find interface with 10.90.29.* network on $(hostname)"
    ip -o -4 addr show  # Show all interfaces for debugging
    exit 1
fi

echo "[INFO] Using IFACE=$IFACE on host $(hostname)"
ip -o -4 addr show "$IFACE"

export NCCL_SOCKET_IFNAME="$IFACE"
export GLOO_SOCKET_IFNAME="$IFACE"

# (keep InfiniBand disabled if IB isn’t properly configured)
export NCCL_IB_DISABLE=1

export TORCH_DISTRIBUTED_DEBUG=DETAIL


##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist        = " $SLURM_JOB_NODELIST
echo "Number of nodes = " $SLURM_JOB_NUM_NODES
echo "Ntasks per node = " $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo ""

# ---------- torchrun multi-node launch ----------
# Pick the first node as rendezvous/master
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

echo "[INFO] MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

# ---------- Optional network diagnostics ----------
DEBUG_NET=${DEBUG_NET:-1}   # set to 0 to skip tests once things work

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
MASTER_NODE=${NODES[0]}

echo "[NET] Nodes in this job: ${NODES[*]}"
echo "[NET] MASTER_NODE=${MASTER_NODE}, IFACE=${IFACE:-<unset>}"

NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
echo "[INFO] Using nproc_per_node=$NPROC_PER_NODE based on GPUs per node"

# Launch torchrun on ALL nodes / tasks via srun
srun python ${PROJECT_DIR}/scripts/train.py \
  --train_dna_npy ${PROJECT_DIR}/data/dna_data/train/train_oneHot.npy \
  --train_labels_npy ${PROJECT_DIR}/data/dna_data/train/train_labels.npy \
  --train_metadata_tsv ${PROJECT_DIR}/data/tf_data/metadata_tfbs.tsv \
  --val_dna_npy ${PROJECT_DIR}/data/dna_data/val/valid_oneHot.npy \
  --val_labels_npy ${PROJECT_DIR}/data/dna_data/val/valid_labels.npy \
  --val_metadata_tsv ${PROJECT_DIR}/data/tf_data/metadata_tfbs.tsv \
  --embedding_dir ${PROJECT_DIR}/data/tf_data/tf_embeddings_test \
  --epochs 20 \
  --batch_size 1024 \
  --num_workers 1 \
  --lr 1e-4 \
  --neg_fraction 0.015 \
  --wandb_project tfbind-train \
  --run_name tfbind_train_ddp \
  --output_dir ${PROJECT_DIR}/checkpoints/tfbind_train_ddp
# tfbind_pipeline.py
# ============================================================
#  COMPLETE TF-BINDING DATA PIPELINE 
#  Supports:
#       - TF alias mapping (only for specified TFs)
#       - Canonicalized embedding loading
#       - Exact alignment with label columns
#       - Positive + downsampled negative sampling
#       - Stable PyTorch Lightning DataModule
# ============================================================

import os
from pathlib import Path
from filelock import FileLock
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning.utilities import rank_zero_info


# ============================================================
# 1. TF ALIAS MAPPING (ONLY these should be canonicalized)
# ============================================================

TF_ALIAS_MAP = {
    "c-Fos": "c-Fos",
    "eGFP-FOS": "c-Fos",

    "GATA2": "GATA2",
    "eGFP-GATA2": "GATA2",

    "eGFP-JunB": "eGFP-JunB",
    "JunB": "eGFP-JunB",

    "UBTF": "UBF",
    "UBF": "UBF",

    "HA-E2F1": "E2F1",
    "E2F1": "E2F1",

    "GR": "GR",
    "NR3C1": "GR",

    "REST": "NRSF",
    "NRSF": "NRSF",
}


def canonical_name(tf):
    """Return canonical TF name ONLY if present in alias map."""
    return TF_ALIAS_MAP.get(tf, tf)


# ============================================================
# 2. EMBEDDING LOADING WITH ORDER MAPPING
# ============================================================

def load_embedding_index(embedding_dir):
    """
    Build mapping: canonical_name → embedding_file_path
    Handles filenames like:
        c-Fos_P01100_embedding.pt
        AP-2alpha_P05549_embedding.pt
        REST_Q12345_embedding.pt
    """
    emb_index = {}

    for fname in os.listdir(embedding_dir):
        if not fname.endswith(".pt"):
            continue

        # strip suffix
        core = fname.replace("_embedding.pt", "")   # e.g. "c-Fos_P01100"

        # split at first underscore
        tf_raw = core.split("_")[0]                 # -> "c-Fos"

        # convert to canonical
        tf_canon = canonical_name(tf_raw)           # alias mapping applied only if needed

        emb_index[tf_canon] = os.path.join(embedding_dir, fname)

    return emb_index


def load_tf_embeddings_in_label_order(tf_names, embedding_dir):
    """
    Load embeddings in same column order as label matrix.
    """
    emb_index = load_embedding_index(embedding_dir)

    tf_embs = []
    canon_names = []
    missing = []

    for tf in tf_names:
        canon = canonical_name(tf)
        canon_names.append(canon)

        if canon not in emb_index:
            missing.append((tf, canon))
            tf_embs.append(None)
            continue

        emb = torch.load(emb_index[canon], weights_only=True).float()
        if emb.ndim == 3:
            emb = emb.squeeze(0)

        tf_embs.append(emb)

    if missing:
        print("\n[WARNING] Missing TF embeddings:")
        for orig, canon in missing:
            print(f"  Label TF '{orig}' → Canonical '{canon}' has no embedding file")

    return tf_embs, canon_names



# ============================================================
# 3. POSITIVE + NEGATIVE DOWNSAMPLING
# ============================================================

def build_sample_indices(labels, neg_fraction=0.001, seed=2024):
    """
    Build list of (dna_idx, tf_idx) pairs:

      - include ALL positives
      - include a random subset of negatives
        (each negative is kept with probability = neg_fraction)

    This avoids ever storing all negatives in memory.
    """
    rng = np.random.default_rng(seed)
    N, T = labels.shape

    pos = []
    neg = []

    for i in range(N):
        row = labels[i]

        # ---- positives for this DNA window ----
        pos_js = np.where(row == 1)[0]
        for j in pos_js:
            pos.append((i, j))

        # ---- sampled negatives for this DNA window ----
        if neg_fraction > 0.0:
            # candidates where label == 0
            neg_js = np.where(row == 0)[0]
            if len(neg_js) > 0:
                # Bernoulli sampling per negative
                keep_mask = rng.random(len(neg_js)) < neg_fraction
                for j in neg_js[keep_mask]:
                    neg.append((i, j))

    pos = np.asarray(pos, dtype=np.int32)
    neg = np.asarray(neg, dtype=np.int32)

    all_pairs = np.concatenate([pos, neg], axis=0)
    rng.shuffle(all_pairs)

    print(f"[Sampling] Pos={len(pos)}, Neg={len(neg)}, Total={len(all_pairs)}")

    return all_pairs


# ============================================================
# 4. DATASET FOR TF–DNA COMBINATIONS
# ============================================================

class TFTargetDataset(Dataset):
    def __init__(self, dna_data, labels, sample_indices):
        self.dna_data = dna_data
        self.labels = labels
        self.samples = sample_indices

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dna_i, tf_j = self.samples[idx]

        dna = torch.from_numpy(self.dna_data[dna_i].copy())
        label = torch.tensor(self.labels[dna_i, tf_j], dtype=torch.float32)

        return dna, label, int(tf_j)

def prepad_tf_embeddings(tf_embs):
    clean_embs = []
    lengths = []

    for emb in tf_embs:
        if emb.ndim == 3:
            emb = emb.squeeze(0)
        emb = emb.float().contiguous()
        clean_embs.append(emb)
        lengths.append(emb.shape[0])

    padded = pad_sequence(
        clean_embs,
        batch_first=True,
        padding_value=0.0,
    )  # (num_tfs, Lmax, emb_dim)

    lengths = torch.tensor(lengths, dtype=torch.long)

    Lmax = padded.shape[1]
    mask = torch.arange(Lmax).unsqueeze(0) >= lengths.unsqueeze(1)

    return padded, mask, lengths

# ============================================================
# 5. COLLATE FUNCTION
# ============================================================


def tfbind_collate(batch):
    dna_batch = torch.stack([x[0] for x in batch])
    labels = torch.stack([x[1] for x in batch]).float()
    tf_idx = torch.tensor([x[2] for x in batch], dtype=torch.long)

    return dna_batch, labels, tf_idx


# ============================================================
# 6. PYTORCH LIGHTNING DATAMODULE
# ============================================================


class TFBindDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dna,
        train_labels,
        val_dna,
        val_labels,
        test_dna=None,
        test_labels=None,
        tf_embs=None,
        tf_embs_padded=None,
        tf_masks=None,
        train_pairs=None,
        val_pairs=None,
        test_pairs=None,
        batch_size=128,
        neg_fraction=0.01,
        num_workers=4,
    ):
        super().__init__()

        self.train_dna = train_dna
        self.train_labels = train_labels
        self.val_dna = val_dna
        self.val_labels = val_labels

        self.test_dna = test_dna
        self.test_labels = test_labels

        self.tf_embs = tf_embs
        self.tf_embs_padded = tf_embs_padded
        self.tf_masks = tf_masks

        self.train_pairs = train_pairs
        self.val_pairs = val_pairs
        self.test_pairs = test_pairs

        self.batch_size = batch_size
        self.neg_fraction = neg_fraction
        self.num_workers = num_workers
        

        self.tf_lengths = None
        
        if self.tf_embs_padded is None or self.tf_masks is None:
            if self.tf_embs is None:
                raise ValueError(
                    "Either provide tf_embs_padded/tf_masks from cache or provide raw tf_embs."
                )

            self.tf_embs_padded, self.tf_masks, self.tf_lengths = prepad_tf_embeddings(
                self.tf_embs
            )

    
    # -------------------------------------------------------
    
    def setup(self, stage=None):
        if stage in ("fit", None):
            if self.train_pairs is None:
                self.train_pairs = build_sample_indices(
                    self.train_labels,
                    neg_fraction=self.neg_fraction,
                )

            if self.val_pairs is None:
                self.val_pairs = build_sample_indices(
                    self.val_labels,
                    neg_fraction=0.08,
                )

            self.train_dataset = TFTargetDataset(
                self.train_dna,
                self.train_labels,
                self.train_pairs,
            )

            self.val_dataset = TFTargetDataset(
                self.val_dna,
                self.val_labels,
                self.val_pairs,
            )

        if stage in ("test", None):
            if self.test_dna is not None and self.test_labels is not None:
                if self.test_pairs is None:
                    self.test_pairs = build_sample_indices(
                        self.test_labels,
                        neg_fraction=1.0,
                    )

                self.test_dataset = TFTargetDataset(
                    self.test_dna,
                    self.test_labels,
                    self.test_pairs,
                )
            else:
                self.test_dataset = None

    

    # -------------------------------------------------------
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=tfbind_collate,
            prefetch_factor=2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=tfbind_collate,
            prefetch_factor=2,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=tfbind_collate,
            prefetch_factor=2,
        )


    '''
    #test train on 20 examples
    def setup(self, stage=None):

        # ---------------------------------------------------
        # Optional: Limit number of examples for debugging
        # ---------------------------------------------------
        if self.limit_examples is not None:
            n = self.limit_examples
            print(f"[DEBUG] Limiting train/val/test to first {n} examples (if available).")

            # train always exists
            self.train_dna = self.train_dna[:n]
            self.train_labels = self.train_labels[:n]

            # val always exists
            self.val_dna = self.val_dna[:n]
            self.val_labels = self.val_labels[:n]

            # test may be None → check before slicing
            if self.test_dna is not None:
                self.test_dna = self.test_dna[:n]
                self.test_labels = self.test_labels[:n]
            else:
                print("[DEBUG] No test set loaded (skipping test slicing).")

        # ---------------------------------------------------
        # Build (dna_idx, tf_idx) pairs
        # ---------------------------------------------------
        train_pairs = build_sample_indices(self.train_labels, neg_fraction=self.neg_fraction)
        val_pairs   = build_sample_indices(self.val_labels,   neg_fraction=0.5)

        # test_labels may be None → test_pairs should be empty
        if self.test_labels is not None:
            test_pairs  = build_sample_indices(self.test_labels, neg_fraction=1.0)
        else:
            test_pairs = []

        # ---------------------------------------------------
        # Build datasets
        # ---------------------------------------------------
        self.train_dataset = TFTargetDataset(self.train_dna, self.train_labels, train_pairs)
        self.val_dataset   = TFTargetDataset(self.val_dna,   self.val_labels,   val_pairs)

        if self.test_dna is not None:
            self.test_dataset = TFTargetDataset(self.test_dna, self.test_labels, test_pairs)
        else:
            self.test_dataset = None

    
'''   

def get_or_build_training_cache(
    cache_path,
    train_labels,
    val_labels,
    tf_names,
    embedding_dir,
    neg_fraction,
    seed=2024,
    force_rebuild=False,
):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    lock_path = str(cache_path) + ".lock"

    with FileLock(lock_path):
        if cache_path.exists() and not force_rebuild:
            rank_zero_info(f"Loading dataloader cache from: {cache_path}")
            return torch.load(cache_path, map_location="cpu", weights_only=False)

        rank_zero_info(f"Building dataloader cache: {cache_path}")

        tf_embs, canon_names = load_tf_embeddings_in_label_order(
            tf_names,
            embedding_dir,
        )

        tf_embs_padded, tf_masks, tf_lengths = prepad_tf_embeddings(tf_embs)

        train_pairs = build_sample_indices(
            train_labels,
            neg_fraction=neg_fraction,
            seed=seed,
        )

        val_pairs = build_sample_indices(
            val_labels,
            neg_fraction=0.08,
            seed=seed,
        )

        cache = {
            "train_pairs": train_pairs,
            "val_pairs": val_pairs,
            "tf_embs_padded": tf_embs_padded.cpu(),
            "tf_masks": tf_masks.cpu(),
            "tf_lengths": tf_lengths.cpu(),
            "canon_names": canon_names,
            "neg_fraction": neg_fraction,
            "seed": seed,
            "train_labels_shape": tuple(train_labels.shape),
            "val_labels_shape": tuple(val_labels.shape),
        }

        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        torch.save(cache, tmp_path)
        os.replace(tmp_path, cache_path)

        rank_zero_info(f"Saved dataloader cache to: {cache_path}")
        return cache
# MetagenBERT

Research pipeline for disease prediction from metagenomic DNA sequences, based on:
- DNABERT embeddings (`embedding-dna2-ordered.py`, `embedding-dnaS-ordered.py`)
- multi-GPU FAISS clustering (`faiss_clustering_gpu*.py`)
- abundance / aggregation extraction (`compute_abundance.py`, `aggregation.py`, `combined_abundances.py`)
- DeepSets classification (`DeepSets.py`)

## Project Status

This repository is an HPC-oriented experimental prototype (absolute paths, dataset-specific scripts, limited CLI parameterization).  
It is useful to reproduce internal experiments, but it requires adaptation for general-purpose usage.

## Repository Structure

- `embedding-dna2-ordered.py`: distributed embedding (DDP + `mp.spawn`) for DNABERT-2
- `embedding-dnaS-ordered.py`: distributed embedding (DDP) for DNABERT-S
- `aggregation.py`: weighted mean from `.pt` embedding chunks to `.npy`
- `faiss_clustering_gpu.py`: per-sample FAISS clustering
- `faiss_clustering_gpu_global.py`: global FAISS clustering (cross-validation)
- `faiss_clustering_gpu_global_optimized.py`: global clustering over multiple cluster sizes
- `compute_abundance.py`: abundance computation from cluster assignments
- `subsample_embeddings.py`: embedding subsampling
- `combined_abundances.py`: L1 logistic regression baseline on abundances
- `DeepSets.py`: DeepSets training for binary classification

## Requirements

- Python 3.10+
- CUDA GPU (for embedding / FAISS scripts)
- Distributed environment (NCCL/torch.distributed variables) for DDP scripts
- Python dependencies:
  - `numpy`
  - `torch`
  - `transformers`
  - `scikit-learn`
  - `pandas`
  - `faiss-gpu`
  - `joblib`
  - `idr_torch` (IDRIS cluster environment; required by some scripts)

Minimal install example (adapt to your CUDA stack):

```bash
pip install numpy torch transformers scikit-learn pandas joblib
# FAISS GPU and idr_torch depend on your cluster/infrastructure
```

## Expected Data Layout

The scripts assume a directory layout like:

```text
dataset_root/
  sample_A/
    mean/
      embeddings_*.pt
    idx/
      idx_*.pt
  sample_B/
    ...
```

Several scripts also assume specific filename patterns (name parsing is used in FAISS scripts).

## Typical Pipeline

### 1) Generate embeddings

DNABERT-2 (local multi-process):

```bash
python embedding-dna2-ordered.py \
  <tokenizer_path> <config_path> <model_path> \
  <sequence_dir> <saving_path> <to_avoid> <batch_size> <world_size>
```

DNABERT-S (DDP in cluster environment):

```bash
python embedding-dnaS-ordered.py \
  <tokenizer_path> <config_path> <model_path> \
  <sequence_dir> <saving_path> <to_avoid> <batch_size>
```

### 2) Aggregate embeddings (mean)

```bash
python aggregation.py <samples_dir> <save_path>
```

### 3) FAISS clustering

Per-sample:

```bash
python faiss_clustering_gpu.py \
  <data_dir> <save_path> <n_clusters> <n_iter> <min_points> <max_points> <nb_file_batch>
```

Global:

```bash
python faiss_clustering_gpu_global.py \
  <data_path> <save_path> <n_clusters> <n_iter> <verbose> <min_points> <max_points> \
  <use_perc> <file_of_lens> <raw> <perc> <nb_batch>
```

### 4) Compute abundances

```bash
python compute_abundance.py <numbers_dir> --processes 32
```

### 5) Train prediction models

DeepSets:

```bash
python DeepSets.py
```

L1 logistic baseline:

```bash
python combined_abundances.py
```

## Current Limitations

- Many absolute paths (`/data/...`) are hardcoded.
- DDP scripts assume an already configured distributed runtime.
- Limited I/O safeguards (empty folders, unexpected formats, etc.).
- No automated tests are included in this repository.

## Recommendations for Production-Readiness

1. Centralize configuration (YAML + consistent argparse usage).
2. Replace hardcoded paths with explicit parameters.
3. Add a minimal CPU/single-GPU mode for local debugging.
4. Add smoke tests (I/O, parsing, short training pass).
5. Add a versioned `requirements.txt` or `environment.yml`.

# TriBSC

**Gene-anchored multimodal biological integration with complementary network evidence for miRNA-disease association prediction**

This repository provides the model code, experiment scripts, and precomputed data for TriBSC, a framework for predicting potential miRNA-disease associations. The available experiments include five-fold cross-validation on HMDD v3.2 and HMDD v4.0, case studies, branch-level ablation studies, and strict disease and miRNA cold-start evaluations.

## Model Overview

TriBSC combines gene-anchored multimodal biological information with complementary association-network evidence. It scores candidate miRNA-disease pairs through three branches.

| Branch | Name | Main Role |
| --- | --- | --- |
| A | Gene-anchored multimodal biological interaction branch | Integrates RiNALMo-derived miRNA sequence representations, BioMedBERT-derived disease text representations, and gene functional context through prior-aware attention, gated fusion, and cross-gene interactions. |
| B | Explicit structural expert | Predicts associations using structural features from the training association network, including paths, similarity propagation, and node degrees. |
| C | Low-rank multi-similarity completion branch | Combines low-rank reconstructions at multiple ranks, multi-source similarity propagation, and auxiliary features to capture global information in the association network. |

Branch predictions are rank-normalized and combined through weighted fusion. Fusion weights are selected on the inner validation set. Network features that depend on miRNA-disease associations are constructed from the corresponding training data.

## Directory Structure

Place this README in the same directory as `main_01.py`. Preserve the directory hierarchy, filenames, and entity ordering in the data files shown below.

```text
.
├── README.md
├── main_01.py
├── run_main01_case_study.py
├── strict_three_branch_case_study.py
├── run_main01_paper_six_variants.py
├── run_main01_strict_cold_start.py
├── data/
│   ├── miRNA-disease association.txt
│   ├── miRNA_sequence_similarity.txt
│   ├── disease_semantic_similarity.txt
│   ├── disease_feature_vector_similarity_matrix.txt
│   ├── mirna-list.xlsx
│   └── disease name.xlsx
├── D14_UnifiedDataCache/
│   ├── benchmark_legacy.npz
│   ├── entity_names.json
│   └── manifest.json
├── Bio_Foundation_Cache/
│   ├── rna_unique_embeddings.npy
│   ├── mirna_group_indices.npy
│   ├── mirna_group_mask.npy
│   ├── disease_embeddings.npy
│   ├── mirna_names.json
│   ├── disease_names.json
│   └── d6_gene_bridge_cache.npz
├── D16_BioHeteroPath_Cache/
│   └── new_model_external_context.npz
└── HMDDV4.0/
    ├── main_01.py
    ├── data/
    ├── D14_UnifiedDataCache/
    ├── Bio_Foundation_Cache/
    └── D16_BioHeteroPath_Cache/
```

`strict_three_branch_case_study.py` contains the pipeline required by the case-study runner. Keep it in the same directory as `run_main01_case_study.py` and `main_01.py`.

## Data and Caches

The root-level data and caches correspond to HMDD v3.2: 788 miRNAs, 374 diseases, and 8,968 known associations. The separate `HMDDV4.0/` directory contains the HMDD v4.0 experiment: 1,219 miRNAs, 1,490 diseases, and 21,996 known associations. Each dataset has its own `D14_UnifiedDataCache/benchmark_legacy.npz` and matching feature caches. Keep each dataset's data and caches together; do not mix files between the two datasets.

| Directory | Contents and Purpose |
| --- | --- |
| `data/` | Contains association and similarity matrices and entity-name tables. The current main model reads miRNA sequence similarity and disease semantic similarity from this directory, while task labels are loaded from `benchmark_legacy.npz`. |
| `D14_UnifiedDataCache/` | Provides unified task labels and entity-alignment information. This distribution includes the association-matrix cache, entity names, and cache manifest to maintain consistent entity ordering across data sources. |
| `Bio_Foundation_Cache/` | Stores pre-trained miRNA sequence and disease text features extracted offline, together with entity mappings and validity masks. `d6_gene_bridge_cache.npz` stores gene-level overlap features; the current `main_01.py` does not read this file directly. |
| `D16_BioHeteroPath_Cache/` | Stores gene functional context constructed from miRNA-gene, disease-gene, GO, Pathway, and PPI relations, together with gene indices, validity masks, and relation prior weights for miRNAs and diseases. |
| `HMDDV4.0/` | Contains a separate main-model entry point, data, and all three cache directories for five-fold cross-validation on HMDD v4.0. |

The experiments load precomputed caches directly, so RiNALMo and BioMedBERT representations do not need to be extracted again at runtime. The model checks that the required cache files are present, entity-name ordering is consistent, and the gene-context cache is explicitly marked as not using miRNA-disease labels.

## Environment and Dependencies

The experimental environment is listed below.

| Component | Version |
| --- | --- |
| Operating system | Ubuntu 22.04 |
| Python | 3.12 |
| PyTorch | 2.8.0 |
| CUDA | 12.8 |
| NumPy | [2.4.6](https://pypi.org/project/numpy/2.4.6/) |
| SciPy | [1.18.1](https://pypi.org/project/scipy/1.18.1/) |
| scikit-learn | [1.9.0](https://pypi.org/project/scikit-learn/1.9.0/) |
| Matplotlib | [3.11.1](https://pypi.org/project/matplotlib/3.11.1/) |

Install the numerical computing and plotting dependencies in a Python 3.12 environment:

```bash
python -m pip install numpy==2.4.6 scipy==1.18.1 scikit-learn==1.9.0 matplotlib==3.11.1
```

For an NVIDIA GPU environment that supports CUDA 12.8, install PyTorch with:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

See the [official PyTorch instructions for previous versions](https://pytorch.org/get-started/previous-versions/) for installation details. The code uses a GPU when CUDA is available and otherwise falls back to the CPU. CPU runtime depends on the hardware.

To read HMDD Excel files or export a combined Excel workbook in the case study, also install:

```bash
python -m pip install pandas openpyxl
```

`pandas` and `openpyxl` are required for these Excel features. Their versions were not specified in the original environment notes. The case study can still produce CSV and JSON results without these features.

## Path Setup

Open a terminal and change to the repository root, which contains `main_01.py`. Unless otherwise noted, run the experiment commands below from that directory.

The current default root-directory settings in the root-level `main_01.py` are:

```python
SCRIPT_DIR = Path(__file__).resolve().parent
SERVER_ROOT = Path("/root/autodl-tmp") if os.name != "nt" else SCRIPT_DIR
```

On Windows, caches are therefore read from the script directory by default. On Linux, the default cache root is `/root/autodl-tmp`. If the project is stored elsewhere on Linux and you want to use the directory structure above directly, change the assignment of `SERVER_ROOT` to:

```python
SERVER_ROOT = SCRIPT_DIR
```

The main model, case study, and cold-start experiments will then read the cache directories located alongside the scripts.

The cold-start runner also supports `--project-root .` to use the current directory without changing the main model's root-directory setting. The basic commands for the main model and case study still require the path setup described above.

The branch-level ablation runner uses its own script directory as the default project root. The separate `HMDDV4.0/main_01.py` already uses `SERVER_ROOT = SCRIPT_DIR` and reads the data and caches inside `HMDDV4.0/`; no path edit is needed for these two entry points.

## Running the Experiments

The experiment entry points can be run independently. You do not need to complete the main model's five-fold experiment before running the other entry points. The experiment scripts invoke the model training and evaluation pipelines. The root-level scripts use HMDD v3.2 by default; the separate HMDD v4.0 experiment is described in Section 5 below.

### 1. Main Model: Five-Fold Cross-Validation on HMDD v3.2

```bash
python main_01.py
```

The default random seed is `10`, and evaluation uses five-fold cross-validation. All known associations are used as positive samples, and an equal number of unrecorded associations are randomly sampled as negative samples. Within each fold, the outer training set is further split to create an inner validation set for selecting training epochs and fusion weights.

The main evaluation metrics are AUC, AUPR, Accuracy, Precision, Recall, and F1.

### 2. Case Studies

```bash
python run_main01_case_study.py
```

By default, case studies are conducted for `Colonic Neoplasms` and `Liver Neoplasms`, with the top 15 novel candidate miRNAs reported for each disease. Known training associations are saved separately and excluded from novel-candidate rankings. Retrospective experiments that hold out known associations are also enabled by default.

To change the target diseases or the number of candidates, run:

```bash
python run_main01_case_study.py --targets "Colonic Neoplasms" "Liver Neoplasms" --topk 20
```

To check supporting evidence against external HMDD or dbDEMC data, supply the corresponding files separately using `--hmdd_xlsx` or `--dbdemc_txt`. These external evidence files are not included in the current data package. HMDD is used for post-hoc corroboration of candidate associations; the dbDEMC input is used only to report disease-level dataset coverage. `FusionRankScore` is a fused ranking score within the candidate pool, not a calibrated association probability.

### 3. Strict Cold-Start Evaluation

```bash
python run_main01_strict_cold_start.py
```

By default, both disease and miRNA cold-start evaluations are performed. Training and test sets are split by whole entities, and inner validation is also performed using entity-level splits. Associations involving test entities are excluded when constructing the corresponding training-network features. The experiments compare the full model with a variant that omits biological Branch A.

To run only one mode, use the corresponding command:

```bash
python run_main01_strict_cold_start.py --modes disease
python run_main01_strict_cold_start.py --modes mirna
```

### 4. Branch-Level Ablation Studies

```bash
python run_main01_paper_six_variants.py
```

This runner evaluates the six branch-level variants reported in the paper, together with the full TriBSC reference, for seven groups in total.

| Variant | Retained Branches |
| --- | --- |
| Biological-only | A |
| Structural-only | B |
| Completion-only | C |
| w/o Biological | B + C |
| w/o Structural | A + C |
| w/o Completion | A + B |

All groups use the same sampled pairs, five-fold splits, and inner validation splits. Each unchanged branch is trained once per fold, and its predictions are reused across variants. Fusion weights are reselected on the inner validation set using only the retained branches. This runner does not perform internal component ablations of Branch A.

### 5. Main Model: Five-Fold Cross-Validation on HMDD v4.0

From the repository root, enter the HMDD v4.0 experiment directory and run its main model:

```bash
cd HMDDV4.0
python main_01.py
cd ..
```

This entry point runs five-fold cross-validation on the supplied HMDD v4.0 dataset using the data and caches within `HMDDV4.0/`. It uses the same evaluation protocol and metrics as the HMDD v3.2 main-model experiment. Keep the HMDD v4.0 directory structure intact when uploading or moving the project.

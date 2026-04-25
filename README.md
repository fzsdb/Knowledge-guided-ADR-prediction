# Informative Relational Learning for Adverse Reaction Prediction with Enhanced Generalization to Novel Drugs

This repository implements a hierarchical and relation-aware framework for adverse drug reaction (ADR) prediction. The method jointly models ADR semantic structure and real-world co-occurrence patterns using a relational graph, while leveraging the ATC hierarchy to construct structured drug relations for improved knowledge transfer to novel drugs. To further enhance robustness under distribution shifts, a conditional domain adversarial module is employed. In addition, a Dual Mixture-of-Experts architecture captures both category-specific and global ADR patterns, enabling stable and accurate prediction, especially for novel drugs and uncommon ADRs.

---

## Table of Contents

- [Two Experimental Settings](#two-experimental-settings)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Running the Code](#running-the-code)
- [Training Output](#training-output)
- [Troubleshooting](#troubleshooting)

---

## Two Experimental Settings

| | KDS — Known Drug Setting | NDS — Novel Drug Setting |
|---|---|---|
| **Code location** | `AAKD/` | `Standard Setting/` |
| **Test drugs** | Seen during training | Unseen during training |
| **Drug features** | Morgan + GIN | Morgan + GIN + MolFormer |
| **Domain adaptation** | No | Yes (CDAN) |

---

## Repository Structure

```
├── AAKD/                        # KDS implementation
│   ├── main.py                
│   ├── config_file.py        
│   ├── models_file.py        
│   ├── graph_construction.py   
│   ├── data_loader_file.py    
│   └── train_eval_file.py      
│
├── Standard Setting/            # NDS implementation
│   ├── main.py                  
│   ├── config.py                
│   ├── models_standard.py       
│   ├── cdan_modules.py         
│   ├── graph_construction.py    
│   ├── data_loader.py          
│   ├── molecular_features.py   
│   ├── train_eval.py            
│   └── utils_file.py           
│
├── requirements.txt
└── README.md
```


---

## Requirements

- Python 3.11
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.3.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- scikit-learn >= 1.3.0
- RDKit >= 2023.3.1

---

## Installation

```bash
# 1. Create and activate environment
conda create -n drug-adr python=3.11
conda activate drug-adr

# 2. Install PyTorch (adjust cuda version as needed, see https://pytorch.org)
pip install torch>=2.0.0

# 3. Install PyTorch Geometric (see https://pyg.org/nn/install.html)
pip install torch-geometric

# 4. Install all other dependencies
pip install -r requirements.txt
```

---

## Data Preparation

> **How to generate MolFormer embeddings:** Use the pretrained MolFormer model from the official [IBM/molformer](https://github.com/IBM/molformer) GitHub repository. Pre-trained checkpoints (~100M molecules) can be downloaded from [ibm.box.com/v/MoLFormer-data](https://ibm.box.com/v/MoLFormer-data).

---

## Running the Code

python main.py


---

## Training Output

Progress is printed to stdout every epoch:

```
Time: 12.34s | Epoch: 5 | LR: 0.000200
[Train] Loss: 0.4821 | LB Loss: 0.0032 | F1: 0.7213 | ROC-AUC: 0.8541 | PR-AUC: 0.8102 | ...
[Test]  Loss: 0.5103 | F1: 0.6987 | ROC-AUC: 0.8312 | PR-AUC: 0.7891 | ...
----------------------------------------------------------------------------------------------------
✓ Model saved at epoch 5 with Test F1: 0.6987, ROC-AUC: 0.8312
```

After training completes, learned graph embeddings are exported:

- `adr_embeddings_learned.csv` — R-GCN embeddings for ADR nodes
- `atc_embeddings_learned.csv` — R-GCN embeddings for ATC level-5 nodes

---

## Troubleshooting

**`Warning: Invalid ATC code format: ...`**
ATC codes must be exactly 7 characters (e.g., `B01AC06`). Codes of other lengths are silently ignored and the corresponding drug will be routed to the fallback expert.

**`Warning: Invalid ADReCS ID format: ...`**
ADReCS IDs must follow the four-level dot-separated format (e.g., `xx.xx.xx.xxx`). IDs not matching this pattern are skipped during graph construction.

**Slow training on CPU**
The model is designed to run on GPU. On CPU, consider reducing `--epochs` for a quick sanity check, or set `num_workers=0` (already the default for non-CUDA devices).

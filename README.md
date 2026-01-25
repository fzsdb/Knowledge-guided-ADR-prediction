# ADR-prediction-framework
This repository implements a hierarchical and relation-aware framework for adverse drug reaction (ADR) prediction. The method jointly models ADR semantic structure and real-world co-occurrence patterns using a relational graph, while leveraging the ATC hierarchy to construct structured drug relations for improved knowledge transfer to novel drugs. To further enhance robustness under distribution shifts, a conditional domain adversarial module is employed. In addition, a Dual Mixture-of-Experts architecture captures both category-specific and global ADR patterns, enabling stable and accurate prediction, especially for novel drugs and uncommon ADRs.

## Create Virtual Environment
conda create -n drug-adr python=3.11.13 

conda activate drug-adr

## Install Dependencies
pip install -r requirements.txt

## Usage
python main.py

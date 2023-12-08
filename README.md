# CKG-IMC: an inductive matrix completion method enhanced by collaborative knowledge graph and graph neural network for Alzheimer’s disease compound-protein interactions prediction
----
## Introduction
---
Alzheimer’s disease (AD) is a prevalent neurodegenerative disorder without effective therapeutic interventions. To address this challenge, we present CKG-IMC, a deep learning model for predicting compound-protein interactions (CPIs) relevant to AD. CKG-IMC integrates three modules: a Collaborative Knowledge Graph (CKG), a Principal Neighborhood Aggregation Graph Neural Network (PNA), and an Inductive Matrix Completion (IMC).
![](./src/fig.jpg)

## Data
---
Please see the [README](./data/README.md) for detailed explanation.

## Environment Requirement
---
The code has been tested running under Python 3.8 and 3.9.18. The required packages are as follows:
```
numpy>=1.25.0
pandas>=2.0.3
scikit-learn>=1.3.0
torch>=2.0.1
torch_geometric>=2.3.1
tqdm>=4.65.0
```
## Usage
---
To perform 10-fold cross-validation, run the following command:
```bash
python main.py -adv
```

For predicting compound-protein interactions (CPIs), use the following command:
```bash
python main.py -adv --do_predict
```
For displays the help message, use the following command:
```bash
python main.py -h
```

## Citation
---
If you find this work useful, please consider citing our paper.


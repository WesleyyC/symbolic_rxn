# Symbolic RXN

Integrating Deep Neural Networks and Symbolic Inference for Organic Reactivity Prediction [[ChemRxiv]](https://doi.org/10.26434/chemrxiv.11659563)

Code for the inference pipeline is here, and the complete training pipeline will be updated once the paper is accepted.

## System Requirements

- `conda=4.7.10` (from [here](https://www.anaconda.com/distribution/) with Python 3.6.9)
- `tensorflow-gpu=1.12.0` (from conda, and GPU is recommended)
- `rdkit=2019.09.1` (from conda)
- `gurobi=8.1.1` (from conda, and free academic licence can be found [here](https://www.gurobi.com/academia/academic-program-and-licenses/))
- `h5py=2.9.0`(from conda)
- other softwares required by the above packages

## Installation Guide

Follwoings are the intructions for installing our software and the dependencies specified above:

```bash
# the total installation time should be less than 15 minutes

# download the source code
git clone https://github.com/WesleyyC/symbolic_rxn.git
cd symbolic_rxn

# create a py3 conda env
conda create -n symbolic_rxn python=3
conda activate symbolic_rxn  

# install the dependencies
# tensorflow-gpu
conda install tensorflow-gpu=1.12.0
# RDKit
conda install -c conda-forge rdkit=2019.09.1
# gurobi
conda config --add channels http://conda.anaconda.org/gurobi
conda install gurobi=8.1.1
```

## Data Preprocessing

To better balance the GPU/CPU workload during training/inference, we precompute the graph features and store them in HDF5 format for fast random query. Followings are the instructions for digesting the raw text input into our graph feature representation.

```bash
# prepare the data
unzip data.zip

# digesting a small demo dataset w/ 2000 reactions (~30 secs)
python -m reactivity_prediction.data_digestion --input data/demo.txt --output data/hdf5_demo 

# digesting the full test dataset (~10 mins)
python -m reactivity_prediction.data_digestion --input data/test.txt --output data/hdf5_test
```

## Downloading the Pre-Trained Models

We provided the pre-trained models for reproducing our Top-K performance on the USPTO dataset:

```bash
# download the ensembles models
wget https://www.dropbox.com/s/x8ruovxxsc5q7q6/ckpt.zip

# unzip the models
unzip ckpt.zip
```

## Running a Sample Pre-trained Model on the Small Demo Dataset

Followings are the instructions for running a single pre-trained model on the provided demo dataset:

```bash
# reactivity prediction w/ existing ckpt (<1 mins with GPU)
python -m reactivity_prediction.run_model \
  --mode infer --ckpt ckpt/mdl_0 \
  --eval_input data/hdf5_demo --eval_output demo_output

# evaluate reactivity prediction (<30 secs)
python -m reactivity_prediction.eval --input demo_output/delta_predictions.pkl
# and the program will report the bond changes coverage:
# ==========================================================
# ================ Reaction Bond Prediction ================
# ==========================================================
# as well as the delta prediction accuracy/F-1 socre:
# ==========================================================
# ==================== Delta Evaluation ====================
# ==========================================================

# run symbolic inference and evaluation (~30 mins with multi cores)
python -m octet_sampling.run_sampler --input demo_output/delta_predictions.pkl 
# and the program will report in the last row with the Top-K prediction accuracy:
# ==================================== #2000 of Reactions Evaluated ====================================
# Gurobi: Top1: ... Top2: ... Top3: ... Top5: ... Top20: ... Average Time: ...
```

## Reproducing the Full USPTO Test Dataset with the Ensemble Models:

Followings are the instruction for reproducing our reprorted Top-k prediction accuracy on the USPTO test dataset using the ensemble models:

```bash
# running the following script to genereate ensembled predictions (~80 mins or ~10 mins/model)
bash reactivity_prediction/run_ensemble_inference.sh

# evaluate reactivity prediction (<10 mins)
python -m reactivity_prediction.eval --input ensemble_output/delta_predictions.pkl

# run symbolic inference and evaluation (<1 day)
python -m octet_sampling.run_sampler --input ensemble_output/delta_predictions.pkl
```

## Contact

- Wesley Wei Qian (weiqian3@illinois.edu)

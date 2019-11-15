# symbolic rxn

Code accompanied with our submission for `Integrating Deep Neural Networks and Symbolic Inference for Organic Reactivity Prediction`

### Installing Dependencies

```bash
# create conda env
conda create -n symbolic_rxn python=3

# install library
# tensorflow-gpu
conda install tensorflow-gpu=1.12.0
# RDKit
conda install -c conda-forge rdkit=2019.09.1
# gurobi (license required)
conda config --add channels http://conda.anaconda.org/gurobi
conda install gurobi=8.1.1
```

### Preparing the Data

```bash
# prepare the data
unzip data.zip

# preprocess the data into HDF5 for the model
python -m reactivity_prediction.data_digestion --input data/test.txt --output data/hdf5_test
```

### Preparing the Pre-Trained Model

```bash
# download the ensembles model
wget https://www.dropbox.com/s/5s3cqyfc775ytm2/ckpt.zip
# unzip the model
unzip ckpt.zip
```

### Running Single Model

```bash
# reactivity prediction w/ existing ckpt
python -m reactivity_prediction.run_model \
  --mode infer --ckpt ckpt/mdl_0 \
  --eval_input data/hdf5_test --eval_output mdl_0_output

# evaluate reactivity prediction
python -m reactivity_prediction.eval --input mdl_0_output/delta_predictions.pkl

# run symbolic inference and evaluation
python -m octet_sampling.run_sampler --input mdl_0_output/delta_predictions.pkl 
```

### Running Ensemble Model

```bash
# running the following script to genereate ensembled predi
bash reactivity_prediction/run_ensemble_inference.sh

# evaluate reactivity prediction
python -m reactivity_prediction.eval --input ensemble_output/delta_predictions.pkl

# run symbolic inference and evaluation
python -m octet_sampling.run_sampler --input ensemble_output/delta_predictions.pkl
```

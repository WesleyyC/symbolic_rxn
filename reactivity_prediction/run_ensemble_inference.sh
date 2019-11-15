#!/usr/bin/env bash

# ensemble inference

python -m reactivity_prediction.run_model --mode infer --ckpt ckpt/mdl_0 --eval_input data/hdf5_test --eval_output mdl_0_output

python -m reactivity_prediction.run_model --mode infer --ckpt ckpt/mdl_1 --eval_input data/hdf5_test --eval_output mdl_1_output

python -m reactivity_prediction.run_model --mode infer --ckpt ckpt/mdl_2 --eval_input data/hdf5_test --eval_output mdl_2_output

python -m reactivity_prediction.run_model --mode infer --ckpt ckpt/mdl_3 --eval_input data/hdf5_test --eval_output mdl_3_output

python -m reactivity_prediction.run_model --mode infer --ckpt ckpt/mdl_4 --eval_input data/hdf5_test --eval_output mdl_4_output

python -m reactivity_prediction.run_model --mode infer --ckpt ckpt/mdl_5 --eval_input data/hdf5_test --eval_output mdl_5_output

python -m reactivity_prediction.run_model --mode infer --ckpt ckpt/mdl_6 --eval_input data/hdf5_test --eval_output mdl_6_output

python -m reactivity_prediction.run_model --mode infer --ckpt ckpt/mdl_7 --eval_input data/hdf5_test --eval_output mdl_7_output

# average ensemble

python -m reactivity_prediction.ensemble_pkl \
  --inputs mdl_0_output/delta_predictions.pkl,mdl_1_output/delta_predictions.pkl,mdl_2_output/delta_predictions.pkl,mdl_3_output/delta_predictions.pkl,mdl_4_output/delta_predictions.pkl,mdl_5_output/delta_predictions.pkl,mdl_6_output/delta_predictions.pkl,mdl_7_output/delta_predictions.pkl \
  --output ensemble_output/delta_predictions.pkl

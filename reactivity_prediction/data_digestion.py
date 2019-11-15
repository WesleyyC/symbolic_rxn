"""Digest the .txt data into HDF5 format."""

import argparse
import os

import h5py
import numpy as np
from rdkit import RDLogger

from input_parsing.parse_reaction import get_reactivity_prediction_features_labels
from input_parsing.util import ATOM_FEATURES_KEY, BOND_FEATURES_KEY, NEIGHBOR_ATOM_KEY, NEIGHBOR_BOND_KEY, \
    NEIGHBOR_MASK_KEY, EDGE_DELTA_KEY, H_DELTA_KEY, C_DELTA_KEY, FEATURE_KEY, LABEL_KEY
from reactivity_prediction.io import read_txt_in_bins, batch_reactivity_prediction_tensor_dict_list, ATOM_MASK_KEY, \
    INPUT_HDF5_FILE_KEY, METADATA_KEY, METADATA_BIN_LIST_KEY, METADATA_WEIGHT_KEY, METADATA_BIN_SIZE_KEY, \
    HDF5_FORMAT_KEY, REACTION_STR_KEY, EDIT_STR_KEY

FEATURES = [ATOM_FEATURES_KEY, ATOM_MASK_KEY, BOND_FEATURES_KEY, NEIGHBOR_ATOM_KEY, NEIGHBOR_BOND_KEY,
            NEIGHBOR_MASK_KEY]
LABELS = [EDGE_DELTA_KEY, H_DELTA_KEY, C_DELTA_KEY]
FLOAT_KEYS = {ATOM_FEATURES_KEY}

parser = argparse.ArgumentParser(description='Reactivity Prediction Data Digestion')
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--format', choices=[HDF5_FORMAT_KEY], default=HDF5_FORMAT_KEY)
args = parser.parse_args()

lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)


def data_digestion_in_hdf5(in_f, out_f):
    hdf5_dataset = h5py.File(os.path.join(out_f, INPUT_HDF5_FILE_KEY), "w")

    input_in_bins = read_txt_in_bins(in_f)
    bins_list = sorted(list(input_in_bins.keys()))
    bins_size = np.array([len(input_in_bins[bin]) for bin in bins_list])
    bins_weight = bins_size / np.sum(bins_size)
    hdf5_dataset.create_dataset('{}/{}'.format(METADATA_KEY, METADATA_BIN_LIST_KEY),
                                data=np.array(bins_list).astype(np.int32))
    hdf5_dataset.create_dataset('{}/{}'.format(METADATA_KEY, METADATA_BIN_SIZE_KEY),
                                data=np.array(bins_size).astype(np.int32))
    hdf5_dataset.create_dataset('{}/{}'.format(METADATA_KEY, METADATA_WEIGHT_KEY),
                                data=bins_weight.astype(np.float32))

    for bin in bins_list:
        print('Processing Bucket Size {} with {} reactions.'.format(bin, len(input_in_bins[bin])))
        tensor_dict_list = [get_reactivity_prediction_features_labels(*str_tuple) for str_tuple in input_in_bins[bin]]
        batched_tensor = batch_reactivity_prediction_tensor_dict_list(tensor_dict_list, train_mode=True)
        hdf5_dataset.create_dataset('{}/{}'.format(bin, REACTION_STR_KEY),
                                    data=np.array(batched_tensor[REACTION_STR_KEY], dtype=h5py.special_dtype(vlen=str)))
        hdf5_dataset.create_dataset('{}/{}'.format(bin, EDIT_STR_KEY),
                                    data=np.array(batched_tensor[EDIT_STR_KEY], dtype=h5py.special_dtype(vlen=str)))
        for f_key in FEATURES:
            if f_key in FLOAT_KEYS:
                hdf5_dataset.create_dataset('{}/{}/{}'.format(bin, FEATURE_KEY, f_key),
                                            data=batched_tensor[FEATURE_KEY][f_key].astype(np.float32))
            else:
                hdf5_dataset.create_dataset('{}/{}/{}'.format(bin, FEATURE_KEY, f_key),
                                            data=batched_tensor[FEATURE_KEY][f_key].astype(np.int8))
        for l_key in LABELS:
            if l_key in FLOAT_KEYS:
                hdf5_dataset.create_dataset('{}/{}/{}'.format(bin, LABEL_KEY, l_key),
                                            data=batched_tensor[LABEL_KEY][l_key].astype(np.float32))
            else:
                hdf5_dataset.create_dataset('{}/{}/{}'.format(bin, LABEL_KEY, l_key),
                                            data=batched_tensor[LABEL_KEY][l_key].astype(np.int8))

    print('Finish HDF5 Data Digestion for {}.'.format(in_f))
    print('HDF5 Data Path is {}.'.format(out_f))


if __name__ == "__main__":
    if args.format == HDF5_FORMAT_KEY:
        os.makedirs(args.output, exist_ok=True)
        data_digestion_in_hdf5(args.input, args.output)
    else:
        raise ValueError('Does not support format: {}'.format(args.format))

"""io util for running the model."""

import os
import pickle
import random

import h5py
import numpy as np
from rdkit import Chem

from input_parsing.util import count_heavy_atom, FEATURE_KEY, ATOM_FEATURES_KEY, BOND_FEATURES_KEY, NEIGHBOR_ATOM_KEY, \
    NEIGHBOR_BOND_KEY, NEIGHBOR_MASK_KEY, LABEL_KEY, EDGE_DELTA_KEY, H_DELTA_KEY, C_DELTA_KEY, OCTET_SUM_KEY, \
    EDGE_DELTA_VAL_LIST, H_DELTA_VAL_LIST, C_DELTA_VAL_LIST, REACTION_STR_KEY, \
    EDIT_STR_KEY

INPUT_BINS = [10, 20, 30, 40, 50, 60, 80, 100, 120, 150]

ATOM_MASK_KEY = 'atom_mask'

HDF5_FORMAT_KEY = 'hdf5'
TXT_FORMAT_KEY = 'txt'
PKL_FORMAT_KEY = 'pkl'
INPUT_HDF5_FILE_KEY = 'data.hdf5'
OUTPUT_HDF5_FILE_KEY = 'delta_predictions.hdf5'
OUTPUT_DELTA_PREDICTION_PKL_FILE_KEY = 'delta_predictions.pkl'

CKPT_FILE_KEY = 'model.ckpt'
CKPT_FINAL_FILE_KEY = 'model.final'

METADATA_KEY = 'metadata'
METADATA_BIN_LIST_KEY = 'bin_list'
METADATA_BIN_SIZE_KEY = 'bin_size'
METADATA_WEIGHT_KEY = 'bin_weight'

OUTPUT_STR_TUPLE_KEY = 'str_tuple'
OUTPUT_REACTION_STR_KEY = 'reaction_str'
OUTPUT_NUM_ATOM_KEY = 'num_atom'
OUTPUT_EDIT_STR_KEY = 'edit_str'
OUTPUT_EDGE_DELTA_KEY = 'edge_delta_pred'
OUTPUT_H_DELTA_KEY = 'h_delta_pred'
OUTPUT_C_DELTA_KEY = 'c_delta_pred'
OUTPUT_ATTENTION_SCORE = 'attention_score'


def remap_atom_to_smile_idx(reaction_str, edit_str):
    reactant_str, product_str = reaction_str.split('>>')
    reactant_mol = Chem.MolFromSmiles(reactant_str)
    product_mol = Chem.MolFromSmiles(product_str)
    remap = {}
    idx = 1
    for atom in reactant_mol.GetAtoms():
        remap[atom.GetIntProp('molAtomMapNumber')] = idx
        atom.SetAtomMapNum(idx)
        idx += 1
    for atom in product_mol.GetAtoms():
        atom.SetAtomMapNum(remap[atom.GetIntProp('molAtomMapNumber')])
    remap_reaction_str = '{}>>{}'.format(Chem.MolToSmiles(reactant_mol), Chem.MolToSmiles(product_mol))

    remap_edits = []
    for edit in edit_str.split(';'):
        a, b = edit.split('-')
        a = remap[int(a)]
        b = remap[int(b)]
        if a >= b:
            a, b = b, a
        remap_edits.append('{}-{}'.format(a, b))
    remap_edits = ';'.join(remap_edits)

    return remap_reaction_str, remap_edits


def read_txt_in_list(f):
    str_tuple_list = []
    with open(f, 'r') as f:
        for line in f:
            reaction_str, edit_str = line.strip("\r\n ").split()
            str_tuple_list.append((reaction_str, edit_str))
    return str_tuple_list


def read_txt_in_bins(f):
    # put input in bin based on heavy ataom count
    input_in_bins = {bin: [] for bin in INPUT_BINS}
    with open(f, 'r') as f:
        for line in f:
            reaction_str, edit_str = line.strip("\r\n ").split()
            num_heavy_atom = count_heavy_atom(reaction_str)
            for bin in INPUT_BINS:
                if num_heavy_atom <= bin or bin == INPUT_BINS[-1]:
                    input_in_bins[bin].append((reaction_str, edit_str))
                    break
    # filter those bins that doesn't have input
    input_in_bins = {bin: inputs for bin, inputs in input_in_bins.items() if len(inputs) > 0}
    return input_in_bins


def _batch_feature(tensor_dict_list, max_num_atom):
    def expand_first_dim(mat):
        n, m = mat.shape
        batched_mat = np.zeros((max_num_atom, m))
        batched_mat[:n] = mat
        return batched_mat

    def expand_first_second_dim(mat):
        n, m, k = mat.shape
        batched_mat = np.zeros((max_num_atom, max_num_atom, k))
        batched_mat[:n, :m] = mat
        return batched_mat

    def expand_1d_neighbor_idx(mat, idx):
        n, m = mat.shape
        batched_mat = np.zeros((max_num_atom, m, 2))
        batched_mat[:n, :, 0] = idx
        batched_mat[:n, :, 1] = mat
        return batched_mat

    def expand_2d_neighbor_idx(mat, idx):
        n, m, _ = mat.shape
        batched_mat = np.zeros((max_num_atom, m, 3))
        batched_mat[:n, :, 0] = idx
        batched_mat[:n, :, 1:] = mat
        return batched_mat

    def get_batched_atom_mask(mat):
        batched_atom_mask = np.zeros(max_num_atom)
        batched_atom_mask[:mat.shape[0]] = 1
        return batched_atom_mask

    # pad data
    batched_atom_features = [expand_first_dim(tensor_dict[FEATURE_KEY][ATOM_FEATURES_KEY])
                             for tensor_dict in tensor_dict_list]
    batched_atom_mask = [get_batched_atom_mask(tensor_dict[FEATURE_KEY][ATOM_FEATURES_KEY])
                         for tensor_dict in tensor_dict_list]
    batched_bond_features = [tensor_dict[FEATURE_KEY][BOND_FEATURES_KEY]
                             for tensor_dict in tensor_dict_list]
    batched_bond_features = [expand_first_second_dim(mat)
                             for mat in batched_bond_features]
    batched_neighbor_atom = [expand_1d_neighbor_idx(tensor_dict[FEATURE_KEY][NEIGHBOR_ATOM_KEY], idx)
                             for idx, tensor_dict in enumerate(tensor_dict_list)]
    batched_neighbor_bond = [expand_2d_neighbor_idx(tensor_dict[FEATURE_KEY][NEIGHBOR_BOND_KEY], idx)
                             for idx, tensor_dict in enumerate(tensor_dict_list)]
    batched_neighbor_mask = [expand_first_dim(tensor_dict[FEATURE_KEY][NEIGHBOR_MASK_KEY])
                             for tensor_dict in tensor_dict_list]

    # stack data
    batched_atom_features = np.stack(batched_atom_features, axis=0)
    batched_atom_mask = np.stack(batched_atom_mask, axis=0)
    batched_bond_features = np.stack(batched_bond_features, axis=0)
    batched_neighbor_bond = np.stack(batched_neighbor_bond, axis=0)
    batched_neighbor_atom = np.stack(batched_neighbor_atom, axis=0)
    batched_neighbor_mask = np.stack(batched_neighbor_mask, axis=0)

    return {ATOM_FEATURES_KEY: batched_atom_features,
            ATOM_MASK_KEY: batched_atom_mask,
            BOND_FEATURES_KEY: batched_bond_features,
            NEIGHBOR_ATOM_KEY: batched_neighbor_atom,
            NEIGHBOR_BOND_KEY: batched_neighbor_bond,
            NEIGHBOR_MASK_KEY: batched_neighbor_mask}


def _batch_label(tensor_dict_list, max_num_atom):
    def batch_2d_expand(mat, val_arr):
        batched_mat = np.zeros((max_num_atom, max_num_atom, len(val_arr)))
        n, m = mat.shape
        for i in range(n):
            for j in range(m):
                batched_mat[i, j] = mat[i, j] == val_arr
        return batched_mat

    def batch_1d_expand(arr, val_arr):
        batched_arr = np.zeros((max_num_atom, len(val_arr)))
        for i in range(len(arr)):
            batched_arr[i] = arr[i] == val_arr
        return batched_arr

    def batch_1d(arr):
        batched_arr = np.zeros(max_num_atom)
        batched_arr[:len(arr)] = arr
        return batched_arr

    # pad data
    batched_edge_delta = [batch_2d_expand(tensor_dict[LABEL_KEY][EDGE_DELTA_KEY], EDGE_DELTA_VAL_LIST)
                          for tensor_dict in tensor_dict_list]
    batched_h_delta = [batch_1d_expand(tensor_dict[LABEL_KEY][H_DELTA_KEY], H_DELTA_VAL_LIST)
                       for tensor_dict in tensor_dict_list]
    batched_c_delta = [batch_1d_expand(tensor_dict[LABEL_KEY][C_DELTA_KEY], C_DELTA_VAL_LIST)
                       for tensor_dict in tensor_dict_list]
    batched_octet_sum = [batch_1d(tensor_dict[LABEL_KEY][OCTET_SUM_KEY]) for tensor_dict in tensor_dict_list]

    # stack data
    batched_edge_delta = np.stack(batched_edge_delta, axis=0)
    batched_h_delta = np.stack(batched_h_delta, axis=0)
    batched_c_delta = np.stack(batched_c_delta, axis=0)
    batched_octet_sum = np.stack(batched_octet_sum, axis=0)

    return {EDGE_DELTA_KEY: batched_edge_delta,
            H_DELTA_KEY: batched_h_delta,
            C_DELTA_KEY: batched_c_delta,
            OCTET_SUM_KEY: batched_octet_sum}


def batch_reactivity_prediction_tensor_dict_list(tensor_dict_list, train_mode=True):
    max_num_atom = max([tensor_dict[FEATURE_KEY][ATOM_FEATURES_KEY].shape[0]
                        for tensor_dict in tensor_dict_list])
    batched_reaction_strs = [tensor_dict[REACTION_STR_KEY] for tensor_dict in tensor_dict_list]
    batched_edit_strs = [tensor_dict[EDIT_STR_KEY] for tensor_dict in tensor_dict_list]
    batched_results = {REACTION_STR_KEY: batched_reaction_strs, EDIT_STR_KEY: batched_edit_strs}
    batched_features = _batch_feature(tensor_dict_list, max_num_atom)
    batched_results[FEATURE_KEY] = batched_features
    if train_mode:
        batched_labels = _batch_label(tensor_dict_list, max_num_atom)
        batched_results[LABEL_KEY] = batched_labels
    return batched_results


def _load_eval_hdf5(input_file, enqueue_op, input_queue, session, input_ph, batch_size):
    def fix_neighbor_batch_idx(mat):
        batch_idx = np.array(range(mat.shape[0])).reshape((mat.shape[0], 1, 1))
        mat[..., 0] = batch_idx
        return mat

    def fill_batch(mat):
        input_batch_size = mat.shape[0]
        if input_batch_size < batch_size:
            augment_shape = tuple([batch_size] + list(mat.shape[1:]))
            augment_mat = np.zeros(augment_shape)
            augment_mat[:input_batch_size] = mat
            mat = augment_mat
        return mat

    dataset = h5py.File(os.path.join(input_file, INPUT_HDF5_FILE_KEY), 'r')
    bins = dataset[METADATA_KEY][METADATA_BIN_LIST_KEY][()].tolist()

    for bin_picked in bins:
        start_idx = 0
        while start_idx < len(dataset[str(bin_picked)][FEATURE_KEY][ATOM_FEATURES_KEY]):
            end_idx = min(start_idx + batch_size, len(dataset[str(bin_picked)][FEATURE_KEY][ATOM_FEATURES_KEY]))
            idx_picked = list(range(start_idx, end_idx))
            reaction_strs = dataset[str(bin_picked)][REACTION_STR_KEY][idx_picked]
            edit_strs = dataset[str(bin_picked)][EDIT_STR_KEY][idx_picked]
            str_tuple_list = [(reaction_str, edit_str) for reaction_str, edit_str in zip(reaction_strs, edit_strs)]
            input_data = [dataset[str(bin_picked)][FEATURE_KEY][ATOM_FEATURES_KEY][idx_picked],
                          dataset[str(bin_picked)][FEATURE_KEY][ATOM_MASK_KEY][idx_picked],
                          dataset[str(bin_picked)][FEATURE_KEY][BOND_FEATURES_KEY][idx_picked],
                          fix_neighbor_batch_idx(dataset[str(bin_picked)][FEATURE_KEY][NEIGHBOR_ATOM_KEY][idx_picked]),
                          fix_neighbor_batch_idx(dataset[str(bin_picked)][FEATURE_KEY][NEIGHBOR_BOND_KEY][idx_picked]),
                          dataset[str(bin_picked)][FEATURE_KEY][NEIGHBOR_MASK_KEY][idx_picked],
                          dataset[str(bin_picked)][LABEL_KEY][EDGE_DELTA_KEY][idx_picked],
                          dataset[str(bin_picked)][LABEL_KEY][H_DELTA_KEY][idx_picked],
                          dataset[str(bin_picked)][LABEL_KEY][C_DELTA_KEY][idx_picked]]
            input_data = [fill_batch(data) for data in input_data]
            feed_dict = {k: v for k, v in zip(input_ph, input_data)}
            session.run(enqueue_op, feed_dict=feed_dict)
            input_queue.put({OUTPUT_STR_TUPLE_KEY: str_tuple_list})
            start_idx = end_idx


def load_eval_queue(input_file, enqueue_op, input_queue, session, input_ph, batch_size, input_format=HDF5_FORMAT_KEY):
    if input_format == HDF5_FORMAT_KEY:
        _load_eval_hdf5(input_file, enqueue_op, input_queue, session, input_ph, batch_size)
    else:
        print('Invalid input format: {}'.format(input_format))
        raise ValueError('Invalid input format: {}'.format(input_format))


def _save_output_to_pkl(output_by_bin, output_path):
    output_list = []
    for bin in INPUT_BINS:
        output_list += output_by_bin[bin]
    with open(os.path.join(output_path, OUTPUT_DELTA_PREDICTION_PKL_FILE_KEY), 'wb') as f:
        pickle.dump(output_list, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_output_by_bin(output_by_bin, output_path, output_format=PKL_FORMAT_KEY):
    if output_format == PKL_FORMAT_KEY:
        _save_output_to_pkl(output_by_bin, output_path)
    else:
        print('Invalid output format: {}'.format(output_format))
        raise ValueError('Invalid output format: {}'.format(output_format))


def _load_delta_pred_list_from_pkl(input_path):
    with open(input_path, 'rb') as f:
        delta_pred_list = pickle.load(f)
    return delta_pred_list


def load_delta_pred_list(input_path, shuffle=True, input_format=PKL_FORMAT_KEY):
    print('Loading data from {}'.format(input_path))
    if input_format == PKL_FORMAT_KEY:
        delta_pred_list = _load_delta_pred_list_from_pkl(input_path)
    else:
        raise ValueError('Invalid input format: {}'.format(input_format))

    if shuffle:
        print('Shuffling data.')
        random.shuffle(delta_pred_list)

    print('Data is ready.')
    return delta_pred_list

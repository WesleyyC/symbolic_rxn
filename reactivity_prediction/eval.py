"""Evaluation code for reactivity prediction."""

import argparse
import warnings

import numpy as np
from rdkit import RDLogger

from input_parsing.delta_label import get_delta_labels
from input_parsing.util import get_reactant_product_molecule, get_reactant_atom_idx, get_reactant_mols, EDGE_DELTA_KEY, \
    H_DELTA_KEY, C_DELTA_KEY, EDGE_DELTA_VAL_LIST, H_DELTA_VAL_LIST, C_DELTA_VAL_LIST
from reactivity_prediction.io import OUTPUT_REACTION_STR_KEY, OUTPUT_EDGE_DELTA_KEY, OUTPUT_H_DELTA_KEY, \
    OUTPUT_C_DELTA_KEY, OUTPUT_EDIT_STR_KEY, load_delta_pred_list
from reactivity_prediction.logger import LOGGER_EDGE_DELTA_KEY, LOGGER_H_DELTA_KEY, LOGGER_C_DELTA_KEY, \
    LOGGER_ACCURACY_KEY, LOGGER_PRECISION_KEY, LOGGER_RECALL_KEY, LOGGER_DELTAS, LOGGER_METRICS, LOGGER_F1_KEY

parser = argparse.ArgumentParser(description='Reactivity Prediction Evaluation')
parser.add_argument('--input', type=str, required=True)
args = parser.parse_args()

lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)

warnings.filterwarnings("error")

TOP_Ks = [6, 8, 10, 12, 14, 16, 18, 20]


def _accuracy(label, pred, index_np):
    label = label.flatten().astype(int)
    mask = (label != 0).astype(float)
    pred = np.argmax(pred, axis=-1)
    pred = pred.flatten()
    pred_val = np.array([index_np[val] for val in pred]).astype(int)

    try:
        acc = np.sum(mask * (pred_val == label)) / np.sum(mask)
    except RuntimeWarning:
        acc = np.nan

    return acc


def _precision_recall(label, pred, index_np):
    label = label.flatten().astype(int)
    label = label != 0

    pred = np.argmax(pred, axis=-1) != index_np.tolist().index(0)
    pred = pred.flatten()

    tp = np.sum((pred == label) * label).astype(float)

    try:
        precision = tp / np.sum(pred)
        recall = tp / np.sum(label)
    except RuntimeWarning:
        precision = np.nan
        recall = np.nan

    return precision, recall


def eval_delta(delta_pred_list):
    metric_dict = {
        LOGGER_EDGE_DELTA_KEY: {
            LOGGER_ACCURACY_KEY: [],
            LOGGER_PRECISION_KEY: [],
            LOGGER_RECALL_KEY: [],
        },
        LOGGER_H_DELTA_KEY: {
            LOGGER_ACCURACY_KEY: [],
            LOGGER_PRECISION_KEY: [],
            LOGGER_RECALL_KEY: [],
        },
        LOGGER_C_DELTA_KEY: {
            LOGGER_ACCURACY_KEY: [],
            LOGGER_PRECISION_KEY: [],
            LOGGER_RECALL_KEY: [],
        },
    }

    for reaction_idx, delta_pred in enumerate(delta_pred_list):
        reaction_smi = delta_pred[OUTPUT_REACTION_STR_KEY]

        reactant_mol, product_mol = get_reactant_product_molecule(reaction_smi)

        delta_labels = get_delta_labels(reactant_mol, product_mol)

        precision, recall = _precision_recall(delta_labels[EDGE_DELTA_KEY],
                                              delta_pred[OUTPUT_EDGE_DELTA_KEY],
                                              EDGE_DELTA_VAL_LIST)
        acc = _accuracy(delta_labels[EDGE_DELTA_KEY],
                        delta_pred[OUTPUT_EDGE_DELTA_KEY],
                        EDGE_DELTA_VAL_LIST)
        metric_dict[LOGGER_EDGE_DELTA_KEY][LOGGER_PRECISION_KEY].append(precision)
        metric_dict[LOGGER_EDGE_DELTA_KEY][LOGGER_RECALL_KEY].append(recall)
        metric_dict[LOGGER_EDGE_DELTA_KEY][LOGGER_ACCURACY_KEY].append(acc)

        precision, recall = _precision_recall(delta_labels[H_DELTA_KEY],
                                              delta_pred[OUTPUT_H_DELTA_KEY],
                                              H_DELTA_VAL_LIST)
        acc = _accuracy(delta_labels[H_DELTA_KEY],
                        delta_pred[OUTPUT_H_DELTA_KEY],
                        H_DELTA_VAL_LIST)
        metric_dict[LOGGER_H_DELTA_KEY][LOGGER_PRECISION_KEY].append(precision)
        metric_dict[LOGGER_H_DELTA_KEY][LOGGER_RECALL_KEY].append(recall)
        metric_dict[LOGGER_H_DELTA_KEY][LOGGER_ACCURACY_KEY].append(acc)

        precision, recall = _precision_recall(delta_labels[C_DELTA_KEY],
                                              delta_pred[OUTPUT_C_DELTA_KEY],
                                              C_DELTA_VAL_LIST)
        acc = _accuracy(delta_labels[C_DELTA_KEY],
                        delta_pred[OUTPUT_C_DELTA_KEY],
                        C_DELTA_VAL_LIST)
        metric_dict[LOGGER_C_DELTA_KEY][LOGGER_PRECISION_KEY].append(precision)
        metric_dict[LOGGER_C_DELTA_KEY][LOGGER_RECALL_KEY].append(recall)
        metric_dict[LOGGER_C_DELTA_KEY][LOGGER_ACCURACY_KEY].append(acc)

    print('==========================================================')
    print('==================== Delta Evaluation ====================')
    print('==========================================================')
    for delta_name in LOGGER_DELTAS:
        print('{}:'.format(delta_name))
        for metric_name in LOGGER_METRICS:
            if metric_name == LOGGER_F1_KEY:
                precision = np.nanmean(metric_dict[delta_name][LOGGER_PRECISION_KEY])
                recall = np.nanmean(metric_dict[delta_name][LOGGER_RECALL_KEY])
                f1 = 2 * precision * recall / (precision + recall)
                print('{}: {:.3f}'.format(metric_name, f1))
            else:
                print('{}: {:.3f}'.format(metric_name, np.nanmean(metric_dict[delta_name][metric_name])))
        print('==========================================================')


def eval_reaction_bond(delta_pred_list):
    def _find_top_k(delta_pred, top_k):

        reactant_mol, product_mol = get_reactant_product_molecule(delta_pred[OUTPUT_REACTION_STR_KEY])
        reactant_atom_idx = get_reactant_atom_idx(get_reactant_mols(reactant_mol), product_mol)

        reaction_center_proba = delta_pred[OUTPUT_EDGE_DELTA_KEY]
        idxs = np.unravel_index(np.argsort(reaction_center_proba.ravel())[::-1],
                                reaction_center_proba.shape)
        top_k_edits = []
        counter = 0
        while len(top_k_edits) < top_k:
            x = idxs[0][counter]
            y = idxs[1][counter]
            z = idxs[2][counter]
            counter += 1
            if z == 3 or x >= y:
                continue
            if x in reactant_atom_idx and y in reactant_atom_idx:
                top_k_edits.append('{}-{}-{}'.format(x + 1, y + 1, EDGE_DELTA_VAL_LIST[z]))

        delta_labels = get_delta_labels(reactant_mol, product_mol)
        edge_delta_label = delta_labels[EDGE_DELTA_KEY]
        edit_str = []
        for x, y in zip(*np.nonzero(edge_delta_label)):
            if x >= y:
                continue
            else:
                edit_str.append('{}-{}-{}'.format(x+1, y+1, edge_delta_label[x, y]))

        return top_k_edits, edit_str

    top_k_result = {top_k: [] for top_k in TOP_Ks}

    for reaction_idx, delta_pred in enumerate(delta_pred_list):
        top_k_edits, edit_str = _find_top_k(delta_pred, top_k=max(TOP_Ks))
        for top_k in TOP_Ks:
            if top_k < len(edit_str):
                top_k_result[top_k].append(np.nan)
            elif set(top_k_edits[:top_k]) >= set(edit_str):
                top_k_result[top_k].append(1.0)
            else:
                top_k_result[top_k].append(0.0)

    print('==========================================================')
    print('================ Reaction Bond Prediction ================')
    print('==========================================================')
    for top_k in TOP_Ks:
        print('Top {}:\t{:.3f}'.format(top_k, np.nanmean(top_k_result[top_k])))
    print('==========================================================')


if __name__ == '__main__':
    delta_pred_list = load_delta_pred_list(args.input)
    eval_reaction_bond(delta_pred_list)
    eval_delta(delta_pred_list)

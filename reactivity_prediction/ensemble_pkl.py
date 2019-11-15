"""Ensembling the prediction outputs from different models."""

import argparse
import os
import pickle

from reactivity_prediction.io import OUTPUT_EDGE_DELTA_KEY, OUTPUT_H_DELTA_KEY, OUTPUT_C_DELTA_KEY, load_delta_pred_list

parser = argparse.ArgumentParser(description='Perdition Dictionary Merge')
parser.add_argument('--inputs', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()


def average_list(input_list):
    output = []
    num_input = len(input_list)
    num_item = len(input_list[0])
    for i in range(num_item):
        item_dict = input_list[0][i]
        for j in range(1, num_input):
            item_dict[OUTPUT_EDGE_DELTA_KEY] += input_list[j][i][OUTPUT_EDGE_DELTA_KEY]
            item_dict[OUTPUT_H_DELTA_KEY] += input_list[j][i][OUTPUT_H_DELTA_KEY]
            item_dict[OUTPUT_C_DELTA_KEY] += input_list[j][i][OUTPUT_C_DELTA_KEY]
        item_dict[OUTPUT_EDGE_DELTA_KEY] /= num_input
        item_dict[OUTPUT_H_DELTA_KEY] /= num_input
        item_dict[OUTPUT_C_DELTA_KEY] /= num_input
        output.append(item_dict)
    return output


if __name__ == '__main__':
    input_fs = args.inputs.split(',')
    input_list = [load_delta_pred_list(f, shuffle=False) for f in input_fs]
    output_list = average_list(input_list)
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(output_list, f, protocol=pickle.HIGHEST_PROTOCOL)

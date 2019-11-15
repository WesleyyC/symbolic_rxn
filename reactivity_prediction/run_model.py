"""Main .py for running the model."""

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import multiprocessing
import os
import threading
from functools import partial

import tensorflow as tf
from rdkit import RDLogger

from input_parsing.util import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, MAX_NEIGHBOR_NUM, EDGE_DELTA_VAL_LIST, \
    C_DELTA_VAL_LIST, H_DELTA_VAL_LIST, get_reactant_mol_size_from_reaction_str, count_heavy_atom
from reactivity_prediction.io import load_eval_queue, OUTPUT_STR_TUPLE_KEY, OUTPUT_REACTION_STR_KEY, \
    OUTPUT_EDIT_STR_KEY, OUTPUT_EDGE_DELTA_KEY, OUTPUT_H_DELTA_KEY, OUTPUT_C_DELTA_KEY, INPUT_BINS, \
    save_output_by_bin, OUTPUT_ATTENTION_SCORE
from reactivity_prediction.nn import message_passing_encoding, convolution_readout

####################################### INIT MODEL PARAMETER #######################################

TF_FIFO_QUEUE_SIZE = 100

parser = argparse.ArgumentParser(description='Run Reactivity Prediction Model')
parser.add_argument('--mode', choices=['infer'], required=True)
parser.add_argument('--eval_input', type=str, default=None)
parser.add_argument('--eval_output', type=str, default=None)
parser.add_argument('--hdf5_output_format', action='store_true', default=False)
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--batch', type=int, default=20)
parser.add_argument('--hidden', type=int, default=800)
parser.add_argument('--mp_step', type=int, default=4)
parser.add_argument('--conv_layer', type=int, default=6)
parser.add_argument('--conv_kernel', type=int, default=3)
parser.add_argument('--verbose', action='store_true', default=False)
args = parser.parse_args()

# rdkit suppress warning logger
if not args.verbose:
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.ERROR)

# setup the model fn
mp_fn = message_passing_encoding
readout_fn = partial(convolution_readout,
                     conv_layer=args.conv_layer,
                     kernel_size=args.conv_kernel)

####################################### DEFINE MODEL #######################################

# data ph
tf_atom_features_ph = tf.placeholder(tf.float32, [args.batch, None, ATOM_FEATURE_DIM])
tf_atom_mask_ph = tf.placeholder(tf.float32, [args.batch, None])
tf_bond_features_ph = tf.placeholder(tf.float32, [args.batch, None, None, BOND_FEATURE_DIM])
tf_neighbor_atom_ph = tf.placeholder(tf.int32, [args.batch, None, MAX_NEIGHBOR_NUM, 2])
tf_neighbor_bond_ph = tf.placeholder(tf.int32, [args.batch, None, MAX_NEIGHBOR_NUM, 3])
tf_neighbor_mask_ph = tf.placeholder(tf.float32, [args.batch, None, MAX_NEIGHBOR_NUM])
tf_edge_delta_ph = tf.placeholder(tf.float32, [args.batch, None, None, len(EDGE_DELTA_VAL_LIST)])
tf_h_delta_ph = tf.placeholder(tf.float32, [args.batch, None, len(H_DELTA_VAL_LIST)])
tf_c_delta_ph = tf.placeholder(tf.float32, [args.batch, None, len(C_DELTA_VAL_LIST)])
tf_train_ph = tf.placeholder(tf.bool)

tf_features_ph = [tf_atom_features_ph,
                  tf_atom_mask_ph,
                  tf_bond_features_ph,
                  tf_neighbor_atom_ph,
                  tf_neighbor_bond_ph,
                  tf_neighbor_mask_ph]
tf_labels_ph = [tf_edge_delta_ph,
                tf_h_delta_ph,
                tf_c_delta_ph]
tf_input_ph = tf_features_ph + tf_labels_ph

# data queue
tf_train_queue = tf.FIFOQueue(TF_FIFO_QUEUE_SIZE, [ph.dtype for ph in tf_input_ph])
tf_eval_queue = tf.FIFOQueue(TF_FIFO_QUEUE_SIZE, [ph.dtype for ph in tf_input_ph])
tf_train_enqueue_op = tf_train_queue.enqueue(tf_input_ph)
tf_eval_enqueue_op = tf_eval_queue.enqueue(tf_input_ph)

tf_input = tf.cond(tf_train_ph, tf_train_queue.dequeue, tf_eval_queue.dequeue)
for tf_tensor_ph, tf_tensor in zip(tf_input_ph, tf_input):
    tf_tensor.set_shape(tf_tensor_ph.shape)
tf_features = tf_input[:len(tf_features_ph)]
tf_labels = tf_input[len(tf_features_ph):]

# model
tf_atom_encoding = mp_fn(tf_features, args.hidden, args.mp_step, training=tf_train_ph)
tf_features = [tf_atom_encoding] + tf_features
tf_delta_scores, tf_attention_score = readout_fn(tf_features, args.hidden, training=tf_train_ph)

# inference
tf_delta_probas = [tf.nn.softmax(delta_score) for delta_score in tf_delta_scores]

####################################### Inference #######################################

# tf session with growing GPU memory
gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# tf data coordinator
data_coord = tf.train.Coordinator()
threads = []

try:

    if args.mode == 'infer':

        if args.eval_input is None:
            raise ValueError('--eval_input has to be specified.')
        if args.eval_output is None:
            raise ValueError('--eval_output has to be specified.')

        os.makedirs(args.eval_output, exist_ok=True)

        saver.restore(session, tf.train.latest_checkpoint(args.ckpt))

        input_queue = multiprocessing.SimpleQueue()
        output_by_bin = {bin: [] for bin in INPUT_BINS}

        threads.append(threading.Thread(target=load_eval_queue, args=(
            args.eval_input, tf_eval_enqueue_op, input_queue, session, tf_input_ph, args.batch,)))
        for thread in threads:
            thread.start()

        inferred_count = 0
        while threads[-1].is_alive() or not input_queue.empty():
            probas, attention_score = session.run([tf_delta_probas, tf_attention_score],
                                                  feed_dict={tf_train_ph: False})
            edge_delta_proba, h_delta_proba, c_delta_proba = probas
            input_dict = input_queue.get()
            str_tuple_list = input_dict[OUTPUT_STR_TUPLE_KEY]
            for i in range(len(str_tuple_list)):
                reaction_str = str_tuple_list[i][0]
                n_atom = get_reactant_mol_size_from_reaction_str(reaction_str)
                output_dict = {OUTPUT_REACTION_STR_KEY: reaction_str,
                               OUTPUT_EDIT_STR_KEY: str_tuple_list[i][1],
                               OUTPUT_ATTENTION_SCORE: attention_score[i][:n_atom, :n_atom],
                               OUTPUT_EDGE_DELTA_KEY: edge_delta_proba[i][:n_atom, :n_atom],
                               OUTPUT_H_DELTA_KEY: h_delta_proba[i][:n_atom],
                               OUTPUT_C_DELTA_KEY: c_delta_proba[i][:n_atom]}
                num_heavy_atom = count_heavy_atom(reaction_str)
                for bin in INPUT_BINS:
                    if num_heavy_atom <= bin or bin == INPUT_BINS[-1]:
                        output_by_bin[bin].append(output_dict)
                        break
                inferred_count += 1
                if inferred_count % 2000 == 0:
                    print('Inferring #{}'.format(inferred_count))

        save_output_by_bin(output_by_bin, args.eval_output)
    else:
        print('Invalid modeL {}'.format(args.mode))
        raise ValueError('Invalid model {}'.format(args.mode))


except Exception as e:
    data_coord.request_stop(e)
finally:
    data_coord.request_stop()
    data_coord.join(threads)

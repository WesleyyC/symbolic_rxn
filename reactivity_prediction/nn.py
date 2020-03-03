import tensorflow as tf

from input_parsing.util import EDGE_DELTA_VAL_LIST, H_DELTA_VAL_LIST, C_DELTA_VAL_LIST


def message_passing_encoding(tf_features, hidden_units, mp_step, activation=tf.nn.relu, drop_out=0, training=False):
    # get input
    atom_features, atom_mask, bond_features, neighbor_atom_idx, neighbor_bond_idx, neighbor_mask = tf_features

    with tf.variable_scope('message_passing_encoding'):
        with tf.variable_scope('input_prep'):
            atom_features = tf.layers.dense(atom_features, hidden_units,
                                            activation=activation,
                                            name='init_atom_features_layer')
            bond_features = tf.layers.dense(bond_features, 16,
                                            activation=activation,
                                            name='init_bond_features_layer')
            atom_features = tf.layers.dropout(atom_features, rate=drop_out, training=training)
            neighbor_bond_features = tf.gather_nd(bond_features, neighbor_bond_idx)
            atom_mask = tf.expand_dims(atom_mask, axis=-1)
            neighbor_mask = tf.expand_dims(neighbor_mask, axis=-1)

        for i in range(mp_step):
            with tf.variable_scope('{}_layer_message_passing'.format(mp_step), reuse=i > 0):
                neighbor_atom_features = tf.gather_nd(atom_features, neighbor_atom_idx)
                neighbor_features = tf.concat([neighbor_atom_features, neighbor_bond_features], axis=-1)
                neighbor_features = tf.layers.dense(neighbor_features, hidden_units,
                                                    activation=activation,
                                                    name='message_function')
                neighbor_features = neighbor_features * neighbor_mask
                neighbor_features = tf.reduce_sum(neighbor_features, axis=-2)
                new_atom_features = tf.concat([atom_features, neighbor_features], axis=-1)
                new_atom_features = tf.layers.dense(new_atom_features, hidden_units,
                                                    activation=activation,
                                                    name='update_function')
                atom_features = new_atom_features
                atom_features = tf.layers.dropout(atom_features, rate=drop_out, training=training)

        with tf.variable_scope('final_encoding'):
            mp_atom_embedding = tf.layers.dense(atom_features, hidden_units, name='final_encoding_fc')
            mp_atom_embedding = mp_atom_embedding * atom_mask
            mp_atom_embedding = tf.layers.dropout(mp_atom_embedding, rate=drop_out, training=training)

    return mp_atom_embedding


def _generate_atom_pair(atom_wl_encoding, bond_features, hidden_units, activation, drop_out, training):
    with tf.variable_scope('input_prep'):
        atom_pair = tf.expand_dims(atom_wl_encoding, axis=1) \
                    + tf.expand_dims(atom_wl_encoding, axis=2)

    with tf.variable_scope('attention'):
        attention_hidden = tf.layers.dense(atom_pair, hidden_units, name='atom_pair_hidden') \
                           + tf.layers.dense(bond_features, hidden_units, name='bond_feature_hidden')
        attention_hidden = activation(attention_hidden)
        attention_score = tf.layers.dense(attention_hidden, 1,
                                          activation=tf.nn.sigmoid,
                                          name='attention_score')
        attention_context = tf.expand_dims(atom_wl_encoding, axis=1) * attention_score
        attention_context = tf.reduce_sum(attention_context, axis=-2)
        attention_pair = tf.expand_dims(attention_context, axis=1) + tf.expand_dims(attention_context, axis=2)

    with tf.variable_scope('atom_pair'):
        atom_pair = tf.layers.dense(atom_pair, hidden_units, name='atom_pair_hidden_projection') \
                    + tf.layers.dense(bond_features, hidden_units, name='atom_pair_bond_projection') \
                    + tf.layers.dense(attention_pair, hidden_units, name='atom_pair_attention_projection')
        atom_pair = activation(atom_pair)
        atom_pair = tf.layers.dropout(atom_pair, rate=drop_out, training=training)
    return atom_pair, attention_score


def _final_readout_from_atom_pair(atom_pair):
    with tf.variable_scope('final_readout'):
        with tf.variable_scope('edge_delta'):
            edge_delta_score = tf.layers.dense(atom_pair, len(EDGE_DELTA_VAL_LIST),
                                               name='edge_delta_score')
            edge_delta_score = (edge_delta_score + tf.transpose(edge_delta_score, perm=[0, 2, 1, 3])) / 2

        with tf.variable_scope('h_delta'):
            h_delta_score = tf.layers.dense(atom_pair, len(H_DELTA_VAL_LIST),
                                            name='h_delta_score')
            h_delta_score = tf.reduce_sum(h_delta_score, axis=-2)

        with tf.variable_scope('c_delta'):
            c_delta_score = tf.layers.dense(atom_pair, len(C_DELTA_VAL_LIST),
                                            name='c_delta_score')
            c_delta_score = tf.reduce_sum(c_delta_score, axis=-2)

        return edge_delta_score, h_delta_score, c_delta_score


def _convolution_transformer(atom_pair, hidden_units, conv_layer, kernel_size, activation, drop_out, training):
    with tf.variable_scope('convolution'):
        for _ in range(conv_layer):
            hidden_units = hidden_units // 2
            atom_pair = tf.layers.conv2d(atom_pair, hidden_units,
                                         kernel_size=kernel_size,
                                         padding='same',
                                         activation=activation)
            atom_pair = tf.layers.dropout(atom_pair, rate=drop_out, training=training)
        return atom_pair


def convolution_readout(tf_features, hidden_units, conv_layer=3, kernel_size=3, activation=tf.nn.relu, drop_out=0,
                        training=False):
    # get data
    atom_wl_encoding, _, _, bond_features, _, _, _ = tf_features

    with tf.variable_scope('convolution_readout'):
        atom_pair, attention_score = _generate_atom_pair(atom_wl_encoding, bond_features, hidden_units, activation,
                                                         drop_out, training)
        atom_pair = _convolution_transformer(atom_pair, hidden_units, conv_layer, kernel_size, activation, drop_out,
                                             training)
        readout = _final_readout_from_atom_pair(atom_pair)
        return readout, attention_score

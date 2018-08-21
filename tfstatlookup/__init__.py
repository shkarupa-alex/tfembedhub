from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub

EMBEDDINGS_VAR_NAME = 'stat_embeddings'


def read_and_split(src_file):
    keys_stats = np.load(src_file)
    assert 2 == len(keys_stats.shape)
    assert 1 <= keys_stats.shape[0]
    assert 2 <= keys_stats.shape[1]
    assert '<-UNIQUE->' == keys_stats[0][0]

    return _split_keys_stats(keys_stats)


def _split_keys_stats(keys_stats):
    keys = keys_stats[:, 0].astype('U')
    stats = keys_stats[:, 1:].astype(np.float32)

    return keys, stats


def export_hub_module(keys, stats, dest_path):
    embed_size = stats.shape[1]
    module_spec = _make_module_spec(keys, embed_size)

    hub_module = hub.Module(module_spec)
    # The embeddings may be very large (e.g., larger than the 2GB serialized Tensor limit). To avoid having them frozen
    # as constant Tensors in the graph we instead assign them through the placeholders and feed_dict mechanism.
    stats_value = tf.placeholder(tf.float32)
    load_stats = tf.assign(hub_module.variable_map[EMBEDDINGS_VAR_NAME], stats_value)

    with tf.Session() as sess:
        sess.run([load_stats], feed_dict={stats_value: stats})
        hub_module.export(dest_path, sess)


def _make_module_spec(keys_vocab, embedding_size):
    def module_fn():
        input_keys = tf.placeholder(shape=[None], dtype=tf.string, name='keys')

        lookup_table = tf.contrib.lookup.index_table_from_tensor(mapping=keys_vocab, default_value=0)
        input_ids = lookup_table.lookup(input_keys)

        embeddings_shape = [len(keys_vocab), embedding_size]
        embeddings_var = tf.get_variable(
            name=EMBEDDINGS_VAR_NAME,
            dtype=tf.float32,
            initializer=tf.zeros(embeddings_shape),
            trainable=False
        )
        output_embedding = tf.nn.embedding_lookup(params=embeddings_var, ids=input_ids)

        hub.add_signature('default', {'keys': input_keys}, {'default': output_embedding})

    return hub.create_module_spec(module_fn)


def main():
    parser = argparse.ArgumentParser(
        description='Convert list of string keys and corresponding 2D NumPy array of floats into TF Hub lookup module')
    parser.add_argument(
        'src_path',
        type=argparse.FileType('rb'),
        help='Path to saved NumPy 2D array. First column should contain string keys. First key should be "<-UNIQUE->"')
    parser.add_argument(
        'dest_path',
        type=str,
        help='Path to export TF Hub module')

    argv, _ = parser.parse_known_args()
    assert not os.path.exists(argv.dest_path) or os.path.isdir(argv.dest_path)
    try:
        os.makedirs(argv.dest_path)
    except:
        pass

    logging.basicConfig(level=logging.INFO)

    keys, stats = read_and_split(argv.src_path)
    export_hub_module(keys, stats, argv.dest_path)

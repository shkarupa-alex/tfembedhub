from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import tensorflow as tf
import tensorflow_hub as hub

_EMBEDDINGS_VAR_NAME = 'embeddings'


def read_and_split(file_name):
    keys, features = [], []

    with open(file_name, 'rb') as src_file:
        for row in src_file:
            row = row.decode('utf-8').strip().split()
            if len(row) < 2:  # at least label and one feature
                continue

            keys.append(row[0])
            features.append([float(f) for f in row[1:]])

    assert len(keys) > 1, 'There should be at leas one row besides <UNQ>'
    assert '<UNQ>' == keys[0], 'First key should equals <UNQ>'

    sizes = [len(f) for f in features]
    assert min(sizes) == max(sizes), 'All embeddings should have same size'

    return keys, features


def export_hub_module(keys, values, dest, combiner='mean', max_norm=None):
    embed_size = len(values[0])
    module_spec = _make_module_spec(keys, embed_size, combiner, max_norm)

    hub_module = hub.Module(module_spec)

    # The embeddings may be very large (e.g., larger than the 2GB serialized Tensor limit).
    # To avoid having them frozen as constant Tensors in the graph we instead assign them through the
    # placeholders and feed_dict mechanism.
    init_values = tf.placeholder(dtype=tf.float32, shape=[len(values), embed_size])
    load_values = tf.assign(hub_module.variable_map[_EMBEDDINGS_VAR_NAME], init_values)

    with tf.Session() as sess:
        sess.run([load_values], feed_dict={init_values: values})
        hub_module.export(dest, sess)


def _make_module_spec(keys_vocab, embedding_size, combiner, max_norm):
    def module_fn():
        embeddings_shape = [len(keys_vocab), embedding_size]
        embedding_weights = tf.get_variable(
            name=_EMBEDDINGS_VAR_NAME,
            dtype=tf.float32,
            initializer=tf.zeros(embeddings_shape),
            trainable=False
        )
        lookup_table = tf.contrib.lookup.index_table_from_tensor(mapping=keys_vocab, default_value=0)

        default_keys = tf.sparse_placeholder(dtype=tf.string)
        default_ids = lookup_table.lookup(default_keys)
        default_embeddings = tf.nn.safe_embedding_lookup_sparse(
            embedding_weights=embedding_weights,
            sparse_ids=default_ids,
            combiner=combiner,
            default_id=0,
            max_norm=max_norm,
        )
        hub.add_signature('default', default_keys, default_embeddings)

        context_keys = tf.sparse_placeholder(dtype=tf.string, shape=(None, None))
        context_ids = lookup_table.lookup(context_keys)
        context_embeddings = tf.nn.safe_embedding_lookup_sparse(
            embedding_weights=embedding_weights,
            sparse_ids=context_ids,
            combiner=combiner,
            default_id=0,
            max_norm=max_norm,
        )
        hub.add_signature('context', context_keys, context_embeddings)

        sequence_keys = tf.sparse_placeholder(dtype=tf.string, shape=(None, None, None))
        sequence_ids = lookup_table.lookup(sequence_keys)
        sequence_embeddings = tf.nn.safe_embedding_lookup_sparse(
            embedding_weights=embedding_weights,
            sparse_ids=sequence_ids,
            combiner=combiner,
            default_id=0,
            max_norm=max_norm,
        )
        hub.add_signature('sequence', sequence_keys, sequence_embeddings)

    return hub.create_module_spec(module_fn)


def main():
    parser = argparse.ArgumentParser(
        description='Convert list of string keys and corresponding 2D NumPy array of floats into TF Hub lookup module')
    parser.add_argument(
        'src_path',
        type=argparse.FileType('rb'),
        help='Path to saved NumPy or text 2D array. First column should contain string keys. '
             'First key should be "<-UNIQUE->"')
    parser.add_argument(
        'dest_path',
        type=str,
        help='Path to export TF Hub module')
    parser.add_argument(
        '--combiner',
        choices=['mean', 'sqrtn', 'sum'],
        default='mean',
        help='How to combine embedding results for each entry')
    parser.add_argument(
        '--max_norm',
        type=float,
        default=None,
        help='If not None, all embeddings are l2-normalized to max_norm before combining')

    argv, _ = parser.parse_known_args()

    src_path = argv.src_path.name
    argv.src_path.close()

    assert not os.path.exists(argv.dest_path) or os.path.isdir(argv.dest_path)
    try:
        os.makedirs(argv.dest_path)
    except:
        pass

    logging.basicConfig(level=logging.INFO)

    logging.info('Reading embeddings from {}'.format(src_path))
    keys, features = read_and_split(src_path)
    logging.info('Loaded {} embeddings'.format(len(keys)))

    logging.info('Exporting embeddings to {} with combiner={} and max_norm={}'.format(
        argv.dest_path, argv.combiner, argv.max_norm))
    export_hub_module(keys, features, argv.dest_path, argv.combiner, argv.max_norm)

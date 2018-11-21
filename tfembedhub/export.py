from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import numpy as np
import os
import tensorflow as tf
import tensorflow_hub as hub

_EMBEDDINGS_VAR_NAME = 'embeddings'
_UNIQUE_KEY_NAME = '<UNQ>'


def _iterate_text_vectors(file_name):
    with open(file_name, 'rb') as src_file:
        for row in src_file:
            row = row.decode('utf-8').strip().split()
            if len(row) < 2:  # at least label and one feature
                continue
            yield row


def read_and_count(file_name):
    key_values = []
    unique_features = None
    features_size = None

    for row in _iterate_text_vectors(file_name):
        if features_size is None:
            features_size = len(row) - 1
        else:
            message = 'All embeddings should have same size (inferred {}), but row {} has size {}'
            assert features_size == len(row) - 1, message.format(features_size, row[0], len(row) - 1)

        key_values.append(row[0])
        if row[0] == _UNIQUE_KEY_NAME:
            unique_features = [float(f) for f in row[1:]]

    unique_row = _UNIQUE_KEY_NAME in key_values
    assert len(key_values) > int(unique_row), 'There should be at least one row except {}'.format(_UNIQUE_KEY_NAME)

    key_counter = collections.Counter(key_values)
    most_common = key_counter.most_common(1)[0]
    assert most_common[1] == 1, 'Dublicate key found: {}'.format(most_common[0])

    if unique_row:
        key_values.remove(_UNIQUE_KEY_NAME)
    else:
        unique_features = [0.] * features_size
    key_values.insert(0, _UNIQUE_KEY_NAME)

    return key_values, features_size, unique_features


def read_and_split(file_name, unique_features, shard_rows):
    size, features = 1, unique_features

    for row in _iterate_text_vectors(file_name):
        if row[0] == _UNIQUE_KEY_NAME:
            continue

        if size == shard_rows:
            yield features
            size, features = 0, []

        features.extend([float(f) for f in row[1:]])
        size += 1

    if size > 0:
        yield features


def export_hub_module(src_path, dest, trainable=False, combiner='mean', max_norm=None, shard_bytes=2 ** 30):
    logging.info('Checking keys and vectors in {}'.format(src_path))
    key_values, embed_size, unique_features = read_and_count(src_path)
    logging.info('Done checking: rows_count={}, embeddings_size={}'.format(len(key_values), embed_size))

    if set(unique_features) == {0.}:
        logging.info('Embedding for {} initialized with zeros'.format(_UNIQUE_KEY_NAME))
    else:
        logging.info('Embedding for {} found, using provided values'.format(_UNIQUE_KEY_NAME))

    logging.info('Creating hub module specs')
    module_spec = _make_module_spec(key_values, embed_size, trainable, combiner, max_norm, shard_bytes)
    hub_module = hub.Module(module_spec)

    # The embeddings may be very large (e.g., larger than the 2GB serialized Tensor limit).
    # To avoid having them frozen as constant Tensors in the graph we instead assign them through the
    # placeholders and feed_dict mechanism.
    # Additionally we should use variable partitioning due to restoring issues in MacOS X.
    shard_rows = hub_module.variable_map[_EMBEDDINGS_VAR_NAME][0].shape[0]
    total_shards = len(hub_module.variable_map[_EMBEDDINGS_VAR_NAME])
    with tf.Session() as sess:
        for part_i, part_values in enumerate(read_and_split(src_path, unique_features, shard_rows)):
            logging.info('Initializing embeddings part: {} of {}'.format(part_i + 1, total_shards))

            init_values = tf.placeholder(dtype=tf.float32, shape=[shard_rows, embed_size])
            part_var = hub_module.variable_map[_EMBEDDINGS_VAR_NAME][part_i]
            load_values = tf.assign(part_var, init_values)

            part_values = np.array(part_values, dtype=np.float32).reshape((shard_rows, embed_size))
            sess.run([load_values], feed_dict={init_values: part_values})

        logging.info('Exporting hub module to {}'.format(dest))
        hub_module.export(dest, sess)


def _make_module_spec(keys_vocab, embedding_size, is_trainable, combiner, max_norm, shard_bytes):
    def module_fn():
        embeddings_shape = [len(keys_vocab), embedding_size]
        embedding_weights = tf.get_variable(
            name=_EMBEDDINGS_VAR_NAME,
            shape=embeddings_shape,
            dtype=tf.float32,
            initializer=tf.zeros(embeddings_shape),
            trainable=is_trainable,
            partitioner=tf.variable_axis_size_partitioner(
                max_shard_bytes=shard_bytes  # 1Gb by default
            )
        )
        lookup_table = tf.contrib.lookup.index_table_from_tensor(mapping=keys_vocab, default_value=0)

        default_keys = tf.placeholder(dtype=tf.string, shape=[None])
        default_ids = lookup_table.lookup(default_keys)
        default_embeddings = tf.nn.embedding_lookup(
            params=embedding_weights,
            ids=default_ids,
            partition_strategy='div',
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
            partition_strategy='div',
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
            partition_strategy='div',
            max_norm=max_norm,
        )
        hub.add_signature('sequence', sequence_keys, sequence_embeddings)

    return hub.create_module_spec(module_fn)


def main():
    parser = argparse.ArgumentParser(
        description='Convert string keys and corresponding vectors of floats into TF Hub lookup module')
    parser.add_argument(
        'src_path',
        type=argparse.FileType('rb'),
        help='Path to saved NumPy or text 2D array. First column should contain string keys. '
             'First key should be "{}"'.format(_UNIQUE_KEY_NAME))
    parser.add_argument(
        'dest_path',
        type=str,
        help='Path to export TF Hub module')
    parser.add_argument(
        '--trainable',
        action='store_true',
        help='Whither embedding variable should be trainable')
    parser.add_argument(
        '--combiner',
        choices=['sum', 'mean', 'sqrtn'],
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

    logging.info('Export options: trainable={}, combiner={}, max_norm={}'.format(
        argv.trainable, argv.combiner, argv.max_norm))
    export_hub_module(src_path, argv.dest_path, argv.trainable, argv.combiner, argv.max_norm)

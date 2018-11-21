# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import shutil
import tempfile
import tensorflow as tf
import tensorflow_hub as hub
import unittest

from ..export import _UNIQUE_KEY_NAME, read_and_count, read_and_split, export_hub_module


class TestRead(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, 'source.txt')

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def testReadAndCount(self):
        source = '\n'.join([
            'key1 1. 2.',
            'key2 3. 4.',
        ]) + '\n'
        with open(self.temp_file, 'wb') as ttf:
            ttf.write(source.encode('utf-8'))

        key_values, features_size, unique_features = read_and_count(self.temp_file)
        self.assertListEqual(key_values, [_UNIQUE_KEY_NAME, 'key1', 'key2'])
        self.assertEqual(features_size, 2)
        self.assertListEqual(unique_features, [0., 0.])

    def testReadAndSplit(self):
        source = '\n'.join([
            'key1 1. 2.',
            '{} -1. 0.'.format(_UNIQUE_KEY_NAME),
            'key2 3. 4.',
        ]) + '\n'
        with open(self.temp_file, 'wb') as ttf:
            ttf.write(source.encode('utf-8'))

        _, _, unique_features = read_and_count(self.temp_file)

        expected = [
            [-1.0, 0.0, 1.0, 2.0],
            [3.0, 4.0],
        ]
        actual = list(read_and_split(self.temp_file, unique_features, 2))

        self.assertEqual(actual, expected)


class TestExportHubModule(tf.test.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, 'source.txt')
        tf.reset_default_graph()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def testExport(self):
        source_keys = ['key{}'.format(i) for i in range(25)]
        source_values = np.array([i * 1. for i in range(50)]).reshape((25, 2)).tolist()

        source_content = '\n'.join(
            ' '.join([k] + [str(v) for v in values]) for k, values in zip(source_keys, source_values))
        with open(self.temp_file, 'wb') as ttf:
            ttf.write(source_content.encode('utf-8'))

        export_hub_module(self.temp_file, self.temp_dir, shard_bytes=16)
        lookup_values = hub.Module(self.temp_dir)

        expected_values_default = [[2., 3.]]
        lookup_values_default = lookup_values(['key1'])

        expected_values_context = [[2., 3.]]
        lookup_values_context = lookup_values(tf.SparseTensor(
            indices=[[0, 0]],
            values=['key1'],
            dense_shape=[1, 1]
        ), signature='context')

        expected_values_sequence = [[[2., 3.]]]
        lookup_values_sequence = lookup_values(tf.SparseTensor(
            indices=[[0, 0, 0]],
            values=['key1'],
            dense_shape=[1, 1, 1]
        ), signature='sequence')

        expected_values_all = [[0., 0.]] + source_values
        lookup_values_all = lookup_values([_UNIQUE_KEY_NAME] + source_keys)

        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            [
                result_values_default,
                result_values_context,
                result_values_sequence,
                result_values_all,
            ] = sess.run([
                lookup_values_default,
                lookup_values_context,
                lookup_values_sequence,
                lookup_values_all,
            ])

        self.assertListEqual(expected_values_default, result_values_default.tolist())
        self.assertListEqual(expected_values_context, result_values_context.tolist())
        self.assertListEqual(expected_values_sequence, result_values_sequence.tolist())
        self.assertListEqual(expected_values_all, result_values_all.tolist())

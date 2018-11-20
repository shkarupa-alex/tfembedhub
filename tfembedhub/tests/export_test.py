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

from ..export import read_and_split, export_hub_module


class TestReadAndSplit(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file_txt = os.path.join(self.temp_dir, 'source.txt')
        self.temp_file_npy = os.path.join(self.temp_dir, 'source.npy')

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def testTxt(self):
        source = '\n'.join([
            '<UNQ> 0. 0. 0.',
            'key1 1. 2. 3.',
            'key2 4. 5. 6.',
        ]) + '\n'
        with open(self.temp_file_txt, 'wb') as ttf:
            ttf.write(source.encode('utf-8'))

        expected_keys = ['<UNQ>', 'key1', 'key2']
        expected_values = [
            [0., 0., 0.],
            [1., 2., 3.],
            [4., 5., 6.],
        ]

        keys, values = read_and_split(self.temp_file_txt)
        self.assertListEqual(expected_keys, keys)
        self.assertListEqual(expected_values, values)


class TestExportHubModule(tf.test.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.source_keys = ['<UNQ>', 'key1', 'key2']
        self.source_values = [
            [0., 0., 0.],
            [1., 2., 3.],
            [4., 5., 6.],
        ]
        tf.reset_default_graph()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def testExportNative(self):
        export_hub_module(self.source_keys, self.source_values, self.temp_dir)
        lookup_values = hub.Module(self.temp_dir)

        expected_values_default = [1., 2., 3.]
        lookup_values_default = lookup_values(tf.SparseTensor(
            indices=[[0]],
            values=['key1'],
            dense_shape=[1]
        ))

        expected_values_context = [[1., 2., 3.]]
        lookup_values_context = lookup_values(tf.SparseTensor(
            indices=[[0, 0]],
            values=['key1'],
            dense_shape=[1, 1]
        ), signature='context')

        expected_values_sequence = [[[1., 2., 3.]]]
        lookup_values_sequence = lookup_values(tf.SparseTensor(
            indices=[[0, 0, 0]],
            values=['key1'],
            dense_shape=[1, 1, 1]
        ), signature='sequence')

        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            [
                result_values_default,
                result_values_context,
                result_values_sequence,
            ] = sess.run([
                lookup_values_default,
                lookup_values_context,
                lookup_values_sequence,
            ])

        self.assertListEqual(expected_values_default, result_values_default.tolist())
        self.assertListEqual(expected_values_context, result_values_context.tolist())
        self.assertListEqual(expected_values_sequence, result_values_sequence.tolist())

    def testExportNpy(self):
        export_hub_module(np.array(self.source_keys), np.array(self.source_values), self.temp_dir)
        lookup_values = hub.Module(self.temp_dir)

        expected_values_default = [1., 2., 3.]
        lookup_values_default = lookup_values(tf.SparseTensor(
            indices=[[0]],
            values=['key1'],
            dense_shape=[1]
        ))

        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            result_values_default = sess.run(lookup_values_default)

        self.assertListEqual(expected_values_default, result_values_default.tolist())

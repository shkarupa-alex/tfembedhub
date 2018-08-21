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

from .. import read_and_split, export_hub_module


class TestReadAndSplit(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, 'source.npy')

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def testNormal(self):
        source = np.array([
            ('<-UNIQUE->', 0., 0., 0.),
            ('key1', 1., 2., 3.),
            ('key2', 4., 5., 6.),
        ])
        np.save(self.temp_file, source)

        expected_keys = ['<-UNIQUE->', 'key1', 'key2']
        expected_values = [
            [0., 0., 0.],
            [1., 2., 3.],
            [4., 5., 6.],
        ]

        keys, values = read_and_split(self.temp_file)
        self.assertListEqual(expected_keys, keys.tolist())
        self.assertListEqual(expected_values, values.tolist())


class TestExportHubModule(tf.test.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.source_keys = np.array(['<-UNIQUE->', 'key1', 'key2'])
        self.source_values = np.array([
            [0., 0., 0.],
            [1., 2., 3.],
            [4., 5., 6.],
        ])
        tf.reset_default_graph()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def testExport(self):
        expected_key1_stats = [[1., 2., 3.]]

        export_hub_module(self.source_keys, self.source_values, self.temp_dir)

        stats_lookup = hub.Module(self.temp_dir)
        key1_stats = stats_lookup(['key1'])

        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            key1_stats_value = sess.run(key1_stats)

        self.assertListEqual(expected_key1_stats, key1_stats_value.tolist())

    def testLookup(self):
        expected_value = [
            [1., 2., 3.],
            [0., 0., 0.],
        ]

        export_hub_module(self.source_keys, self.source_values, self.temp_dir)
        stat_lookup_feature_column = hub.text_embedding_column(key="tokens", module_spec=self.temp_dir)

        features = {
            'tokens': tf.constant(['key1', 'key999'])
        }
        context_input = tf.feature_column.input_layer(features=features, feature_columns=[stat_lookup_feature_column])

        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            features_value = sess.run(context_input)

        self.assertListEqual(expected_value, features_value.tolist())

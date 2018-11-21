from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile
import tensorflow as tf

from ..export import export_hub_module
from ..column import text_embedding_column, sequence_text_embedding_column


class TestTextEmbeddingColumn(tf.test.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, 'source.txt')

        source = '\n'.join([
            'key1 1. 2. 3.',
            'key2 4. 5. 6.',
        ]) + '\n'
        with open(self.temp_file, 'wb') as ttf:
            ttf.write(source.encode('utf-8'))

        export_hub_module(self.temp_file, self.temp_dir)

        tf.reset_default_graph()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def testContextDense1D(self):
        expected_value = [
            [1., 2., 3.],
            [0., 0., 0.],
        ]

        features = {
            'tokens': ['key1', 'key999']
        }
        context_input = tf.feature_column.input_layer(
            features=features,
            feature_columns=[
                text_embedding_column(key='tokens', module_spec=self.temp_dir)
            ]
        )

        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            features_value = sess.run(context_input)

        self.assertListEqual(expected_value, features_value.tolist())

    def testContextSparse1D(self):
        expected_value = [
            [1., 2., 3.],
            [0., 0., 0.],
        ]

        features = {
            'tokens': tf.SparseTensor(
                indices=[[0], [1]],
                values=['key1', 'key999'],
                dense_shape=[2]
            )
        }
        context_input = tf.feature_column.input_layer(
            features=features,
            feature_columns=[
                text_embedding_column(key='tokens', module_spec=self.temp_dir)
            ]
        )

        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            features_value = sess.run(context_input)

        self.assertListEqual(expected_value, features_value.tolist())

    def testContextSparse2D(self):
        expected_value = [
            [2.5, 3.5, 4.5],
            [0., 0., 0.],
        ]

        features = {
            'tokens': tf.SparseTensor(
                indices=[[0, 0], [0, 1], [1, 0]],
                values=['key1', 'key2', 'key999'],
                dense_shape=[2, 2]
            )
        }
        context_input = tf.feature_column.input_layer(
            features=features,
            feature_columns=[
                text_embedding_column(key='tokens', module_spec=self.temp_dir)
            ]
        )

        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            features_value = sess.run(context_input)

        self.assertListEqual(expected_value, features_value.tolist())

    def testSequenceSparse2D(self):
        expected_value = [
            [[1., 2., 3.], [4., 5., 6.], [0., 0., 0.]],
            [[4., 5., 6.], [0., 0., 0.], [0., 0., 0.]],
        ]

        features = {
            'tokens': tf.SparseTensor(
                indices=[[0, 0], [0, 1], [0, 2], [1, 0]],
                values=['key1', 'key2', 'key999', 'key2'],
                dense_shape=[2, 3],
            ),
        }
        sequence_input = tf.contrib.feature_column.sequence_input_layer(
            features=features,
            feature_columns=[
                sequence_text_embedding_column(key='tokens', module_spec=self.temp_dir)
            ])

        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            features_value, features_length = sess.run(sequence_input)

        self.assertListEqual(expected_value, features_value.tolist())
        self.assertListEqual([3, 1], features_length.tolist())

    def testSequenceSparse3D(self):
        expected_value = [
            [[2.5, 3.5, 4.5], [0., 0., 0.]],
            [[4., 5., 6.], [0., 0., 0.]],
        ]

        features = {
            'tokens': tf.SparseTensor(
                indices=[[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]],
                values=['key1', 'key2', 'key999', 'key2'],
                dense_shape=[2, 2, 2],
            ),
        }

        sequence_input = tf.contrib.feature_column.sequence_input_layer(
            features=features,
            feature_columns=[
                sequence_text_embedding_column(key='tokens', module_spec=self.temp_dir)
            ])

        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            features_value, features_length = sess.run(sequence_input)

        self.assertListEqual(expected_value, features_value.tolist())
        self.assertListEqual([2, 1], features_length.tolist())

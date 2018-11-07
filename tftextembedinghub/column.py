from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.feature_column import feature_column
from tensorflow_hub import module


def text_embedding_column(key, module_spec):
    """Upgraded version of tensorflow_hub.text_embedding_column

    Returns:
      `_SequenceDenseColumn` that converts from text input.

    Raises:
       ValueError: if module_spec is not suitable for use in this feature column.
    """

    module_spec = module.as_module_spec(module_spec)
    _check_module_is_text_embedding(module_spec)

    return _TextEmbeddingColumn(
        key=key,
        module_spec=module_spec,
        signature='context',
    )


def sequence_text_embedding_column(key, module_spec):
    """Upgraded version of tensorflow_hub.text_embedding_column with sequential input support

    Returns:
      `_SequenceDenseColumn` that converts from text input.

    Raises:
       ValueError: if module_spec is not suitable for use in this feature column.
    """

    module_spec = module.as_module_spec(module_spec)
    _check_module_is_sequence_text_embedding(module_spec)

    return _TextEmbeddingColumn(
        key=key,
        module_spec=module_spec,
        signature='sequence',
    )


class _TextEmbeddingColumn(
    feature_column._DenseColumn,
    feature_column._SequenceDenseColumn,
    collections.namedtuple('_ModuleEmbeddingColumn', ('key', 'module_spec', 'signature'))):
    """Returned by text_embedding_column() or sequence_text_embedding_column(). Do not use directly."""

    @property
    def name(self):
        if not hasattr(self, '_name'):
            self._name = '{}_{}_hub_text_embedding'.format(self.key, self.signature)
        return self._name

    @property
    def _parse_example_spec(self):
        return {self.key: tf.VarLenFeature(tf.string)}

    def _transform_feature(self, inputs):
        input_tensor = inputs.get(self.key)
        input_tensor = feature_column._to_sparse_input_and_drop_ignore_values(input_tensor)

        if input_tensor.dtype != dtypes.string:
            raise ValueError('column {} dtype must be string, got: {}.'.format(self.key, input_tensor.dtype))

        return input_tensor

    @property
    def _variable_shape(self):
        if not hasattr(self, '_shape'):
            output_info = self.module_spec.get_output_info_dict(signature=self.signature)
            embed_size = output_info['default'].get_shape()[-1]
            self._shape = tf.TensorShape([embed_size])
        return self._shape

    @property
    def _hub_module(self):
        if not hasattr(self, '_module'):
            self._module = module.Module(self.module_spec, trainable=False)
        return self._module

    def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        del weight_collections, trainable

        if self.signature != 'context':
            raise ValueError(
                'Column {} could not be used as context feature column. '
                'Use text_embedding_column instead'.format(self.name)
            )

        sparse_keys = inputs.get(self)
        sparse_keys.shape.with_rank_at_least(1)
        sparse_keys.shape.with_rank_at_most(2)

        batch_size = sparse_keys.dense_shape[0]
        sparse_keys = tf.sparse_reshape(sparse_keys, shape=tf.stack([batch_size, -1]))

        return self._hub_module(sparse_keys, signature=self.signature)

    def _get_sequence_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        del weight_collections, trainable

        if self.signature != 'sequence':
            raise ValueError(
                'Column {} could not be used as sequence feature column. '
                'Use sequence_text_embedding_column instead'.format(self.name)
            )

        sparse_keys = inputs.get(self)
        sparse_keys.shape.with_rank_at_least(2)
        sparse_keys.shape.with_rank_at_most(3)

        batch_size, max_length = sparse_keys.dense_shape[0], sparse_keys.dense_shape[1]
        sparse_keys = tf.sparse_reshape(sparse_keys, shape=tf.stack([batch_size, max_length, -1]))

        dense_tensor = self._hub_module(sparse_keys, signature=self.signature)
        sequence_length = feature_column._sequence_length_from_sparse_tensor(inputs.get(self))

        return feature_column._SequenceDenseColumn.TensorSequenceLengthPair(
            dense_tensor=dense_tensor, sequence_length=sequence_length)


def _check_module_is_text_embedding(module_spec):
    """Raises ValueError if `module_spec` is not a text-embedding module.

    Args:
      module_spec: A `ModuleSpec` to test.

    Raises:
      ValueError: if `module_spec` default signature is not compatible with
      SparseTensor(string, shape=(?,)) -> Tensor(float32, shape=(?,K)).
    """
    issues = []

    # Find issues with signature inputs.
    input_info_dict = module_spec.get_input_info_dict(signature='context')
    if len(input_info_dict) != 1:
        issues.append('Module "context" signature must require only one input')
    else:
        input_info, = input_info_dict.values()
        input_shape = input_info.get_shape()
        if not (input_info.dtype == tf.string and
                input_info.is_sparse and
                input_shape.ndims == 2 and
                input_shape.as_list() == [None, None]):
            issues.append(
                'Module "context" signature must have only one input '
                'tf.SparseTensor(shape=(?,?), dtype=string)'
            )

    # Find issues with signature outputs.
    output_info_dict = module_spec.get_output_info_dict(signature='context')
    if 'default' not in output_info_dict:
        issues.append('Module "context" signature must have a "default" output.')
    else:
        output_info = output_info_dict['default']
        output_shape = output_info.get_shape()
        if not (output_info.dtype == tf.float32 and
                output_shape.ndims == 2 and
                not output_shape.as_list()[0] and
                output_shape.as_list()[1]):
            issues.append(
                'Module "context" signature must have only one output '
                'tf.Tensor(shape=(?,K), dtype=float32).'
            )

    if issues:
        raise ValueError('Module is not a context text-embedding: %r' % issues)

def _check_module_is_sequence_text_embedding(module_spec):
    """Raises ValueError if `module_spec` is not a text-embedding module.

    Args:
      module_spec: A `ModuleSpec` to test.

    Raises:
      ValueError: if `module_spec` default signature is not compatible with
      SparseTensor(string, shape=(?,)) -> Tensor(float32, shape=(?,K)).
    """
    issues = []

    # Find issues with signature inputs.
    input_info_dict = module_spec.get_input_info_dict(signature='sequence')
    if len(input_info_dict) != 1:
        issues.append('Module "sequence" signature must require only one input')
    else:
        input_info, = input_info_dict.values()
        input_shape = input_info.get_shape()
        if not (input_info.dtype == tf.string and
                input_info.is_sparse and
                input_shape.ndims == 3 and
                input_shape.as_list() == [None, None, None]):
            issues.append(
                'Module "sequence" signature must have only one input '
                'tf.SparseTensor(shape=(?,?,?), dtype=string)'
            )

    # Find issues with signature outputs.
    output_info_dict = module_spec.get_output_info_dict(signature='sequence')
    if 'default' not in output_info_dict:
        issues.append('Module "sequence" signature must have a "default" output.')
    else:
        output_info = output_info_dict['default']
        output_shape = output_info.get_shape()
        if not (output_info.dtype == tf.float32 and
                output_shape.ndims == 3 and
                not output_shape.as_list()[0] and
                not output_shape.as_list()[1] and
                output_shape.as_list()[2]):
            issues.append(
                'Module "sequence" signature must have only one output '
                'tf.Tensor(shape=(?,?,K), dtype=float32).'
            )

    if issues:
        raise ValueError('Module is not a sequence text-embedding: %r' % issues)

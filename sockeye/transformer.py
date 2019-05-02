# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import Dict, List, Optional, TYPE_CHECKING

import mxnet as mx
import numpy as np

from . import config
from . import constants as C
from . import layers
from .utils import check_condition

if TYPE_CHECKING:
    from . import encoder



class TransformerConfig(config.Config):

    def __init__(self,
                 model_size: int,
                 attention_heads: int,
                 feed_forward_num_hidden: int,
                 act_type: str,
                 proj_type: str,
                 num_layers: int,
                 dropout_attention: float,
                 dropout_enc_attention: List[float],
                 dropout_act: float,
                 dropout_prepost: float,
                 positional_embedding_type: str,
                 preprocess_sequence: str,
                 postprocess_sequence: str,
                 max_seq_len_source: int,
                 max_seq_len_target: int,
                 multisource_attention_type: str,
                 conv_config: Optional['encoder.ConvolutionalEmbeddingConfig'] = None,
                 lhuc: bool = False,
                 num_multisource: int = 1,
                 dtype: str = C.DTYPE_FP32) -> None:  # type: ignore
        super().__init__()
        self.model_size = model_size
        self.attention_heads = attention_heads
        self.feed_forward_num_hidden = feed_forward_num_hidden
        self.act_type = act_type
        self.proj_type = proj_type
        self.num_layers = num_layers
        self.dropout_enc_attention = dropout_enc_attention
        self.dropout_attention = dropout_attention
        self.dropout_act = dropout_act
        self.dropout_prepost = dropout_prepost
        self.positional_embedding_type = positional_embedding_type
        self.preprocess_sequence = preprocess_sequence
        self.postprocess_sequence = postprocess_sequence
        self.max_seq_len_source = max_seq_len_source
        self.max_seq_len_target = max_seq_len_target
        self.conv_config = conv_config
        self.use_lhuc = lhuc
        self.num_multisource = num_multisource
        self.dtype = dtype
        self.multisource_attention_type = multisource_attention_type


class TransformerEncoderBlock:
    """
    A transformer encoder block consists self-attention and a feed-forward layer with pre/post process blocks
    in between.
    """

    def __init__(self,
                 config: TransformerConfig,
                 prefix: str) -> None:
        self.pre_self_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                          dropout=config.dropout_prepost,
                                                          prefix="%satt_self_pre/" % prefix)
        self.self_attention = layers.MultiHeadSelfAttention(depth_att=config.model_size,
                                                            heads=config.attention_heads,
                                                            depth_out=config.model_size,
                                                            dropout=config.dropout_attention,
                                                            prefix="%satt_self/" % prefix)
        self.post_self_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                           dropout=config.dropout_prepost,
                                                           prefix="%satt_self_post/" % prefix)

        self.pre_ff = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                              dropout=config.dropout_prepost,
                                              prefix="%sff_pre/" % prefix)
        self.ff = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                         num_model=config.model_size,
                                         act_type=config.act_type,
                                         dropout=config.dropout_act,
                                         prefix="%sff/" % prefix)
        self.post_ff = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                               dropout=config.dropout_prepost,
                                               prefix="%sff_post/" % prefix)
        self.lhuc = None
        if config.use_lhuc:
            self.lhuc = layers.LHUC(config.model_size, prefix=prefix)

    def __call__(self, data: mx.sym.Symbol, bias: mx.sym.Symbol) -> mx.sym.Symbol:
        # self-attention
        data_self_att = self.self_attention(inputs=self.pre_self_attention(data, None),
                                            bias=bias,
                                            cache=None)
        data = self.post_self_attention(data_self_att, data)

        # feed-forward
        data_ff = self.ff(self.pre_ff(data, None))
        data = self.post_ff(data_ff, data)

        if self.lhuc:
            data = self.lhuc(data)

        return data


class TransformerDecoderBlock:
    """
    A transformer encoder block consists self-attention, encoder attention, and a feed-forward layer
    with pre/post process blocks in between.
    """

    def __init__(self,
                 config: TransformerConfig,
                 prefix: str) -> None:
        self.num_source = config.num_multisource

        # Self-Attention
        self.pre_self_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                          dropout=config.dropout_prepost,
                                                          prefix="%satt_self_pre/" % prefix)
        self.self_attention = layers.MultiHeadSelfAttention(depth_att=config.model_size,
                                                            heads=config.attention_heads,
                                                            depth_out=config.model_size,
                                                            dropout=config.dropout_attention,
                                                            prefix="%satt_self/" % prefix)
        self.post_self_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                           dropout=config.dropout_prepost,
                                                           prefix="%satt_self_post/" % prefix)

        # Encoder Attention
        self.pre_enc_attention = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                                         dropout=config.dropout_prepost,
                                                         prefix="%satt_enc_pre/" % prefix)
        check_condition(len(config.dropout_enc_attention) == config.num_multisource,
                "Not enough dropout encoder attention values")
        self.enc_attention = [ layers.MultiHeadAttention(depth_att=config.model_size,
                                                         heads=config.attention_heads,
                                                         depth_out=config.model_size,
                                                         dropout=config.dropout_enc_attention[i],
                                                         prefix="%satt_enc_source%d/" % (prefix, i))
                                                       for i, dropout in enumerate(config.dropout_enc_attention) ]
        self.multisource_attention = layers.get_multisource_attention(config, prefix)
        self.post_enc_attention = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                                          dropout=config.dropout_prepost,
                                                          prefix="%satt_enc_post/" % prefix)

        # FeedFoward
        self.pre_ff = TransformerProcessBlock(sequence=config.preprocess_sequence,
                                              dropout=config.dropout_prepost,
                                              prefix="%sff_pre/" % prefix)
        self.ff = TransformerFeedForward(num_hidden=config.feed_forward_num_hidden,
                                         num_model=config.model_size,
                                         act_type=config.act_type,
                                         dropout=config.dropout_act,
                                         prefix="%sff/" % prefix)
        self.post_ff = TransformerProcessBlock(sequence=config.postprocess_sequence,
                                               dropout=config.dropout_prepost,
                                               prefix="%sff_post/" % prefix)

        # Learning Hidden Unit Contribution
        self.lhuc = None
        if config.use_lhuc:
            self.lhuc = layers.LHUC(config.model_size, prefix=prefix)

    def __call__(self,
                 target: mx.sym.Symbol,
                 target_bias: mx.sym.Symbol,
                 source: mx.sym.Symbol,
                 source_bias: mx.sym.Symbol,
                 cache: Optional[Dict[str, Optional[mx.sym.Symbol]]] = None) -> mx.sym.Symbol:
        # target.infer_shape(target=(16,61)) => (batch_size, max_seq_length, depth)
        # target_bias.infer_shape() => ([], [(1, 61, 61)], [])
        # source.infer_shape(source=(16,2,61,1)) => (batch_size, num_source, max_seq_length, depth)
        # source_bias.infer_shape(source=(16,2,61,1)) => [(128, 2, 1, 61)]
        # len(self.enc_attention) = num_source

        # self-attention
        target_self_att = self.self_attention(inputs=self.pre_self_attention(target, None),
                                              bias=target_bias,
                                              cache=cache)
        target = self.post_self_attention(target_self_att, target)

        # encoder attention
        queries = self.pre_enc_attention(target, None)
        # queries.infer_shape(target=(16,61))   => [(16, 61, 32)]
        # queries = (batch_size, max_seq_length, depth)
        source_per_multisource      = mx.sym.split(data=source, axis=1, num_outputs=self.num_source, squeeze_axis=True)
        # len(source_per_multisource)   => 2 = num_multisource
        # source_per_multisource.infer_shape(source=(16,2,61,1))   => [(16, 61, 32), (16, 61, 32)]
        # source_per_multisource = [(batch_size, max_seq_length, depth), (batch_size, max_seq_length, depth)]
        source_bias_per_multisource = mx.sym.split(data=source_bias, axis=1, num_outputs=self.num_source, squeeze_axis=True)
        # len(source_bias_per_multisource)   => 2 = num_multisource
        # source_bias_per_multisource[0].infer_shape(source=(16,2,61,1))   => [(128, 1, 61)]
        # source_bias_per_multisource = (8 x batch_size, 1, max_seq_length)
        target_enc_atts = [ enc_attention(queries=queries, memory=_source, bias=_source_bias)
                                for enc_attention, _source, _source_bias in zip(self.enc_attention, source_per_multisource, source_bias_per_multisource) ]
        # target_enc_atts[0].infer_shape(target=(16,61), source=(16,2,61,1))   => [(16, 61, 32)]
        # target_enc_atts[0] = (batch, max_seq_length, output_depth)

        target_enc_att = self.multisource_attention(queries, target_enc_atts, target)

        target = self.post_enc_attention(target_enc_att, target)

        # feed-forward
        target_ff = self.ff(self.pre_ff(target, None))
        target    = self.post_ff(target_ff, target)

        if self.lhuc:
            target = self.lhuc(target)

        return target


class TransformerProcessBlock:
    """
    Block to perform pre/post processing on layer inputs.
    The processing steps are determined by the sequence argument, which can contain one of the three operations:
    n: layer normalization
    r: residual connection
    d: dropout
    """

    def __init__(self,
                 sequence: str,
                 dropout: float,
                 prefix: str) -> None:
        self.sequence = sequence
        self.dropout = dropout
        self.prefix = prefix
        self.layer_norm = None
        if "n" in sequence:
            self.layer_norm = layers.LayerNormalization(prefix="%snorm" % self.prefix)

    def __call__(self,
                 data: mx.sym.Symbol,
                 prev: Optional[mx.sym.Symbol]) -> mx.sym.Symbol:
        """
        Apply processing sequence to data with optional previous input.

        :param data: Input data. Shape: (batch, length, num_hidden).
        :param prev: Previous data. Shape: (batch, length, num_hidden).
        :return: Processed data. Shape: (batch, length, num_hidden).
        """
        if not self.sequence:
            return data

        if prev is None:
            assert 'r' not in self.sequence, "Residual connection not allowed if no previous value given."

        for step in self.sequence:

            if step == "r":
                data = mx.sym._internal._plus(data, prev, name="%sresidual" % self.prefix)

            elif step == "n":
                data = self.layer_norm(data=data)

            elif step == "d":
                if self.dropout > 0.0:
                    data = mx.sym.Dropout(data, p=self.dropout, name="%sdropout" % self.prefix)
            else:
                raise ValueError("Unknown step in sequence: %s" % step)

        return data


class TransformerFeedForward:
    """
    Position-wise feed-forward network with activation.
    """

    def __init__(self,
                 num_hidden: int,
                 num_model: int,
                 act_type: str,
                 dropout: float,
                 prefix: str) -> None:
        self.num_hidden = num_hidden
        self.num_model = num_model
        self.dropout = dropout
        self.prefix = prefix
        self.act_type = act_type
        self.w_i2h = mx.sym.Variable('%si2h_weight' % prefix)
        self.b_i2h = mx.sym.Variable('%si2h_bias' % prefix)
        self.w_h2o = mx.sym.Variable('%sh2o_weight' % prefix)
        self.b_h2o = mx.sym.Variable('%sh2o_bias' % prefix)

    def __call__(self, x) -> mx.sym.Symbol:
        """
        Position-wise feed-forward network with activation.

        :param x: Symbol of shape (batch_size, seq_len, num_hidden)
        :return: Symbol of shape (batch_size, seq_len, num_hidden)
        """
        h = mx.sym.FullyConnected(data=x, num_hidden=self.num_hidden, weight=self.w_i2h, bias=self.b_i2h, flatten=False)
        h = layers.activation(h, act_type=self.act_type)
        if self.dropout > 0.0:
            h = mx.sym.Dropout(h, p=self.dropout)
        y = mx.sym.FullyConnected(data=h, num_hidden=self.num_model, weight=self.w_h2o, bias=self.b_h2o, flatten=False)
        return y


class VariableLengthBias(mx.operator.CustomOp):
    """
    Returns bias/mask given a vector of sequence lengths.
    """

    def __init__(self, max_length: int) -> None:
        super().__init__()
        self.max_length = max_length

    def forward(self, is_train, req, in_data, out_data, aux):
        # lengths: (batch_size,)
        lengths = in_data[0]
        dtype = lengths.dtype
        dtype_str = np.dtype(dtype).name

        # (batch_size, max_length)
        data = mx.nd.zeros((lengths.shape[0], self.max_length), dtype=dtype, ctx=lengths.context)
        data = mx.nd.SequenceMask(data=data,
                                  use_sequence_length=True,
                                  sequence_length=lengths,
                                  axis=1,
                                  value=-C.LARGE_VALUES[dtype_str])
        self.assign(out_data[0], req[0], data)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass


@mx.operator.register("variable_length_bias")
class VariableLengthBiasProp(mx.operator.CustomOpProp):

    def __init__(self, max_length: str) -> None:
        super().__init__()
        self.max_length = int(max_length)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        batch_size = in_shape[0][0]
        return in_shape, [(batch_size, self.max_length)], []

    def infer_type(self, in_type):
        return in_type, in_type, []

    def create_operator(self, ctx, shapes, dtypes):
        return VariableLengthBias(max_length=self.max_length)


def get_variable_length_bias(lengths: mx.sym.Symbol,
                             max_length: int,
                             num_heads: Optional[int] = None,
                             fold_heads: bool = True,
                             name: str = '') -> mx.sym.Symbol:
    """
    Returns bias/mask for variable sequence lengths.

    :param lengths: Sequence lengths. Shape: (batch,).
    :param max_length: Maximum sequence length.
    :param num_heads: Number of attention heads.
    :param fold_heads: Whether to fold heads dimension into batch dimension.
    :param name: Name of symbol.
    :return: Bias symbol.
    """
    # (batch_size, max_length)
    x = mx.symbol.Custom(data=lengths, max_length=max_length, op_type='variable_length_bias')
    if num_heads is not None:
        # (batch_size, heads, max_length) if fold_heads == False else (batch_size * heads, max_length)
        x = layers.broadcast_to_heads(x, num_heads, ndim=2, fold_heads=fold_heads)
    return mx.sym.BlockGrad(x, name='%sbias' % name)


def get_autoregressive_bias(max_length: int, name: str) -> mx.sym.Symbol:
    """
    Returns bias/mask to ensure position i can only attend to positions <i.

    :param max_length: Sequence length.
    :param name: Name of symbol.
    :return: Bias symbol of shape (1, max_length, max_length).
    """
    return mx.sym.BlockGrad(mx.symbol.Custom(length=max_length,
                                             name=name,
                                             op_type='auto_regressive_bias'))


class AutoRegressiveBias(mx.operator.CustomOp):
    """
    Returns a symbol of shape (1, length, length) with cells above the main diagonal
    set to a large negative value, e.g.
    length=4

    0 1 1 1
    0 0 1 1   * LARGE_NEGATIVE_VALUE
    0 0 0 1
    0 0 0 0
    """

    def __init__(self, length: int, dtype: str, ctx: mx.Context) -> None:
        super().__init__()
        self.bias = self.get_bias(length, dtype, ctx)

    @staticmethod
    def get_bias(length: int, dtype: str, ctx: mx.Context):
        # matrix with lower triangle and main diagonal set to 0, upper triangle set to 1
        upper_triangle = np.triu(np.ones((length, length), dtype=dtype), k=1)
        # (1, length, length)
        bias = -C.LARGE_VALUES[dtype] * np.reshape(upper_triangle, (1, length, length))
        return mx.nd.array(bias, ctx=ctx)

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], self.bias)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass


@mx.operator.register("auto_regressive_bias")
class AutoRegressiveBiasProp(mx.operator.CustomOpProp):

    def __init__(self, length: str, dtype: str = C.DTYPE_FP32) -> None:
        super().__init__()
        self.length = int(length)
        self.dtype = dtype

    def list_arguments(self):
        return []

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return [], [(1, self.length, self.length)], []

    def infer_type(self, in_type):
        return [], [np.dtype(self.dtype).type], []

    def create_operator(self, ctx, shapes, dtypes):
        return AutoRegressiveBias(length=self.length, dtype=self.dtype, ctx=ctx)

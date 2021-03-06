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

import copy
import logging
import os
from typing import cast, Dict, Optional, Tuple, List

import mxnet as mx

from sockeye import __version__
from sockeye.config import Config
from . import constants as C
from . import data_io
from . import decoder
from . import encoder
from . import layers
from . import loss
from . import utils

logger = logging.getLogger(__name__)


class ModelConfig(Config):
    """
    ModelConfig defines model parameters defined at training time which are relevant to model inference.
    Add new model parameters here. If you want backwards compatibility for models trained with code that did not
    contain these parameters, provide a reasonable default under default_values.

    :param config_data: Used training data.
    :param vocab_source_size: Source vocabulary size.
    :param vocab_target_size: Target vocabulary size.
    :param config_embed_sources: Embedding config for sources.
    :param config_embed_target: Embedding config for target.
    :param config_encoders: Encoders' configuration.
    :param config_decoder: Decoder configuration.
    :param config_loss: Loss configuration.
    :param weight_tying: Enables weight tying if True.
    :param weight_tying_type: Determines which weights get tied. Must be set if weight_tying is enabled.
    :param lhuc: LHUC (Vilar 2018) is applied at some part of the model.
    """

    def __init__(self,
                 config_data: data_io.DataConfig,
                 vocab_source_size: List[int],
                 vocab_target_size: int,
                 config_embed_sources: List[encoder.EmbeddingConfig],
                 config_embed_target: encoder.EmbeddingConfig,
                 config_encoders: List[encoder.EncoderConfig],
                 config_decoder: decoder.DecoderConfig,
                 config_loss: loss.LossConfig,
                 multisource_attention_type: str,
                 weight_tying: bool = False,
                 weight_tying_type: Optional[str] = C.WEIGHT_TYING_TRG_SOFTMAX,
                 weight_normalization: bool = False,
                 lhuc: bool = False) -> None:
        super().__init__()
        self.config_data = config_data
        self.vocab_source_size = vocab_source_size
        self.vocab_target_size = vocab_target_size
        self.config_embed_sources = config_embed_sources
        self.config_embed_target = config_embed_target
        self.config_encoders = config_encoders
        self.config_decoder = config_decoder
        self.config_loss = config_loss
        self.weight_tying = weight_tying
        self.weight_tying_type = weight_tying_type
        self.weight_normalization = weight_normalization
        if weight_tying and weight_tying_type is None:
            raise RuntimeError("weight_tying_type must be specified when using weight_tying.")
        self.lhuc = lhuc
        self.multisource_attention_type = multisource_attention_type

    @property
    def num_source(self) -> int:
        return len(self.config_encoders)


class SockeyeModel:
    """
    SockeyeModel shares components needed for both training and inference.
    The main components of a Sockeye model are
    1) Source embedding
    2) Target embedding
    3) Encoder
    4) Decoder
    5) Output Layer

    ModelConfig contains parameters and their values that are fixed at training time and must be re-used at inference
    time.

    :param config: Model configuration.
    :param prefix: Name prefix for all parameters of this model.
    """

    def __init__(self, config: ModelConfig, prefix: str = '') -> None:
        self.config = copy.deepcopy(config)
        self.config.freeze()
        self.prefix = prefix
        logger.info("%s", self.config)
        num_sources = len(self.config.config_encoders)

        # encoder & decoder first (to know the decoder depth)
        self.encoders : List[encoder.Encoder] = []
        for i, config_encoder in enumerate(self.config.config_encoders):
            with mx.name.Prefix('multisource_%d' % i):
                enc = encoder.get_encoder(config_encoder, prefix=self.prefix + 'menc%d' % i)
                self.encoders.append(enc)

        self.decoder = decoder.get_decoder(self.config.config_decoder, prefix=self.prefix)

        # source & target embeddings
        embed_weight_source, embed_weight_target, out_weight_target = self._get_embed_weights(self.prefix)
        assert len(embed_weight_source) == len(self.config.config_embed_sources)
        self.embedding_source : List[encoder.Encoder] = []
        for i, (weights, config_embed) in enumerate(zip(embed_weight_source, self.config.config_embed_sources)):
            if isinstance(config_embed, encoder.PassThroughEmbeddingConfig):
                self.embedding_source.append(encoder.PassThroughEmbedding(config_embed))  # type: encoder.Encoder
            else:
                self.embedding_source.append(encoder.Embedding(config_embed,
                                                          prefix=self.prefix + C.SOURCE_EMBEDDING_PREFIX + '%d_'%i,
                                                          embed_weight=weights,
                                                          is_source=True))  # type: encoder.Encoder
        assert len(self.config.config_encoders) == len(self.embedding_source)

        self.embedding_target = encoder.Embedding(self.config.config_embed_target,
                                                  prefix=self.prefix + C.TARGET_EMBEDDING_PREFIX,
                                                  embed_weight=embed_weight_target)

        # multisource projection
        if self.config.multisource_attention_type == C.MULTISOURCE_ENCODER_COMBINATION and num_sources > 1:
            logger.info("Using multisource encoder2decoder projection matrix.")
            self.encoder2decoder = layers.OutputLayer(hidden_size=sum(encoder.get_num_hidden() for encoder in self.encoders),
                                                   vocab_size=self.decoder.get_num_hidden(),
                                                   weight = None,
                                                   weight_normalization=self.config.weight_normalization,
                                                   prefix=self.prefix + 'multisource_embedding_projection')
        else:
            self.encoder2decoder = None

        # output layer
        self.output_layer = layers.OutputLayer(hidden_size=self.decoder.get_num_hidden(),
                                               vocab_size=self.config.vocab_target_size,
                                               weight=out_weight_target,
                                               weight_normalization=self.config.weight_normalization,
                                               prefix=self.prefix + C.DEFAULT_OUTPUT_LAYER_PREFIX)

        self.params = None  # type: Optional[Dict]
        self.aux_params = None  # type: Optional[Dict]

    def save_config(self, folder: str):
        """
        Saves model configuration to <folder>/config

        :param folder: Destination folder.
        """
        fname = os.path.join(folder, C.CONFIG_NAME)
        self.config.save(fname)
        logger.info('Saved config to "%s"', fname)

    @staticmethod
    def load_config(fname: str) -> ModelConfig:
        """
        Loads model configuration.

        :param fname: Path to load model configuration from.
        :return: Model configuration.
        """
        config = ModelConfig.load(fname)
        logger.info('ModelConfig loaded from "%s"', fname)
        return cast(ModelConfig, config)  # type: ignore

    def save_params_to_file(self, fname: str):
        """
        Saves model parameters to file.

        :param fname: Path to save parameters to.
        """
        if self.aux_params is not None:
            utils.save_params(self.params.copy(), fname, self.aux_params.copy())
        else:
            utils.save_params(self.params.copy(), fname)
        logging.info('Saved params to "%s"', fname)

    def load_params_from_file(self, fname: str):
        """
        Loads and sets model parameters from file.

        :param fname: Path to load parameters from.
        """
        utils.check_condition(os.path.exists(fname), "No model parameter file found under %s. "
                                                     "This is either not a model directory or the first training "
                                                     "checkpoint has not happened yet." % fname)
        self.params, self.aux_params = utils.load_params(fname)
        utils.check_condition(all(name.startswith(self.prefix) for name in self.params.keys()),
                              "Not all parameter names start with model prefix '%s'" % self.prefix)
        utils.check_condition(all(name.startswith(self.prefix) for name in self.aux_params.keys()),
                              "Not all auxiliary parameter names start with model prefix '%s'" % self.prefix)
        logger.info('Loaded params from "%s"', fname)

    @staticmethod
    def save_version(folder: str):
        """
        Saves version to <folder>/version.

        :param folder: Destination folder.
        """
        fname = os.path.join(folder, C.VERSION_NAME)
        with open(fname, "w") as out:
            out.write(__version__)

    def _get_embed_weights(self, prefix: str) -> Tuple[List[mx.sym.Symbol], mx.sym.Symbol, mx.sym.Symbol]:
        """
        Returns embedding parameters for source and target.
        When source and target embeddings are shared, they are created here and passed in to each side,
        instead of being created in the Embedding constructors.

        :param prefix: Prefix.
        :return: Tuple of source and target parameter symbols.
        """
        w_embed_source = [ mx.sym.Variable(prefix + C.SOURCE_EMBEDDING_PREFIX + "%d_weight" % i,
                                         shape=(config.vocab_size, config.num_embed)) 
                            for i, config in enumerate(self.config.config_embed_sources) ]
        w_embed_target = mx.sym.Variable(prefix + C.TARGET_EMBEDDING_PREFIX + "weight",
                                         shape=(self.config.config_embed_target.vocab_size,
                                                self.config.config_embed_target.num_embed))

        w_out_target = mx.sym.Variable(prefix + "target_output_weight",
                                       shape=(self.config.vocab_target_size,
                                              self.decoder.get_num_hidden()))

        if self.config.weight_tying:
            if C.WEIGHT_TYING_SRC in self.config.weight_tying_type \
                    and C.WEIGHT_TYING_TRG in self.config.weight_tying_type:
                utils.check_condition(all(config.vocab_size == self.config.config_embed_target.vocab_size
                    for config in self.config.config_embed_sources),
                    "Incompatible vocab sizes between sources and target.")
                utils.check_condition(all(config.num_embed == self.config.config_embed_target.num_embed
                    for config in self.config.config_embed_sources),
                    "Incompatible embed sizes between sources and target.")
                logger.info("Tying the source and target embeddings.")
                # TODO: Sam Implement weight tying with multiple sources.  Should we only tie vocabularies for the main factor and not the other factors?
                del w_embed_source
                del w_embed_target
                w_embed_target = mx.sym.Variable(prefix + C.SHARED_EMBEDDING_PREFIX + "weight",
                                                 shape=(self.config.config_embed_target.vocab_size,
                                                        self.config.config_embed_target.num_embed))
                w_embed_source = [w_embed_target] * len(self.config.config_embed_sources)

            if C.WEIGHT_TYING_SOFTMAX in self.config.weight_tying_type:
                logger.info("Tying the target embeddings and output layer parameters.")
                # TODO: Sam Implement weight tying with multiple sources.
                utils.check_condition(self.config.config_embed_target.num_embed == self.decoder.get_num_hidden(),
                                      "Weight tying requires target embedding size and decoder hidden size " +
                                      "to be equal: %d vs. %d" % (self.config.config_embed_target.num_embed,
                                                                  self.decoder.get_num_hidden()))
                del w_out_target
                w_out_target = w_embed_target

        self._embed_weight_source_name = None
        # TODO: Sam how can w_embed_source be None?
        if w_embed_source is not None:
            self._embed_weight_source_name = [ weights.name for weights in w_embed_source ]
        self._embed_weight_target_name = w_embed_target.name
        self._out_weight_target_name = w_out_target.name
        return w_embed_source, w_embed_target, w_out_target

    def get_source_embed_params(self) -> Optional[mx.nd.NDArray]:
        if self.params is None:
            return None
        return self.params.get(self._embed_weight_source_name)

    def get_target_embed_params(self) -> Optional[mx.nd.NDArray]:
        if self.params is None:
            return None
        return self.params.get(self._embed_weight_target_name)

    def get_output_embed_params(self) -> Optional[mx.nd.NDArray]:
        if self.params is None:
            return None
        return self.params.get(self._out_weight_target_name)

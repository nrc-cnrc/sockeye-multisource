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

"""
Implements a thin wrapper around Translator to compute BLEU scores on (a sample of) validation data during training.
"""
import logging
import os
import random
import time
from contextlib import ExitStack
from typing import Any, Dict, Optional, List, Sequence, Tuple

import mxnet as mx

import sockeye.output_handler
import sockeye.translate
from . import constants as C
from . import data_io
from . import evaluate
from . import inference
from . import utils

logger = logging.getLogger(__name__)


class CheckpointDecoder:
    """
    Decodes a (random sample of a) dataset using parameters at given checkpoint and computes BLEU against references.

    :param context: MXNet context to bind the model to.
    :param inputs: Path(s) to file containing input sentences (and their factors).
    :param references: Path to file containing references.
    :param model: Model to load.
    :param max_input_len: Maximum input length.
    :param batch_size: Batch size.
    :param beam_size: Size of the beam.
    :param nbest_size: Size of nbest lists.
    :param bucket_width_source: Source bucket width.
    :param length_penalty_alpha: Alpha factor for the length penalty
    :param length_penalty_beta: Beta factor for the length penalty
    :param softmax_temperature: Optional parameter to control steepness of softmax distribution.
    :param max_output_length_num_stds: Number of standard deviations as safety margin for maximum output length.
    :param ensemble_mode: Ensemble mode: linear or log_linear combination.
    :param sample_size: Maximum number of sentences to sample and decode. If <=0, all sentences are used.
    :param random_seed: Random seed for sampling. Default: 42.
    """

    def __init__(self,
                 context: mx.context.Context,
                 multisource: List[List[str]],
                 references: str,
                 model: str,
                 max_input_len: Optional[int] = None,
                 batch_size: int = 16,
                 beam_size: int = C.DEFAULT_BEAM_SIZE,
                 nbest_size: int = C.DEFAULT_NBEST_SIZE,
                 bucket_width_source: int = 10,
                 length_penalty_alpha: float = 1.0,
                 length_penalty_beta: float = 0.0,
                 softmax_temperature: Optional[float] = None,
                 max_output_length_num_stds: int = C.DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH,
                 ensemble_mode: str = 'linear',
                 sample_size: int = -1,
                 random_seed: int = 42) -> None:
        logger.info('{} {} {} {} {}'.format(context, multisource, references, model, max_input_len))
        self.context = context
        self.max_input_len = max_input_len
        self.max_output_length_num_stds = max_output_length_num_stds
        self.ensemble_mode = ensemble_mode
        self.beam_size = beam_size
        self.nbest_size = nbest_size
        self.batch_size = batch_size
        self.bucket_width_source = bucket_width_source
        self.length_penalty_alpha = length_penalty_alpha
        self.length_penalty_beta = length_penalty_beta
        self.softmax_temperature = softmax_temperature
        self.model = model

        with ExitStack() as exit_stack:
            inputs_fins = [ [ exit_stack.enter_context(data_io.smart_open(factor)) for factor in source ] for source in multisource ]
            references_fin = exit_stack.enter_context(data_io.smart_open(references))

            inputs_sentences = [ [ factor.readlines() for factor in source ] for source in inputs_fins ]
            target_sentences = references_fin.readlines()

            utils.check_condition(all(all(len(l) == len(target_sentences) for l in source) for source in inputs_sentences),
                                  "Sentences differ in length")

            num_samples = len(inputs_sentences[0][0])
            if sample_size <= 0:
                sample_size = num_samples

            if sample_size < num_samples:
                self.inputs_sentences, self.target_sentences = parallel_subsample(inputs_sentences, target_sentences, sample_size, random_seed)
            else:
                self.inputs_sentences, self.target_sentences = inputs_sentences, target_sentences

            if sample_size < self.batch_size:
                self.batch_size = sample_size

        for s, source in enumerate(self.inputs_sentences):
            for f, factor in enumerate(source):
                write_to_file(factor, os.path.join(self.model, C.DECODE_IN_NAME % (s, f)))
        write_to_file(self.target_sentences, os.path.join(self.model, C.DECODE_REF_NAME))

        self.inputs_sentences = list(zip(*(zip(*l) for l in self.inputs_sentences)))  # type: Sequence[Sequence[Sequence[str]]]
        # self.inputs_sentences => num_samples, num_sources, num_factors
        #self.inputs_sentences = list(zip(*self.inputs_sentences))  # type: List[List[str]]

        logger.info("Created CheckpointDecoder(max_input_len=%d, beam_size=%d, model=%s, num_sentences=%d, context=%s)",
                    max_input_len if max_input_len is not None else -1, beam_size, model, len(self.target_sentences),
                    context)

    def decode_and_evaluate(self,
                            checkpoint: Optional[int] = None,
                            output_name: str = os.devnull) -> Dict[str, float]:
        """
        Decodes data set and evaluates given a checkpoint.

        :param checkpoint: Checkpoint to load parameters from.
        :param output_name: Filename to write translations to. Defaults to /dev/null.
        :return: Mapping of metric names to scores.
        """
        models, source_vocabs, target_vocab = inference.load_models(
            self.context,
            self.max_input_len,
            self.beam_size,
            self.batch_size,
            [self.model],
            [checkpoint],
            softmax_temperature=self.softmax_temperature,
            max_output_length_num_stds=self.max_output_length_num_stds)
        translator = inference.Translator(context=self.context,
                                          ensemble_mode=self.ensemble_mode,
                                          bucket_source_width=self.bucket_width_source,
                                          length_penalty=inference.LengthPenalty(self.length_penalty_alpha, self.length_penalty_beta),
                                          beam_prune=0.0,
                                          beam_search_stop='all',
                                          nbest_size=self.nbest_size,
                                          models=models,
                                          source_vocabs=source_vocabs,
                                          target_vocab=target_vocab,
                                          restrict_lexicon=None,
                                          store_beam=False)
        trans_wall_time = 0.0
        translations = []
        with data_io.smart_open(output_name, 'w') as output:
            handler = sockeye.output_handler.StringOutputHandler(output)
            tic = time.time()
            trans_inputs = []  # type: List[inference.TranslatorInput]
            # TODO: Sam are we expecting the source to change?  Why are we rebuilding trans_inputs?  Shouldn't it be computed once at __init__?
            for i, inputs in enumerate(self.inputs_sentences):
                trans_inputs.append(sockeye.inference.make_input_from_multiple_strings(i, inputs))
            logger.debug("CheckpointDecoder.decode_and_evaluate trans_inputs: " + str(trans_inputs))
            trans_outputs = translator.translate(trans_inputs)
            logger.debug("CheckpointDecoder.decode_and_evaluate trans_outputs: " + str(trans_outputs))
            trans_wall_time = time.time() - tic
            for trans_input, trans_output in zip(trans_inputs, trans_outputs):
                handler.handle(trans_input, trans_output)
                translations.append(trans_output.translation)
        avg_time = trans_wall_time / len(self.target_sentences)

        # TODO(fhieber): eventually add more metrics (METEOR etc.)
        return {C.BLEU_VAL: evaluate.raw_corpus_bleu(hypotheses=translations,
                                                     references=self.target_sentences,
                                                     offset=0.01),
                C.CHRF_VAL: evaluate.raw_corpus_chrf(hypotheses=translations,
                                                     references=self.target_sentences),
                C.ROUGE_1_VAL: evaluate.raw_corpus_rouge1(hypotheses=translations,
                                                          references=self.target_sentences),
                C.ROUGE_2_VAL: evaluate.raw_corpus_rouge2(hypotheses=translations,
                                                          references=self.target_sentences),
                C.ROUGE_L_VAL: evaluate.raw_corpus_rougel(hypotheses=translations,
                                                          references=self.target_sentences),
                C.AVG_TIME: avg_time,
                C.DECODING_TIME: trans_wall_time}


def parallel_subsample(multisource: Sequence[Sequence[Sequence[str]]], target: Sequence[str], sample_size: int, seed: int) -> Tuple[List[List[List[str]]], List[str]]:
    # custom random number generator to guarantee the same samples across runs in order to be able to
    # compare metrics across independent runs
    multisource_sample = []  # type: List[List[List[str]]]
    for source in multisource:
        sample = []
        for factor in source:
            random_gen = random.Random(seed)
            sample.append(random_gen.sample(factor, sample_size))
        multisource_sample.append(sample)

    random_gen = random.Random(seed)
    target_sample = random_gen.sample(target, sample_size)

    return multisource_sample, target_sample


def write_to_file(data: List[str], fname: str):
    with data_io.smart_open(fname, 'w') as f:
        for x in data:
            print(x.rstrip(), file=f)






if __name__ == '__main__':
    cp = CheckpointDecoder(
            context=mx.cpu(),
            multisource=[['corpora.2/dev.en'], ['corpora.2/dev.de']],
            references='corpora.2/dev.fr',
            model='model',
            max_input_len = 60)

    cp.decode_and_evaluate()

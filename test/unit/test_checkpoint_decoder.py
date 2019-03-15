import pytest
import mxnet as mx

import sockeye.checkpoint_decoder


source_0_0 = [ 'source_0_0-%d' % i for i in range(10) ]
source_0_1 = [ 'source_0_1-%d' % i for i in range(10) ]
source_0_2 = [ 'source_0_2-%d' % i for i in range(10) ]

source_1_0 = [ 'source_1_0-%d' % i for i in range(10) ]
source_1_1 = [ 'source_1_1-%d' % i for i in range(10) ]
source_1_2 = [ 'source_1_2-%d' % i for i in range(10) ]

target = [ 'target-%d' % i for i in range(10) ]

@pytest.mark.parametrize("multisource, target, sample_size, seed, expected", [
    ( # a single source with no factor
        (
            # multisource source 0 with its factors
            ( source_0_0, ),
        ),
        target,
        3,
        42,
        (
            [
                [ # source 0
                    ['source_0_0-1', 'source_0_0-0', 'source_0_0-4'],  # multisource source 0 factor 0
                ],
            ],
            ['target-1', 'target-0', 'target-4']
        )
    ),
    ( # a single source with multiple factors
        (
            # multisource source 0 with its factors
            ( source_0_0, source_0_1, source_0_2),
        ),
        target,
        3,
        42,
        (
            [
                [ # multisource source 0
                    ['source_0_0-1', 'source_0_0-0', 'source_0_0-4'],  # multisource source 0 factor 0
                    ['source_0_1-1', 'source_0_1-0', 'source_0_1-4'],  # multisource source 0 factor 1
                    ['source_0_2-1', 'source_0_2-0', 'source_0_2-4'],  # multisource source 0 factor 2
                ],
            ],
            ['target-1', 'target-0', 'target-4']
        )
    ),
    (  # multisource with no factor
        (
            # multisource source 0 with no factor
            ( source_0_0, ),
            # multisource source 1 with no factor
            ( source_1_0, )
        ),
        target,
        3,
        42,
        (
            [
                [ # multisource source 0
                    ['source_0_0-1', 'source_0_0-0', 'source_0_0-4'],  # multisource source 0 no factor
                ],
                [ # multisource source 1
                    ['source_1_0-1', 'source_1_0-0', 'source_1_0-4'],  # multisource source 1 no factor
                ],
            ],
            ['target-1', 'target-0', 'target-4']
        )
    ),
    (   # multisource with multiple factors
        (
            # multisource source 0 with its factors
            ( source_0_0, source_0_1, source_0_2),
            # multisource source 1 with its factors
            ( source_1_0, source_1_1, source_1_2)
        ),
        target,
        3,
        42,
        (
            [
                [ # multisource source 0
                    ['source_0_0-1', 'source_0_0-0', 'source_0_0-4'],  # multisource source 0 factor 0
                    ['source_0_1-1', 'source_0_1-0', 'source_0_1-4'],  # multisource source 0 factor 1
                    ['source_0_2-1', 'source_0_2-0', 'source_0_2-4'],  # multisource source 0 factor 2
                ],
                [ # multisource source 1
                    ['source_1_0-1', 'source_1_0-0', 'source_1_0-4'],  # multisource source 1 factor 0
                    ['source_1_1-1', 'source_1_1-0', 'source_1_1-4'],  # multisource source 1 factor 1
                    ['source_1_2-1', 'source_1_2-0', 'source_1_2-4'],  # multisource source 1 factor 2
                ],
            ],
            ['target-1', 'target-0', 'target-4']
        )
    ),
    ]) 
def test_convolutional_embedding_encoder(multisource, target, sample_size, seed, expected):
    """
    Test parallel sampling sentences of multisource with multiple factors.
    """
    sampled = sockeye.checkpoint_decoder.parallel_subsample(multisource, target, sample_size, seed)
    assert sampled == expected



@pytest.fixture(scope="session")
def multisource_with_factors(tmp_path_factory):
    """
    Creates 2 sources with three factors each having 10 sentences.
    """
    data_dir = tmp_path_factory.mktemp("data")
    target = data_dir / 'target'
    with target.open(mode='w') as f:
        for s in [ 'target-%d' % i for i in range(10) ]:
            print(s, file=f)
    multisource = []
    for s in range(2):
        factors = []
        for f in range(3):
            name = 'source_%d_%d' % (s, f)
            factor = data_dir / name
            factors.append(str(factor))
            with factor.open(mode='w') as f:
                for i in range(10):
                    print(name + '-%d' % i, file=f)
        multisource.append(factors)
    model = tmp_path_factory.mktemp('model')
    return model, multisource, str(target)



def test_simple_CheckpointDecoder(multisource_with_factors):
    """
    Test that the CheckpointDecoder has properly reorder its input sentences
    from #sources X #factors/source X sentences/factor
    to #sentences X #source X #factors/source
    """
    model, multisource, target = multisource_with_factors
    cd = sockeye.checkpoint_decoder.CheckpointDecoder(context=mx.context,
            multisource=multisource,
            references=target,
            model='model')
    # 10 sentence tuples each
    assert len(cd.target_sentences) == 10
    assert len(cd.inputs_sentences) == 10
    # Two sources
    assert len(cd.inputs_sentences[0]) == 2
    # Three factors each (per source)
    assert len(cd.inputs_sentences[0][0]) == 3
    assert len(cd.inputs_sentences[0][1]) == 3
    # First sentence, first source's factors
    assert cd.inputs_sentences[0][0][0].strip() == 'source_0_0-0'
    assert cd.inputs_sentences[0][0][1].strip() == 'source_0_1-0'
    assert cd.inputs_sentences[0][0][2].strip() == 'source_0_2-0'
    # First sentence, second source's factors
    assert cd.inputs_sentences[0][1][0].strip() == 'source_1_0-0'
    assert cd.inputs_sentences[0][1][1].strip() == 'source_1_1-0'
    assert cd.inputs_sentences[0][1][2].strip() == 'source_1_2-0'

"""test_legality.py - Unit tests for pass legality masking and structural preconditions."""

import pytest
from beampass.legality import IRInfo, legal_actions
from beampass.passes import Pass, PassVocabulary


@pytest.fixture
def sample_vocabulary():
    passes = [
        Pass(name="canonicalize", short="canon", iree_pass="canonicalize", max_count=2, requires=None),
        Pass(name="convert-conv-to-channels-last", short="CC", iree_pass="cc-pass", max_count=2, requires="has_conv"),
        Pass(name="convert-conv2d-to-img2col", short="I", iree_pass="i-pass", max_count=2, requires="has_conv"),
        Pass(name="generalize", short="gen", iree_pass="gen-pass", max_count=1, requires=None),
    ]
    return PassVocabulary(passes)


def test_per_pass_count_limit(sample_vocabulary):
    """Passes must be masked once they reach max_count applications."""
    # 'gen' has max_count = 1
    seq = ["gen"]
    legal = legal_actions(seq, vocabulary=sample_vocabulary)
    assert "gen" not in [p.short for p in legal]
    assert "canon" in [p.short for p in legal]
    assert "CC" in [p.short for p in legal]

    # 'canon' has max_count = 2
    seq2 = ["canon", "CC", "canon"]
    legal2 = legal_actions(seq2, vocabulary=sample_vocabulary)
    assert "canon" not in [p.short for p in legal2]


def test_structural_preconditions_mask_conv(sample_vocabulary):
    """Convolution-specific passes must be masked if model has no convolutions."""
    no_conv_ir = IRInfo(has_conv=False, has_matmul=True)
    legal = legal_actions([], vocabulary=sample_vocabulary, ir_info=no_conv_ir)
    legal_shorts = [p.short for p in legal]

    # 'CC' and 'I' require has_conv and must be masked
    assert "CC" not in legal_shorts
    assert "I" not in legal_shorts
    # 'canon' and 'gen' do not require conv
    assert "canon" in legal_shorts
    assert "gen" in legal_shorts


def test_consecutive_repeats_masking(sample_vocabulary):
    """Immediate repeats of the same pass must be masked when allow_consecutive_repeats=False."""
    seq = ["canon"]
    legal_no_repeat = legal_actions(seq, vocabulary=sample_vocabulary, allow_consecutive_repeats=False)
    assert "canon" not in [p.short for p in legal_no_repeat]

    legal_allow_repeat = legal_actions(seq, vocabulary=sample_vocabulary, allow_consecutive_repeats=True)
    assert "canon" in [p.short for p in legal_allow_repeat]


def test_deterministic_legal_order(sample_vocabulary):
    """Order of legal passes must be deterministic across calls."""
    seq = ["canon"]
    res1 = [p.short for p in legal_actions(seq, vocabulary=sample_vocabulary)]
    res2 = [p.short for p in legal_actions(seq, vocabulary=sample_vocabulary)]
    assert res1 == res2


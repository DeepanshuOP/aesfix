"""Regression test for bug 3: aggregator.py detected text-vs-image features
by checking for a "confidence" key. In pipeline.py, multi-comment samples are
first passed through confidence_weighted_aggregate, which never emitted that
key, so adaptive_fusion never fired for --mode run on any multi-comment
sample. The dispatch must use an explicit modality field instead, and that
field (plus a usable confidence signal) must survive the pre-aggregation
step.
"""
import pytest

from baselines.legacy_fuzzy.our_system.aggregator import (
    aggregate_critiques,
    confidence_weighted_aggregate,
)


def _text_feature(composition, composition_confidence):
    return {
        "composition": composition,
        "lighting": 0.5,
        "color": 0.5,
        "subject_clarity": 0.5,
        "technical": 0.5,
        "summary": "s",
        "confidence": {
            "composition": composition_confidence,
            "lighting": 0.0,
            "color": 0.0,
            "subject_clarity": 0.0,
            "technical": 0.0,
        },
        "modality": "text",
    }


_IMAGE_FEATURES = {
    "composition": 0.5, "lighting": 0.5, "color": 0.5,
    "subject_clarity": 0.5, "technical": 0.5, "modality": "image",
}


def test_multi_comment_sample_still_triggers_adaptive_fusion():
    """Reproduces pipeline.py's exact two-step flow for a multi-comment
    sample: comments are pre-aggregated, then fused with image features."""
    per_comment = [_text_feature(0.9, 0.8), _text_feature(0.9, 0.8)]
    text_features = confidence_weighted_aggregate(per_comment)
    assert text_features["modality"] == "text"

    agg, _disagreement, _subjectivity = aggregate_critiques([text_features, _IMAGE_FEATURES])

    # Adaptive fusion with wt=0.8 on composition: 0.8*0.9 + 0.2*0.5 = 0.82.
    # A silent fallback to confidence_weighted_aggregate would instead give
    # the plain mean of 0.9 and 0.5, i.e. 0.7.
    assert agg["composition"] == pytest.approx(0.82, abs=1e-3)


def test_single_comment_sample_still_triggers_adaptive_fusion():
    per_comment = [_text_feature(0.9, 0.8)]
    text_features = confidence_weighted_aggregate(per_comment)
    assert text_features["modality"] == "text"

    agg, _disagreement, _subjectivity = aggregate_critiques([text_features, _IMAGE_FEATURES])
    assert agg["composition"] == pytest.approx(0.82, abs=1e-3)


def test_dispatch_keys_off_modality_not_confidence_key_presence():
    """A text feature dict with no "confidence" key at all must still be
    routed as text (not silently treated as a second image), proving
    dispatch no longer duck-types on "confidence" in f."""
    text_no_confidence = {
        "composition": 0.9, "lighting": 0.5, "color": 0.5,
        "subject_clarity": 0.5, "technical": 0.5, "modality": "text",
    }
    agg, _d, _s = aggregate_critiques([text_no_confidence, _IMAGE_FEATURES])
    # adaptive_fusion() with confidence={} -> wt=0 for every key -> pure
    # image value (0.5). confidence_weighted_aggregate (the old fallback)
    # would instead certainty-weight 0.9 and 0.5 to 0.9. 0.5 proves the
    # adaptive_fusion branch was taken.
    assert agg["composition"] == pytest.approx(0.5)

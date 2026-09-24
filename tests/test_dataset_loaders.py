"""Tests for Standardized Dataset Loaders and Evaluation Harness (AF-0.3).

Why these tests verify the data layer:
1. Validates mathematical correctness of SRCC, PLCC, MAE, and MSE correlation metrics,
   including edge cases (zero-variance constant predictions, length mismatches).
2. Verifies that AADB's 11 raw attributes project into AesFix's 5 continuous aspects
   accurately (including the negative inversion of motion blur into technical quality).
3. Verifies that PCCD section headings map directly to corresponding aspect critique buckets.
4. Validates end-to-end evaluation harness execution against sample data fixtures.
"""
import json
import os
import tempfile
import pytest

from aesfix.data import (
    ASPECT_DIMENSIONS,
    AADBLoader,
    DatasetSample,
    EvaluationHarness,
    PCCDLoader,
    compute_correlation_metrics,
)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TEST_PROCESSED_JSON = os.path.join(_PROJECT_ROOT, "data", "processed_info_test.json")


def test_correlation_metrics_perfect_and_inverse_agreement():
    preds = [0.1, 0.4, 0.7, 0.9]
    targets = [0.1, 0.4, 0.7, 0.9]
    metrics = compute_correlation_metrics(preds, targets)

    assert metrics["srcc"] == pytest.approx(1.0)
    assert metrics["plcc"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx(0.0)

    # Inverse linear ordering should give exact negative correlation
    inverse_targets = [round(1.0 - p, 1) for p in preds]
    inv_metrics = compute_correlation_metrics(preds, inverse_targets)
    assert inv_metrics["srcc"] == pytest.approx(-1.0)
    assert inv_metrics["plcc"] == pytest.approx(-1.0)


def test_correlation_metrics_handles_edge_cases():
    # Fewer than 3 samples must fail loudly (insufficient statistical support)
    with pytest.raises(ValueError, match="At least 3 paired samples"):
        compute_correlation_metrics([0.5, 0.6], [0.5, 0.6])

    # Length mismatch must fail loudly
    with pytest.raises(ValueError, match="does not match"):
        compute_correlation_metrics([0.5, 0.6, 0.7], [0.5, 0.6])

    # Zero variance (constant predictions) must return 0.0 instead of crashing with NaN
    constant_preds = [0.5, 0.5, 0.5, 0.5]
    targets = [0.1, 0.3, 0.6, 0.8]
    zero_var_metrics = compute_correlation_metrics(constant_preds, targets)
    assert zero_var_metrics["srcc"] == 0.0
    assert zero_var_metrics["plcc"] == 0.0


def test_aadb_attribute_mapping():
    raw_attrs = {
        "rule_of_thirds": 0.8,
        "balancing_elements": 0.6,
        "symmetry": 0.7,
        "light": 0.85,
        "color_harmony": 0.9,
        "vivid_color": 0.7,
        "do_depth_of_field": 0.75,
        "content": 0.65,
        "motion_blur": 0.2,  # Motion blur degrades quality, so technical = 1.0 - 0.2 = 0.8
    }

    mapped = AADBLoader.map_attributes_to_aspects(raw_attrs)

    assert set(mapped.keys()) == set(ASPECT_DIMENSIONS)
    assert mapped["composition"] == pytest.approx((0.8 + 0.6 + 0.7) / 3.0, abs=1e-3)
    assert mapped["lighting"] == pytest.approx(0.85)
    assert mapped["color"] == pytest.approx((0.9 + 0.7) / 2.0, abs=1e-3)
    assert mapped["subject_clarity"] == pytest.approx((0.75 + 0.65) / 2.0, abs=1e-3)
    assert mapped["technical"] == pytest.approx(0.80)


def test_aadb_loader_from_json_and_csv():
    loader = AADBLoader()

    # Test JSON loading
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump([
            {
                "id": "aadb_001",
                "image_path": "img_001.jpg",
                "overall_score": 0.78,
                "attributes": {"light": 0.8, "motion_blur": 0.1, "rule_of_thirds": 0.9},
            }
        ], tf)
        tmp_json = tf.name

    try:
        samples = loader.load_from_json(tmp_json)
        assert len(samples) == 1
        assert samples[0].sample_id == "aadb_001"
        assert samples[0].overall_score == pytest.approx(0.78)
        assert samples[0].aspect_scores["lighting"] == pytest.approx(0.8)
        assert samples[0].aspect_scores["technical"] == pytest.approx(0.9)
    finally:
        os.remove(tmp_json)


def test_pccd_loader_sections():
    loader = PCCDLoader()

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump([
            {
                "id": "pccd_100",
                "image_path": "photo_100.jpg",
                "score": 0.82,
                "critiques": {
                    "Composition and Framing": "Well balanced using rule of thirds.",
                    "Use of Camera and Lighting": "Soft diffuse lighting from window.",
                    "Color and Tone": "Warm palette with saturated highlights.",
                    "Subject of the Photo": "Clear subject separation against bokeh.",
                    "General Impression": "A standout environmental portrait.",
                },
            }
        ], tf)
        tmp_json = tf.name

    try:
        samples = loader.load_from_json(tmp_json)
        assert len(samples) == 1
        sample = samples[0]
        assert sample.sample_id == "pccd_100"
        assert "rule of thirds" in sample.critiques["composition"][0]
        assert "diffuse lighting" in sample.critiques["lighting"][0]
        assert "standout" in sample.critiques["overall"][0]
    finally:
        os.remove(tmp_json)


def test_evaluation_harness_with_samples():
    sample1 = DatasetSample(
        sample_id="s1",
        overall_score=0.4,
        aspect_scores={"composition": 0.3, "lighting": 0.5},
    )
    sample2 = DatasetSample(
        sample_id="s2",
        overall_score=0.6,
        aspect_scores={"composition": 0.6, "lighting": 0.7},
    )
    sample3 = DatasetSample(
        sample_id="s3",
        overall_score=0.8,
        aspect_scores={"composition": 0.8, "lighting": 0.9},
    )

    harness = EvaluationHarness([sample1, sample2, sample3])

    def _mock_predictor(sample: DatasetSample):
        # Predict near-perfect ranking with slight offset
        pred_overall = sample.overall_score + 0.05
        pred_aspects = dict(sample.aspect_scores)
        return pred_overall, pred_aspects

    results = harness.evaluate_model(_mock_predictor)

    assert "overall" in results
    assert results["overall"]["srcc"] == pytest.approx(1.0)
    assert results["overall"]["plcc"] == pytest.approx(1.0)
    assert "composition" in results["aspects"]
    assert results["aspects"]["composition"]["srcc"] == pytest.approx(1.0)

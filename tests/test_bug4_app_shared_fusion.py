"""Regression test for bug 4: app.py did its own hardcoded 0.70/0.30 text/
image fusion instead of calling the shared aggregator, so the dashboard and
the CLI returned different scores for identical input. app.py must call
aggregate_critiques (the same path main.py/pipeline.py use).
"""
import pytest

from baselines.legacy_fuzzy import app
from baselines.legacy_fuzzy.our_system.aggregator import aggregate_critiques

_TEXT_FEATURES = {
    "composition": 0.9, "lighting": 0.3, "color": 0.5,
    "subject_clarity": 0.5, "technical": 0.5, "summary": "s",
    "confidence": {
        "composition": 0.8, "lighting": 0.6, "color": 0.0,
        "subject_clarity": 0.0, "technical": 0.0,
    },
    "modality": "text",
}
_IMAGE_FEATURES = {
    "composition": 0.4, "lighting": 0.7, "color": 0.5,
    "subject_clarity": 0.5, "technical": 0.5, "modality": "image",
}


class _FakeUploadedImage:
    def save(self, path):
        pass


def test_dashboard_calls_shared_aggregator_with_text_and_image(monkeypatch):
    calls = []
    real_aggregate = aggregate_critiques

    def _spy(feature_list):
        calls.append([dict(f) for f in feature_list])
        return real_aggregate(feature_list)

    monkeypatch.setattr(app, "aggregate_critiques", _spy)
    monkeypatch.setattr(app, "extract_image_features", lambda path: dict(_IMAGE_FEATURES))
    monkeypatch.setattr(app, "extract_aesthetic_features", lambda critique: dict(_TEXT_FEATURES))

    app.analyze(_FakeUploadedImage(), "some critique text")

    assert len(calls) == 1, "app.analyze() must fuse text+image through aggregate_critiques"
    assert calls[0][0]["modality"] == "text"
    assert calls[0][1]["modality"] == "image"


def test_dashboard_score_matches_cli_aggregator_not_hardcoded_weights(monkeypatch):
    captured = {}
    real_generate_features_html = app.generate_features_html

    def _capture(features):
        captured["features"] = dict(features)
        return real_generate_features_html(features)

    monkeypatch.setattr(app, "generate_features_html", _capture)
    monkeypatch.setattr(app, "extract_image_features", lambda path: dict(_IMAGE_FEATURES))
    monkeypatch.setattr(app, "extract_aesthetic_features", lambda critique: dict(_TEXT_FEATURES))

    app.analyze(_FakeUploadedImage(), "some critique text")

    expected_agg, _disagreement, _subjectivity = aggregate_critiques([_TEXT_FEATURES, _IMAGE_FEATURES])
    for key in ["composition", "lighting", "color", "subject_clarity", "technical"]:
        assert captured["features"][key] == pytest.approx(expected_agg[key])

    # The old hardcoded blend for composition: 0.70*0.9 + 0.30*0.4 = 0.75.
    # Confidence-weighted adaptive fusion (wt=0.8) gives 0.8*0.9+0.2*0.4=0.8.
    # These differ, so this pins down that the hardcoded formula is gone.
    hardcoded_composition = round(0.70 * _TEXT_FEATURES["composition"] + 0.30 * _IMAGE_FEATURES["composition"], 4)
    assert captured["features"]["composition"] != pytest.approx(hardcoded_composition)

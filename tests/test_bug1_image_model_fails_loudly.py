"""Regression test for bug 1: image_model.py used to swallow every exception
(including a failed CLIP weight load) and silently return all-0.5 default
scores, so the pipeline printed a confident MEDIUM verdict for a system that
had actually failed. extract_image_features must now raise instead.
"""
import os

import pytest

from baselines.legacy_fuzzy.our_system import image_model as im

_TEST_JPG = os.path.join(os.path.dirname(__file__), "..", "test.jpg")


def test_missing_image_path_raises_loudly_not_silent_defaults():
    with pytest.raises(im.ImageModelError):
        im.extract_image_features("no/such/file.jpg")


def test_model_load_failure_raises_loudly_not_silent_defaults(monkeypatch):
    def _boom():
        raise OSError("simulated: CLIP weights failed to download/load")

    monkeypatch.setattr(im, "_load_model", _boom)

    # A valid path, so we get past the path-existence check and into the part
    # of the function that used to catch this and return default_scores.
    with pytest.raises(im.ImageModelError):
        im.extract_image_features(_TEST_JPG)


def test_inference_failure_raises_loudly_not_silent_defaults(monkeypatch):
    monkeypatch.setattr(im, "_load_model", lambda: None)
    monkeypatch.setattr(
        im, "_get_text_embeddings",
        lambda: (_ for _ in ()).throw(RuntimeError("simulated inference failure")),
    )

    with pytest.raises(im.ImageModelError):
        im.extract_image_features(_TEST_JPG)

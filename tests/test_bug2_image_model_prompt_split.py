"""Regression test for bug 2: image_model.py hardcoded probs[:3]/probs[3:] to
split positive vs negative CLIP prompts, but subject_clarity has 6 positive
prompts (not 3) — so positives 4-6 were silently counted as negatives. The
split must use len(positive_prompts) per dimension instead.

We fake out the CLIP model/processor/text-embedding cache so this is a fast,
deterministic unit test with no real model inference, but reuse the real
AESTHETIC_PROMPTS positive/negative counts so it actually exercises the bug.
"""
import os

import torch

from baselines.legacy_fuzzy.our_system import image_model as im

_TEST_JPG = os.path.join(os.path.dirname(__file__), "..", "test.jpg")
_EMBED_DIM = 512
_LOGIT_SCALE = 10.0


class _FakeVisionOutputs:
    def __init__(self, pooler_output):
        self.pooler_output = pooler_output


class _FakeImage:
    def convert(self, mode):
        return self


class _FakeImageModule:
    @staticmethod
    def open(path):
        return _FakeImage()


class _FakeModel:
    """Image embedding fixed along axis 0; visual_projection is identity."""

    def __init__(self):
        embed = torch.zeros(1, _EMBED_DIM)
        embed[0, 0] = 1.0
        self._embed = embed
        self.logit_scale = torch.tensor(float(_LOGIT_SCALE)).log()

    def vision_model(self, pixel_values=None):
        return _FakeVisionOutputs(pooler_output=self._embed)

    def visual_projection(self, pooler_output):
        return pooler_output


def _make_fake_text_embeddings():
    """For every dimension, build (n_pos + n_neg) prompt embeddings using the
    REAL prompt counts from AESTHETIC_PROMPTS. The positive-prompt rows are
    aligned with the fake image embedding (axis 0, high similarity); the
    negative-prompt rows are orthogonal (axis 1, ~zero similarity). Only
    subject_clarity has n_pos != 3, so it's the one dimension a hardcoded
    probs[:3]/probs[3:] split gets wrong.
    """
    embeddings = {}
    for dim, prompts in im.AESTHETIC_PROMPTS.items():
        n_pos = len(prompts["positive"])
        n_neg = len(prompts["negative"])
        e = torch.zeros(n_pos + n_neg, _EMBED_DIM)
        e[:n_pos, 0] = 1.0
        e[n_pos:, 1] = 1.0
        embeddings[dim] = e
    return embeddings


def test_subject_clarity_uses_all_six_positive_prompts(monkeypatch):
    n_pos = len(im.AESTHETIC_PROMPTS["subject_clarity"]["positive"])
    n_neg = len(im.AESTHETIC_PROMPTS["subject_clarity"]["negative"])
    assert (n_pos, n_neg) == (6, 3), "fixture assumption about AESTHETIC_PROMPTS changed"

    monkeypatch.setattr(im, "_load_model", lambda: None)
    monkeypatch.setattr(im, "_model", _FakeModel())
    monkeypatch.setattr(
        im, "_processor",
        lambda images=None, return_tensors=None: {"pixel_values": torch.zeros(1, 3, 224, 224)},
    )
    monkeypatch.setattr(im, "Image", _FakeImageModule)
    monkeypatch.setattr(im, "_get_text_embeddings", _make_fake_text_embeddings)

    scores = im.extract_image_features(_TEST_JPG)

    # Correct split (n_pos=6): all 6 positive-prompt probabilities are high
    # and all 3 negative-prompt probabilities are ~0 -> raw_score ~1 -> ~0.8.
    #
    # Buggy split (hardcoded probs[:3]/probs[3:]): 3 of the true positive
    # rows (indices 3-5) get folded into the "negative" bucket, dragging the
    # negative mean up and the final score down to ~0.6.
    #
    # 0.75 sits strictly between the two, so this fails under the old code.
    assert scores["subject_clarity"] > 0.75

"""Unit and integration tests for Taste Fingerprint and Personalization Engine (AF-6.3 & Roadmap §21, §25 F4.4).

Why these tests are critical for user-centric aesthetic integrity:
1. Preference Model Convergence: Validates that Bayesian/Ridge Bradley-Terry optimization
   converges cleanly on pairwise comparison data without degenerate zero or infinite weights.
2. Honest Taste Divergence: Asserts that AesFix correctly identifies when an edit benefits
   general population aesthetic appeal but diverges from a creator's personal stylistic signature.
3. Creator Taste-Pack Portability: Verifies serialization round-trip for exportable Taste-Packs,
   enabling decentralized community aesthetic profiles.
"""
from __future__ import annotations

from pathlib import Path

from aesfix.critic.critic import CriticFix
from aesfix.judges.taste import (
    BUILTIN_TASTE_PACKS,
    DEFAULT_ASPECT_PRIORS,
    PairwiseComparison,
    TasteFingerprint,
    TastePack,
    TastePersonalizer,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TEST_JPG = str(_PROJECT_ROOT / "test.jpg")
_DARK_JPG = str(_PROJECT_ROOT / "dark.jpg")


def test_default_fingerprint_initialization() -> None:
    """Verifies that default fingerprint anchors to PARA priors with proper normalization."""
    fp = TasteFingerprint(user_id="user_test_init")

    assert fp.user_id == "user_test_init"
    assert len(fp.aspect_weights) == 5
    assert abs(sum(fp.aspect_weights.values()) - 1.0) < 1e-3
    for dim, prior in DEFAULT_ASPECT_PRIORS.items():
        assert abs(fp.aspect_weights[dim] - prior) < 1e-3

    # Initial confidence is low prior to user pairwise training
    assert fp.confidence_score < 0.20


def test_score_aspects_calculation() -> None:
    """Verifies personal aesthetic rating calculation S_user(I)."""
    fp = TasteFingerprint(user_id="user_scorer")
    aspects = {
        "composition": 0.80,
        "lighting": 0.70,
        "color": 0.60,
        "subject_clarity": 0.90,
        "technical": 0.85,
    }
    score = fp.score_aspects(aspects)
    assert 0.0 <= score <= 1.0
    assert 0.70 <= score <= 0.85


def test_pairwise_fitting_convergence() -> None:
    """Verifies that pairwise preference optimization learns user bias toward specific dimensions."""
    fp = TasteFingerprint(user_id="user_lighting_enthusiast")
    initial_light_weight = fp.aspect_weights["lighting"]

    # Synthesize 15 comparisons where image A has notably better lighting and user prefers A
    synthetic_pairs = [
        PairwiseComparison(
            image_a_path=_TEST_JPG,
            image_b_path=_DARK_JPG,
            preferred="A",
            aspect_diff={
                "composition": 0.0,
                "lighting": 0.35,  # Strong positive lighting delta
                "color": 0.05,
                "subject_clarity": 0.0,
                "technical": 0.0,
            },
        )
        for _ in range(15)
    ]

    updated_weights = fp.fit_from_pairwise_comparisons(synthetic_pairs)

    # Lighting weight should have increased above initial prior
    assert updated_weights["lighting"] > initial_light_weight
    assert abs(sum(updated_weights.values()) - 1.0) < 1e-3
    assert len(fp.history) == 15
    # Confidence score should have increased significantly with 15 pairs
    assert fp.confidence_score > 0.50


def test_compute_personal_gain_and_divergence() -> None:
    """Verifies that personal aesthetic gain correctly quantifies taste divergence."""
    # Profile that overwhelmingly cares about composition (0.50) over lighting (0.05)
    fp = TasteFingerprint(
        user_id="user_geometry_purist",
        aspect_weights={
            "composition": 0.50,
            "lighting": 0.05,
            "color": 0.15,
            "subject_clarity": 0.15,
            "technical": 0.15,
        },
    )

    orig_aspects = {"composition": 0.50, "lighting": 0.50, "color": 0.50, "subject_clarity": 0.50, "technical": 0.50}

    # Case 1: Edit improves composition -> aligns with user taste
    comp_edit = {"composition": 0.80, "lighting": 0.50, "color": 0.50, "subject_clarity": 0.50, "technical": 0.50}
    p_gain, div, badge = fp.compute_personal_gain(orig_aspects, comp_edit, global_gain=0.06)
    assert p_gain > 0.10
    assert div > 0.03
    assert "Aligns With Your Taste" in badge

    # Case 2: Edit improves lighting only -> helps globally but fights user taste focus
    light_edit = {"composition": 0.50, "lighting": 0.90, "color": 0.50, "subject_clarity": 0.50, "technical": 0.50}
    p_gain2, div2, badge2 = fp.compute_personal_gain(orig_aspects, light_edit, global_gain=0.08)
    assert div2 < -0.03
    assert "Fights Your Taste Signature" in badge2


def test_taste_pack_curated_and_export() -> None:
    """Verifies built-in taste packs and creator packaging export."""
    # 1. Built-in packs
    assert "cinematic_moody" in BUILTIN_TASTE_PACKS
    assert "high_key_commercial" in BUILTIN_TASTE_PACKS
    assert "minimalist_editorial" in BUILTIN_TASTE_PACKS

    moody = BUILTIN_TASTE_PACKS["cinematic_moody"]
    assert moody.aspect_weights["lighting"] >= 0.25
    assert moody.target_adjustments["ev_delta"] < 0.0  # Underexposed moody aesthetic

    # 2. Export personal fingerprint to creator pack
    fp = TasteFingerprint(user_id="creator_elena")
    pack = fp.export_taste_pack(
        pack_name="Nordic Minimal",
        creator="Elena Rostova",
        description="Cool tones, clean lines, and soft dynamic range.",
    )
    assert pack.pack_id.startswith("pack_creator_elena")
    assert pack.name == "Nordic Minimal"

    # 3. Serialization round-trip
    pack_dict = pack.to_dict()
    reconstructed = TastePack.from_dict(pack_dict)
    assert reconstructed.pack_id == pack.pack_id
    assert reconstructed.aspect_weights == pack.aspect_weights


def test_personalize_fix_ranking() -> None:
    """Verifies that candidate fixes are re-ranked based on user taste fingerprint."""
    # User who prioritizes subject clarity and composition
    fp = TasteFingerprint(
        user_id="user_portrait_photographer",
        aspect_weights={
            "composition": 0.35,
            "lighting": 0.10,
            "color": 0.10,
            "subject_clarity": 0.40,
            "technical": 0.05,
        },
    )

    orig_aspects = {
        "composition": 0.50,
        "lighting": 0.50,
        "color": 0.50,
        "subject_clarity": 0.50,
        "technical": 0.50,
    }

    candidate_fixes = [
        CriticFix(op="adjust_exposure", expected_aspect="lighting", nl_instruction="Boost midtones"),
        CriticFix(op="unsharp_mask", expected_aspect="subject_clarity", nl_instruction="Enhance eye sharpness"),
        CriticFix(op="smart_crop", expected_aspect="composition", nl_instruction="Tighten headshot framing"),
    ]

    ranked = TastePersonalizer.personalize_fix_ranking(
        candidate_fixes=candidate_fixes,
        original_aspects=orig_aspects,
        fingerprint=fp,
        global_weight=0.2,  # Strong personal weighting
    )

    # First ranked fix should be subject_clarity or composition, NOT lighting
    assert ranked[0][0].expected_aspect in ("subject_clarity", "composition")
    assert ranked[-1][0].expected_aspect == "lighting"
    # Ensure badges are attached
    assert "Personal Gain" in ranked[0][3]


def test_fingerprint_serialization() -> None:
    """Verifies serialization and restoration of TasteFingerprint profiles."""
    fp = TasteFingerprint(
        user_id="user_restore_test",
        aspect_weights={"composition": 0.3, "lighting": 0.3, "color": 0.2, "subject_clarity": 0.1, "technical": 0.1},
    )
    fp_dict = fp.to_dict()
    restored = TasteFingerprint.from_dict(fp_dict)

    assert restored.user_id == fp.user_id
    assert restored.aspect_weights == fp.aspect_weights
    assert restored.score_aspects({"composition": 0.8, "lighting": 0.6, "color": 0.5, "subject_clarity": 0.5, "technical": 0.5}) > 0.5

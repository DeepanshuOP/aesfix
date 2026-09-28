"""Unit tests for the Skill Curriculum and Longitudinal Trajectory Engine (AF-6.4).

Why these tests verify pedagogical and analytical integrity:
Validates that aspect scores correctly translate into growth trajectories, that the
bottleneck detector reliably isolates the weakest dimension, and that counterfactual
drills faithfully pull from the user's own historical uploads.
"""
from __future__ import annotations

import datetime
from typing import Dict, List

from aesfix.eval.curriculum import (
    ASPECT_TITLES,
    MasteryLevel,
    SkillCurriculumEngine,
    TrajectoryStatus,
    UserPhotoRecord,
)


def _make_dummy_records(n: int = 6, bias_aspect: str = "lighting") -> List[UserPhotoRecord]:
    """Helper to generate a sequence of mock upload telemetry across time."""
    records: List[UserPhotoRecord] = []
    base_time = datetime.datetime(2026, 8, 1, 10, 0, 0)

    for i in range(n):
        # Progressively improve composition and color, but keep bias_aspect low
        aspect_scores: Dict[str, float] = {
            "composition": 0.50 + (i * 0.05),  # 0.50 -> 0.75
            "lighting": 0.45 if bias_aspect == "lighting" else 0.70,
            "color": 0.55 + (i * 0.04),        # 0.55 -> 0.75
            "subject_clarity": 0.60 + (i * 0.03),
            "technical": 0.65,
        }
        overall = sum(aspect_scores.values()) / 5.0
        weakest = min(aspect_scores.keys(), key=lambda k: aspect_scores[k])

        records.append(
            UserPhotoRecord(
                photo_id=f"photo_{i:03d}",
                timestamp=base_time + datetime.timedelta(days=i * 2),
                image_path=f"/photos/raw/photo_{i:03d}.jpg",
                aspect_scores=aspect_scores,
                overall_score=overall,
                weakest_aspect=weakest,
                best_fix_name="Exposure & Tone Lift" if weakest == "lighting" else "Crop & Rule of Thirds",
                best_fix_path=f"/photos/fixed/photo_{i:03d}_fixed.jpg",
                verified_gain=0.18 if weakest == "lighting" else 0.10,
                metadata={"iso": 800, "f_stop": 2.8},
            )
        )
    return records


def test_user_photo_record_serialization() -> None:
    """Verifies that UserPhotoRecord roundtrips to and from dict without precision loss."""
    now = datetime.datetime.now()
    rec = UserPhotoRecord(
        photo_id="test_001",
        timestamp=now,
        image_path="/path/test.jpg",
        aspect_scores={"composition": 0.8, "lighting": 0.6},
        overall_score=0.7,
        weakest_aspect="lighting",
        best_fix_name="Shadow Recovery",
        verified_gain=0.12,
    )

    data = rec.to_dict()
    assert data["photo_id"] == "test_001"
    assert data["weakest_aspect"] == "lighting"
    assert data["verified_gain"] == 0.12

    restored = UserPhotoRecord.from_dict(data)
    assert restored.photo_id == rec.photo_id
    assert restored.aspect_scores == rec.aspect_scores
    assert restored.verified_gain == rec.verified_gain


def test_empty_history_defaults() -> None:
    """Verifies that an engine with zero history provides safe non-crashing defaults."""
    engine = SkillCurriculumEngine(user_history=[])
    graph = engine.build_skill_graph(user_id="new_user")

    assert graph.user_id == "new_user"
    assert graph.total_photos_analyzed == 0
    assert graph.mastery_level == MasteryLevel.NOVICE
    assert graph.bottleneck_aspect in ASPECT_TITLES

    drills = engine.generate_curriculum_drills(graph)
    assert len(drills) > 0
    assert drills[0].target_aspect == graph.bottleneck_aspect

    digest = engine.generate_weekly_digest(user_id="new_user")
    md = digest.to_markdown()
    assert "Weekly Studio Digest" in md
    assert "`0` images" in md


def test_trajectory_growth_and_status() -> None:
    """Verifies longitudinal trajectory tracking, growth percentage, and trend detection."""
    records = _make_dummy_records(n=6, bias_aspect="lighting")
    engine = SkillCurriculumEngine(records)

    trajectories = engine.compute_aspect_trajectories(baseline_window=3, recent_window=3)

    # Composition went from ~0.55 to ~0.70 -> should be 'improving'
    comp_traj = trajectories["composition"]
    assert comp_traj.recent_score > comp_traj.baseline_score
    assert comp_traj.growth_rate_pct > 10.0
    assert comp_traj.status == TrajectoryStatus.IMPROVING

    # Lighting stayed constant at 0.45 -> should be 'stagnant'
    light_traj = trajectories["lighting"]
    assert abs(light_traj.recent_score - light_traj.baseline_score) < 0.01
    assert light_traj.status == TrajectoryStatus.STAGNANT


def test_skill_graph_bottleneck_and_mastery() -> None:
    """Verifies that the skill graph identifies the empirical bottleneck and sets proper mastery tier."""
    records = _make_dummy_records(n=8, bias_aspect="lighting")
    engine = SkillCurriculumEngine(records)

    graph = engine.build_skill_graph(user_id="deepanshu_vit")

    assert graph.user_id == "deepanshu_vit"
    assert graph.total_photos_analyzed == 8
    # With lighting persistently at 0.45 while other aspects rise to >0.70, lighting is bottleneck
    assert graph.bottleneck_aspect == "lighting"
    # Highest aspect is composition or color
    assert graph.top_strength in ("composition", "color", "technical")
    assert graph.mastery_level in (MasteryLevel.INTERMEDIATE, MasteryLevel.ADVANCED)


def test_counterfactual_drill_generation() -> None:
    """Verifies that drills extract the user's own photos as counterfactual coaching exemplars."""
    records = _make_dummy_records(n=6, bias_aspect="lighting")
    engine = SkillCurriculumEngine(records)
    graph = engine.build_skill_graph(user_id="test_user")

    drills = engine.generate_curriculum_drills(graph, max_drills=2)
    assert len(drills) >= 1

    primary_drill = drills[0]
    assert primary_drill.target_aspect == "lighting"
    assert len(primary_drill.action_steps) >= 3

    # Check counterfactual exemplars pulled from user history
    assert len(primary_drill.counterfactual_exemplars) > 0
    first_ex = primary_drill.counterfactual_exemplars[0]
    assert first_ex.photo_id.startswith("photo_")
    assert first_ex.verified_gain > 0.0
    assert "photo_0" in first_ex.coaching_note


def test_weekly_digest_markdown_rendering() -> None:
    """Verifies that the weekly digest compiles into a clean, fully formatted Markdown email document."""
    records = _make_dummy_records(n=6, bias_aspect="lighting")
    engine = SkillCurriculumEngine(records)

    digest = engine.generate_weekly_digest(
        user_id="deepanshu_vit",
        digest_date=datetime.date(2026, 9, 28),
    )

    md = digest.to_markdown()
    assert "# 📸 AesFix Weekly Studio Digest" in md
    assert "deepanshu_vit" in md
    assert "2026-09-28" in md
    assert "Aspect Trajectories (Longitudinal Growth)" in md
    assert "Lighting & Dynamic Range" in md
    assert "Active Personal Coaching Drills" in md
    assert "Your Counterfactual Lessons" in md
    assert "Photo ID `photo_" in md


def test_high_skill_mastery_classification() -> None:
    """Verifies that an elite portfolio triggers the Master tier and appropriate drill scaling."""
    base_time = datetime.datetime(2026, 9, 1)
    elite_records: List[UserPhotoRecord] = []
    for i in range(5):
        elite_records.append(
            UserPhotoRecord(
                photo_id=f"master_{i}",
                timestamp=base_time + datetime.timedelta(days=i),
                image_path=f"/master_{i}.jpg",
                aspect_scores={
                    "composition": 0.92,
                    "lighting": 0.89,
                    "color": 0.90,
                    "subject_clarity": 0.88,
                    "technical": 0.94,
                },
                overall_score=0.91,
                weakest_aspect="subject_clarity",
            )
        )

    engine = SkillCurriculumEngine(elite_records)
    graph = engine.build_skill_graph(user_id="pro_photographer")

    assert graph.mastery_level == MasteryLevel.MASTER
    assert graph.overall_skill_index >= 0.88

    drills = engine.generate_curriculum_drills(graph)
    assert drills[0].difficulty == "mastery"

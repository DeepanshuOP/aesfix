"""Skill Curriculum Engine ("Duolingo for Photography" — AF-6.4 & Roadmap §25 F4.2, §23 F2.5).

Why this module exists:
Generic photo tutorials present stock examples disconnected from the user's specific habits.
AesFix coaches photographers through longitudinal aspect-score tracking across their own uploads,
identifying empirical bottleneck dimensions and prescribing targeted counterfactual drills
anchored to their own past photos where algorithmic fixes produced verified aesthetic gains.
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from aesfix.data.loaders import ASPECT_DIMENSIONS

logger = logging.getLogger("aesfix.curriculum")

# Canonical aspect domain labels and pedagogical descriptions
ASPECT_TITLES: Dict[str, str] = {
    "composition": "Composition & Framing",
    "lighting": "Lighting & Dynamic Range",
    "color": "Color Harmony & Palettes",
    "subject_clarity": "Subject Clarity & Separation",
    "technical": "Technical Precision & Acutance",
}

# Curated action plans mapped to each dimension when identified as a bottleneck
ASPECT_DRILL_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "lighting": {
        "title": "Dynamic Range & Exposure Balance",
        "description": "Master shadow detail recovery and highlight rolloff to prevent clipped whites and crushed blacks.",
        "action_steps": [
            "Meter for highlights in high-contrast scenes to preserve cloud and skin texture.",
            "Use ambient bounce or gentle fill-flash to lift deep facial shadows before post-processing.",
            "Inspect histogram post-shot to ensure luminance curves do not bunch against either boundary.",
        ],
    },
    "composition": {
        "title": "Visual Hierarchy & Leading Lines",
        "description": "Strengthen frame balance by placing key focal anchors along harmonic power points.",
        "action_steps": [
            "Activate 3x3 gridlines in viewfinder; align horizon with upper or lower third rather than center.",
            "Scan frame edges before triggering the shutter to eliminate distracting background amputations.",
            "Use natural architectural or environmental lines to guide viewer gaze directly to the subject.",
        ],
    },
    "color": {
        "title": "Chromatic Harmony & White Balance",
        "description": "Harmonize color palettes by calibrating color temperature and mitigating unnatural saturation casts.",
        "action_steps": [
            "Set custom Kelvin white balance when shooting mixed artificial and ambient lighting.",
            "Limit dominant frame hues to 2-3 complementary or analogous color families.",
            "Desaturate competing background neon/vivid elements to isolate subject tonality.",
        ],
    },
    "subject_clarity": {
        "title": "Subject Isolation & Depth Cues",
        "description": "Enhance spatial depth and ensure the primary narrative subject immediately commands viewer attention.",
        "action_steps": [
            "Widen aperture (lower f-stop) or increase camera-to-subject distance ratio for selective focus falloff.",
            "Choose clean backgrounds with high luminance or chrominance contrast relative to the subject.",
            "Position the primary subject in the cleanest negative space available in the scene.",
        ],
    },
    "technical": {
        "title": "Acutance, Focus & Noise Hygiene",
        "description": "Eliminate micro-motion blur and sensory noise artifacts through calibrated exposure mechanics.",
        "action_steps": [
            "Enforce reciprocal shutter rule (minimum 1/(2x focal length)) when shooting handheld.",
            "Keep ISO within base-to-medium range; bracket exposures rather than pushing underexposed files in post.",
            "Verify single-point autofocus lock on the nearest high-contrast edge or subject eye.",
        ],
    },
}


class MasteryLevel(str, Enum):
    """Categorical skill mastery tiers derived from empirical mean aspect scores."""
    NOVICE = "Novice"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    MASTER = "Master"


class TrajectoryStatus(str, Enum):
    """Directional progress trend for an aesthetic aspect over recent uploads."""
    IMPROVING = "improving"
    STAGNANT = "stagnant"
    REGRESSING = "regressing"
    MASTERED = "mastered"


@dataclass
class UserPhotoRecord:
    """Historical snapshot of an uploaded photo and its AesFix evaluation telemetry.

    Why this container is preserved:
    Stores both initial aspect scores and counterfactual fix telemetry so future coaching
    can use the photographer's own images as positive training exemplars.
    """
    photo_id: str
    timestamp: datetime.datetime
    image_path: str
    aspect_scores: Dict[str, float]
    overall_score: float
    weakest_aspect: str
    best_fix_name: Optional[str] = None
    best_fix_path: Optional[str] = None
    verified_gain: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UserPhotoRecord:
        ts = data["timestamp"]
        if isinstance(ts, str):
            ts = datetime.datetime.fromisoformat(ts)
        return cls(
            photo_id=data["photo_id"],
            timestamp=ts,
            image_path=data["image_path"],
            aspect_scores=dict(data.get("aspect_scores", {})),
            overall_score=float(data.get("overall_score", 0.5)),
            weakest_aspect=str(data.get("weakest_aspect", "lighting")),
            best_fix_name=data.get("best_fix_name"),
            best_fix_path=data.get("best_fix_path"),
            verified_gain=data.get("verified_gain"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class AspectTrajectory:
    """Longitudinal progression metrics for a single aesthetic aspect dimension.

    Why growth rate is bounded and normalized:
    A photographer scoring 0.50 moving to 0.60 represents +20% skill improvement,
    whereas an advanced photographer scoring 0.85 moving to 0.90 is +5.8% but near mastery.
    """
    aspect: str
    baseline_score: float
    recent_score: float
    growth_rate_pct: float
    status: TrajectoryStatus
    sample_count: int
    history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class CounterfactualExemplar:
    """A user's own past photo paired with its verified fix as an actionable coaching lesson.

    Why own-photo counterfactuals beat textbook tutorials:
    Cognitive psychology shows self-referential feedback produces dramatically higher
    retention and habit correction compared to observing stranger work.
    """
    photo_id: str
    aspect: str
    original_score: float
    fixed_score: float
    verified_gain: float
    fix_applied: str
    original_path: str
    fixed_path: str
    coaching_note: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CurriculumDrill:
    """Weekly actionable coaching drill targeting the user's primary bottleneck aspect."""
    drill_id: str
    target_aspect: str
    title: str
    description: str
    difficulty: str
    action_steps: List[str]
    counterfactual_exemplars: List[CounterfactualExemplar] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SkillGraph:
    """Comprehensive multi-dimensional skill profile constructed from upload history."""
    user_id: str
    mastery_level: MasteryLevel
    overall_skill_index: float
    bottleneck_aspect: str
    top_strength: str
    trajectories: Dict[str, AspectTrajectory]
    total_photos_analyzed: int
    active_streak_weeks: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "mastery_level": self.mastery_level.value,
            "overall_skill_index": round(self.overall_skill_index, 4),
            "bottleneck_aspect": self.bottleneck_aspect,
            "top_strength": self.top_strength,
            "total_photos_analyzed": self.total_photos_analyzed,
            "active_streak_weeks": self.active_streak_weeks,
            "trajectories": {k: v.to_dict() for k, v in self.trajectories.items()},
        }


@dataclass
class WeeklyProgressDigest:
    """Formatted retention & progress report for weekly notification/email dispatch (F2.5)."""
    user_id: str
    digest_date: datetime.date
    skill_graph: SkillGraph
    active_drills: List[CurriculumDrill]
    headline: str
    growth_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "digest_date": self.digest_date.isoformat(),
            "headline": self.headline,
            "growth_summary": self.growth_summary,
            "skill_graph": self.skill_graph.to_dict(),
            "active_drills": [d.to_dict() for d in self.active_drills],
        }

    def to_markdown(self) -> str:
        """Renders an elegant Markdown report suitable for email newsletters or web display.

        Why Markdown structure is prioritized:
        Allows direct dispatch into SendGrid/SES email templates, terminal renderers,
        or GitHub issue digests without external rendering dependencies.
        """
        sg = self.skill_graph
        bottleneck_title = ASPECT_TITLES.get(sg.bottleneck_aspect, sg.bottleneck_aspect.title())
        strength_title = ASPECT_TITLES.get(sg.top_strength, sg.top_strength.title())

        lines = [
            f"# 📸 AesFix Weekly Studio Digest: {self.headline}",
            f"*Progress report for user `{self.user_id}` • Date: {self.digest_date.isoformat()}*",
            "",
            "---",
            "",
            "## 🏆 Skill Overview",
            f"- **Mastery Tier:** `{sg.mastery_level.value}` (Skill Index: `{sg.overall_skill_index * 100:.1f} / 100`)",
            f"- **Photos Evaluated:** `{sg.total_photos_analyzed}` images",
            f"- **Top Pillar:** 🌟 **{strength_title}**",
            f"- **Focus Bottleneck:** 🎯 **{bottleneck_title}**",
            "",
            f"> {self.growth_summary}",
            "",
            "## 📈 Aspect Trajectories (Longitudinal Growth)",
            "",
            "| Aspect Dimension | Baseline | Recent | Growth | Status |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]

        for aspect_name in ASPECT_DIMENSIONS:
            traj = sg.trajectories.get(aspect_name)
            if not traj:
                continue
            title = ASPECT_TITLES.get(aspect_name, aspect_name.title())
            symbol = "+" if traj.growth_rate_pct > 0 else ""
            status_badge = {
                TrajectoryStatus.MASTERED: "⭐ Mastered",
                TrajectoryStatus.IMPROVING: "📈 Improving",
                TrajectoryStatus.STAGNANT: "⚖️ Stagnant",
                TrajectoryStatus.REGRESSING: "📉 Regressing",
            }.get(traj.status, traj.status.value)

            lines.append(
                f"| **{title}** | `{traj.baseline_score:.2f}` | `{traj.recent_score:.2f}` | `{symbol}{traj.growth_rate_pct:.1f}%` | {status_badge} |"
            )

        lines.extend(["", "---", "", "## 🎯 Active Personal Coaching Drills", ""])

        if not self.active_drills:
            lines.append("*All core dimensions are currently balanced! Keep shooting to uncover deeper patterns.*")
        else:
            for i, drill in enumerate(self.active_drills, start=1):
                target_title = ASPECT_TITLES.get(drill.target_aspect, drill.target_aspect.title())
                lines.append(f"### Drill {i}: {drill.title} ({target_title})")
                lines.append(f"*{drill.description}* (Difficulty: `{drill.difficulty.title()}`)")
                lines.append("")
                lines.append("**Prescribed Field Actions:**")
                for step in drill.action_steps:
                    lines.append(f"1. {step}")
                lines.append("")

                if drill.counterfactual_exemplars:
                    lines.append("**Your Counterfactual Lessons (How AesFix Fixed Your Own Past Photos):**")
                    for ex in drill.counterfactual_exemplars:
                        lines.append(
                            f"- **Photo ID `{ex.photo_id}`**: Original `{ex.aspect}` was `{ex.original_score:.2f}`. "
                            f"Applying `{ex.fix_applied}` elevated score to `{ex.fixed_score:.2f}` (+`{ex.verified_gain:+.2f}` gain)."
                        )
                        lines.append(f"  *Coaching takeaway: {ex.coaching_note}*")
                    lines.append("")

        lines.append("---")
        lines.append("*AesFix Closed-Loop Aesthetics Engine • 'The tool that makes users not need it is the tool they evangelize.'*")
        return "\n".join(lines)


class SkillCurriculumEngine:
    """Longitudinal skill trajectory engine that transforms photo history into actionable coaching.

    Why this class is the core coordinator:
    Encapsulates chronological parsing of user telemetry, empirical calculation of
    trajectory baselines, bottleneck detection, counterfactual lesson pairing, and
    weekly progress digest synthesis.
    """

    def __init__(self, user_history: Optional[List[UserPhotoRecord]] = None) -> None:
        self.history: List[UserPhotoRecord] = []
        if user_history:
            for rec in user_history:
                self.add_record(rec)

    def add_record(self, record: UserPhotoRecord) -> None:
        """Appends a new photo evaluation record and preserves temporal ordering."""
        self.history.append(record)
        # Ensure chronological ordering by timestamp
        self.history.sort(key=lambda r: r.timestamp)

    def compute_aspect_trajectories(
        self,
        baseline_window: int = 3,
        recent_window: int = 3,
    ) -> Dict[str, AspectTrajectory]:
        """Calculates trajectory metrics across the 5 canonical photographic dimensions.

        Why baseline window vs recent window comparison:
        Comparing the earliest N uploads against the latest N uploads provides robust
        longitudinal progress quantification resilient to single-photo outliers.
        """
        trajectories: Dict[str, AspectTrajectory] = {}
        n = len(self.history)

        for aspect in ASPECT_DIMENSIONS:
            if n == 0:
                # Default baseline for empty history
                trajectories[aspect] = AspectTrajectory(
                    aspect=aspect,
                    baseline_score=0.5,
                    recent_score=0.5,
                    growth_rate_pct=0.0,
                    status=TrajectoryStatus.STAGNANT,
                    sample_count=0,
                    history=[],
                )
                continue

            scores = [r.aspect_scores.get(aspect, 0.5) for r in self.history]

            if n <= baseline_window:
                base_score = float(sum(scores) / n)
                rec_score = base_score
                growth_pct = 0.0
            else:
                base_slice = scores[:baseline_window]
                rec_slice = scores[-recent_window:]
                base_score = float(sum(base_slice) / len(base_slice))
                rec_score = float(sum(rec_slice) / len(rec_slice))
                growth_pct = ((rec_score - base_score) / max(0.01, base_score)) * 100.0

            # Determine progression status
            if rec_score >= 0.85:
                status = TrajectoryStatus.MASTERED
            elif growth_pct >= 5.0:
                status = TrajectoryStatus.IMPROVING
            elif growth_pct <= -5.0:
                status = TrajectoryStatus.REGRESSING
            else:
                status = TrajectoryStatus.STAGNANT

            trajectories[aspect] = AspectTrajectory(
                aspect=aspect,
                baseline_score=round(base_score, 4),
                recent_score=round(rec_score, 4),
                growth_rate_pct=round(growth_pct, 2),
                status=status,
                sample_count=n,
                history=scores,
            )

        return trajectories

    def build_skill_graph(self, user_id: str = "photographer_01") -> SkillGraph:
        """Constructs an aggregated skill graph identifying top strengths and bottlenecks."""
        trajectories = self.compute_aspect_trajectories()
        n = len(self.history)

        if n == 0:
            avg_skill = 0.5
            bottleneck = "lighting"
            top_strength = "composition"
        else:
            recent_scores = {k: v.recent_score for k, v in trajectories.items()}
            avg_skill = float(sum(recent_scores.values()) / len(recent_scores))

            # Bottleneck is prioritized by lowest recent score; break ties with lowest growth
            bottleneck = min(
                ASPECT_DIMENSIONS,
                key=lambda a: (trajectories[a].recent_score, trajectories[a].growth_rate_pct),
            )
            # Top strength is highest recent score; break ties with highest growth
            top_strength = max(
                ASPECT_DIMENSIONS,
                key=lambda a: (trajectories[a].recent_score, trajectories[a].growth_rate_pct),
            )

        # Categorize into discrete mastery levels
        if avg_skill >= 0.85:
            level = MasteryLevel.MASTER
        elif avg_skill >= 0.72:
            level = MasteryLevel.ADVANCED
        elif avg_skill >= 0.58:
            level = MasteryLevel.INTERMEDIATE
        else:
            level = MasteryLevel.NOVICE

        return SkillGraph(
            user_id=user_id,
            mastery_level=level,
            overall_skill_index=round(avg_skill, 4),
            bottleneck_aspect=bottleneck,
            top_strength=top_strength,
            trajectories=trajectories,
            total_photos_analyzed=n,
            active_streak_weeks=max(1, (n // 5) + 1),
        )

    def find_counterfactual_exemplars(
        self,
        aspect: str,
        max_exemplars: int = 2,
    ) -> List[CounterfactualExemplar]:
        """Retrieves user's historical photos where the given aspect was improved by a fix.

        Why historical counterfactuals are mined:
        Rather than showing arbitrary good photos, we extract images where the user
        specifically struggled on this aspect, but an automated fix resolved it.
        """
        exemplars: List[CounterfactualExemplar] = []

        # Find historical uploads where this aspect was the primary weakness or had low score
        eligible_records = [
            r for r in self.history
            if r.best_fix_name is not None
            and r.verified_gain is not None
            and r.verified_gain > 0.0
            and (r.weakest_aspect == aspect or r.aspect_scores.get(aspect, 1.0) < 0.65)
        ]

        # Prioritize exemplars with highest verified gain
        eligible_records.sort(key=lambda r: float(r.verified_gain or 0.0), reverse=True)

        for rec in eligible_records[:max_exemplars]:
            orig_score = rec.aspect_scores.get(aspect, 0.5)
            # Estimated fixed aspect score after applying the verified fix
            fixed_score = min(1.0, orig_score + float(rec.verified_gain or 0.15))
            coaching_note = (
                f"In your photo '{rec.photo_id}', {aspect.replace('_', ' ')} scored {orig_score:.2f}. "
                f"Executing {rec.best_fix_name} unlocked a +{float(rec.verified_gain):.2f} gain by resolving visual tension."
            )

            exemplars.append(
                CounterfactualExemplar(
                    photo_id=rec.photo_id,
                    aspect=aspect,
                    original_score=round(orig_score, 3),
                    fixed_score=round(fixed_score, 3),
                    verified_gain=round(float(rec.verified_gain or 0.0), 3),
                    fix_applied=rec.best_fix_name or "Auto-Exposure & Tone Curve",
                    original_path=rec.image_path,
                    fixed_path=rec.best_fix_path or rec.image_path,
                    coaching_note=coaching_note,
                )
            )

        return exemplars

    def generate_curriculum_drills(
        self,
        skill_graph: SkillGraph,
        max_drills: int = 2,
    ) -> List[CurriculumDrill]:
        """Synthesizes targeted drills prioritizing the primary bottleneck dimension."""
        drills: List[CurriculumDrill] = []

        # Primary drill: direct attack on the bottleneck
        primary_aspect = skill_graph.bottleneck_aspect
        template = ASPECT_DRILL_TEMPLATES.get(primary_aspect, ASPECT_DRILL_TEMPLATES["lighting"])
        primary_exemplars = self.find_counterfactual_exemplars(primary_aspect, max_exemplars=2)

        difficulty_map = {
            MasteryLevel.NOVICE: "foundational",
            MasteryLevel.INTERMEDIATE: "intermediate",
            MasteryLevel.ADVANCED: "advanced",
            MasteryLevel.MASTER: "mastery",
        }
        diff = difficulty_map.get(skill_graph.mastery_level, "intermediate")

        drills.append(
            CurriculumDrill(
                drill_id=f"drill_{primary_aspect}_primary",
                target_aspect=primary_aspect,
                title=template["title"],
                description=template["description"],
                difficulty=diff,
                action_steps=list(template["action_steps"]),
                counterfactual_exemplars=primary_exemplars,
            )
        )

        if max_drills > 1:
            # Secondary drill: find the next weakest dimension that is not yet mastered
            candidate_aspects = [
                a for a in ASPECT_DIMENSIONS
                if a != primary_aspect and skill_graph.trajectories[a].status != TrajectoryStatus.MASTERED
            ]
            if candidate_aspects:
                secondary_aspect = min(
                    candidate_aspects,
                    key=lambda a: skill_graph.trajectories[a].recent_score,
                )
                sec_template = ASPECT_DRILL_TEMPLATES.get(secondary_aspect, ASPECT_DRILL_TEMPLATES["composition"])
                sec_exemplars = self.find_counterfactual_exemplars(secondary_aspect, max_exemplars=1)

                drills.append(
                    CurriculumDrill(
                        drill_id=f"drill_{secondary_aspect}_secondary",
                        target_aspect=secondary_aspect,
                        title=sec_template["title"],
                        description=sec_template["description"],
                        difficulty=diff,
                        action_steps=list(sec_template["action_steps"]),
                        counterfactual_exemplars=sec_exemplars,
                    )
                )

        return drills

    def generate_weekly_digest(
        self,
        user_id: str = "photographer_01",
        digest_date: Optional[datetime.date] = None,
    ) -> WeeklyProgressDigest:
        """Synthesizes the complete weekly progress and retention digest."""
        date = digest_date or datetime.date.today()
        graph = self.build_skill_graph(user_id=user_id)
        drills = self.generate_curriculum_drills(graph, max_drills=2)

        bottleneck_title = ASPECT_TITLES.get(graph.bottleneck_aspect, graph.bottleneck_aspect.title())
        growth_val = graph.trajectories[graph.bottleneck_aspect].growth_rate_pct

        if growth_val > 5.0:
            headline = f"Strong Momentum: +{growth_val:.1f}% Growth on {bottleneck_title}!"
            summary = (
                f"You are actively leveling up in {bottleneck_title}. Your recent uploads demonstrate "
                f"sharper discipline, with consistent gains across your evaluation history."
            )
        elif growth_val < -5.0:
            headline = f"Focus Challenge: Recalibrating Your {bottleneck_title}"
            summary = (
                f"Your recent shoots encountered tricky conditions for {bottleneck_title} ({growth_val:.1f}% dip). "
                f"Check this week's counterfactual drills to reset your foundation."
            )
        else:
            headline = f"Consistency Check: Leveling Up {bottleneck_title}"
            summary = (
                f"Your photographic fundamentals remain rock-solid. To break through the {graph.mastery_level.value} "
                f"ceiling, target your {bottleneck_title} exercises below."
            )

        return WeeklyProgressDigest(
            user_id=user_id,
            digest_date=date,
            skill_graph=graph,
            active_drills=drills,
            headline=headline,
            growth_summary=summary,
        )

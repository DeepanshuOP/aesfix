"""Beta User Cohort Management, Activation Funnel, and Feedback Engine (AF-5.6 & Roadmap §8, §30, §42).

Why structured beta cohort instrumentation is essential for GTM success:
1. Persona-Driven Onboarding: AesFix serves distinct customer segments (Marketplace Sellers,
   Enthusiasts, Dating Optimizers). Tracking cohorts by persona reveals whether retention is
   driven by Lightroom XMP exports (Enthusiasts) or conversion-ready listing fixes (Sellers).
2. The Core Activation Metric: The critical aha moment is 'first_verified_fix_viewed' (where
   the user sees their photo improved alongside verified gain numbers). Tracking this funnel
   step isolates onboarding friction before spending capital on public launch campaigns.
3. Feedback & Net Promoter Score (NPS): Collects quantifiable satisfaction scores across
   critique accuracy, visual fix appeal, and willingness to pay to guide final launch polish.
"""
from __future__ import annotations

import enum
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("aesfix.beta")


class BetaPersona(str, enum.Enum):
    """Target launch personas per Roadmap §8."""
    MARKETPLACE_SELLER = "marketplace_seller"  # P1: Etsy/eBay/Airbnb/Realtor
    PHOTO_ENTHUSIAST = "photo_enthusiast"      # P2: Lightroom hobbyists
    DATING_OPTIMIZER = "dating_optimizer"      # P4: Dating profile users


class FunnelStep(str, enum.Enum):
    """Sequential milestones in the user activation journey."""
    ONBOARDED = "onboarded"
    PHOTO_UPLOADED = "photo_uploaded"
    CRITIQUE_VIEWED = "critique_viewed"
    FIRST_VERIFIED_FIX_VIEWED = "first_verified_fix_viewed"  # The North Star Activation Moment
    XMP_DOWNLOADED = "xmp_downloaded"
    COMPARISON_SHARED = "comparison_shared"


@dataclass
class BetaUser:
    """Participant in the 20-user closed beta cohort."""
    user_id: str
    email: str
    persona: BetaPersona
    invite_code: str
    is_activated: bool = False
    invited_at: float = field(default_factory=time.time)
    activated_at: Optional[float] = None
    completed_funnel_steps: List[str] = field(default_factory=list)


@dataclass
class FeedbackSubmission:
    """User qualitative evaluation and NPS rating."""
    feedback_id: str
    user_id: str
    score_1_to_10: int  # Standard NPS scale
    category: str       # critique_accuracy, fix_quality, usability, pricing
    comment: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class FunnelAnalyticsSummary:
    """Aggregated cohort conversion and activation metrics."""
    total_invited: int
    total_activated: int
    activation_rate_pct: float
    funnel_conversion: Dict[str, int]
    aha_moment_conversion_pct: float  # % of activated users reaching first_verified_fix_viewed
    power_user_conversion_pct: float  # % of activated users reaching xmp_downloaded
    nps_score: float
    nps_breakdown: Dict[str, int]     # promoters, passives, detractors


class BetaCohortManager:
    """Coordinates beta invitations, funnel milestone tracking, and user feedback."""

    def __init__(self) -> None:
        self._users: Dict[str, BetaUser] = {}
        self._invite_codes: Dict[str, str] = {}  # invite_code -> user_id
        self._feedback: List[FeedbackSubmission] = []
        self._seed_initial_cohort()

    def _seed_initial_cohort(self) -> None:
        """Initializes the canonical 20-user beta cohort slots across target personas."""
        personas = [
            (BetaPersona.MARKETPLACE_SELLER, 8),
            (BetaPersona.PHOTO_ENTHUSIAST, 7),
            (BetaPersona.DATING_OPTIMIZER, 5),
        ]
        user_idx = 1
        for persona, count in personas:
            for _ in range(count):
                uid = f"beta_user_{user_idx:02d}"
                code = f"AESFIX-{persona.value[:4].upper()}-{user_idx:02d}"
                user = BetaUser(
                    user_id=uid,
                    email=f"{uid}@{persona.value}.test",
                    persona=persona,
                    invite_code=code,
                )
                self._users[uid] = user
                self._invite_codes[code] = uid
                user_idx += 1

    def activate_user(self, invite_code: str, email: Optional[str] = None) -> BetaUser:
        """Activates a beta participant by their unique invitation code."""
        code_clean = invite_code.strip().upper()
        if code_clean not in self._invite_codes:
            raise KeyError(f"Invalid or expired invite code: '{invite_code}'")

        uid = self._invite_codes[code_clean]
        user = self._users[uid]
        user.is_activated = True
        user.activated_at = time.time()
        if email:
            user.email = email

        if FunnelStep.ONBOARDED.value not in user.completed_funnel_steps:
            user.completed_funnel_steps.append(FunnelStep.ONBOARDED.value)

        logger.info("Activated beta user %s (%s)", user.user_id, user.persona.value)
        return user

    def record_step(self, user_id: str, step: FunnelStep) -> bool:
        """Records progression along the activation funnel."""
        if user_id not in self._users:
            return False

        user = self._users[user_id]
        if step.value not in user.completed_funnel_steps:
            user.completed_funnel_steps.append(step.value)
            logger.info("User %s completed funnel milestone: %s", user_id, step.value)
        return True

    def submit_feedback(
        self,
        user_id: str,
        score: int,
        category: str,
        comment: str,
    ) -> FeedbackSubmission:
        """Ingests user evaluation and NPS rating."""
        if score < 0 or score > 10:
            raise ValueError(f"NPS score must be between 0 and 10, got {score}")

        sub = FeedbackSubmission(
            feedback_id=f"fb_{uuid.uuid4().hex[:8]}",
            user_id=user_id,
            score_1_to_10=score,
            category=category,
            comment=comment,
        )
        self._feedback.append(sub)
        return sub

    def get_funnel_summary(self) -> FunnelAnalyticsSummary:
        """Calculates cohort conversion, activation drop-offs, and Net Promoter Score."""
        total_invited = len(self._users)
        activated_users = [u for u in self._users.values() if u.is_activated]
        total_activated = len(activated_users)

        activation_rate = (total_activated / total_invited) * 100.0 if total_invited > 0 else 0.0

        # Step count aggregation
        step_counts: Dict[str, int] = {s.value: 0 for s in FunnelStep}
        for u in activated_users:
            for s in u.completed_funnel_steps:
                if s in step_counts:
                    step_counts[s] += 1

        # Aha moment conversion (% of activated who viewed first verified fix)
        aha_count = step_counts.get(FunnelStep.FIRST_VERIFIED_FIX_VIEWED.value, 0)
        aha_pct = (aha_count / total_activated) * 100.0 if total_activated > 0 else 0.0

        # Power user conversion (% who downloaded XMP)
        xmp_count = step_counts.get(FunnelStep.XMP_DOWNLOADED.value, 0)
        power_pct = (xmp_count / total_activated) * 100.0 if total_activated > 0 else 0.0

        # Net Promoter Score computation
        promoters = sum(1 for f in self._feedback if f.score_1_to_10 >= 9)
        passives = sum(1 for f in self._feedback if 7 <= f.score_1_to_10 <= 8)
        detractors = sum(1 for f in self._feedback if f.score_1_to_10 <= 6)
        total_nps_responses = len(self._feedback)

        if total_nps_responses > 0:
            nps = round(((promoters - detractors) / total_nps_responses) * 100.0, 1)
        else:
            nps = 0.0

        return FunnelAnalyticsSummary(
            total_invited=total_invited,
            total_activated=total_activated,
            activation_rate_pct=round(activation_rate, 1),
            funnel_conversion=step_counts,
            aha_moment_conversion_pct=round(aha_pct, 1),
            power_user_conversion_pct=round(power_pct, 1),
            nps_score=nps,
            nps_breakdown={"promoters": promoters, "passives": passives, "detractors": detractors},
        )

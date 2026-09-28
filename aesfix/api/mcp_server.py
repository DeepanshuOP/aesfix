"""Model Context Protocol (MCP) Server for AesFix (AF-6.1 & Roadmap §25 F4.6).

Why an MCP Server is a high-leverage moonshot for AesFix:
1. Agent-Native Aesthetic Primitive: Exposes critique, fix, and verification as
   first-class tools that external LLM agents (Claude Desktop, Cursor, Antigravity,
   and autonomous generation pipelines) can call over standard JSON-RPC (stdio/SSE).
2. Closes the Agent Loop: AI agents generating images or building landing pages can
   autonomously lint and retouch visual assets, verifying aesthetic gain before
   deploying assets to production.
3. Interview-Defensible Architecture: Built on the official Anthropic Model Context
   Protocol SDK (MCP 2.x), supporting both stdio transport (local IDE plugins) and
   SSE ASGI mounting (direct integration into the production FastAPI gateway).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import tempfile
import time
from typing import Any, Dict, Optional

from PIL import Image
from mcp.server.mcpserver import MCPServer

from aesfix.api.demo import compute_grade_label, get_demo_components
from aesfix.critic.critic import CriticFix
from aesfix.data.loaders import ASPECT_DIMENSIONS
from aesfix.executor.guard import IdentityGuard
from aesfix.executor.tier_a import TIER_A_OPERATIONS, TierAExecutor

logger = logging.getLogger("aesfix.mcp")


def create_mcp_server() -> MCPServer:
    """Factory creating and configuring the AesFix MCP server with tools, resources, and prompts.

    Why a factory function:
    Enables isolated instantiation in unit tests, prevents shared state across test runs,
    and cleanly separates server definition from transport execution.
    """
    server = MCPServer("aesfix-mcp")

    # =========================================================================
    # 1. MCP Tools: Exposing Core Photographic Capabilities
    # =========================================================================

    @server.tool()
    def critique_image(
        image_path: str,
        goal_or_genre: Optional[str] = None,
        user_critique: Optional[str] = None,
    ) -> str:
        """Critique an image across 5 aesthetic dimensions (composition, lighting, color, subject clarity, technical).

        Returns 5-aspect scores, neuro-fuzzy ANFIS rulebook trace, qualitative
        rationales, and concrete machine-executable fix proposals.
        """
        # File existence validation prevents opaque downstream failures in vision backends
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Target image not found at path: {image_path!r}")

        critic, head, _ = get_demo_components()

        # Aspect Critic analyzes the 5 dimensions and proposes falsifiable fixes
        critique_res = critic.critique(
            image_path=image_path,
            user_critique=user_critique,
            genre=goal_or_genre,
        )

        # Takagi-Sugeno ANFIS neuro-fuzzy head computes interpretable overall score
        anfis_trace = head.explain(critique_res.aspect_scores, top_k=3)
        overall_score = round(anfis_trace.overall_score, 4)
        grade = compute_grade_label(overall_score)

        payload = {
            "image_path": image_path,
            "overall_score": overall_score,
            "grade_label": grade,
            "aspect_scores": critique_res.aspect_scores,
            "anfis_rulebook_summary": anfis_trace.rulebook_summary,
            "top_fired_rules": [
                {
                    "rule_index": r.rule_index,
                    "firing_strength": round(r.firing_strength, 4),
                    "antecedent_description": r.antecedent_description,
                    "contribution": round(r.contribution, 4),
                }
                for r in anfis_trace.top_fired_rules
            ],
            "rationales": critique_res.rationales,
            "overall_summary": critique_res.overall_summary,
            "proposed_fixes": [
                {
                    "op": fix.op,
                    "params": fix.params,
                    "expected_aspect": fix.expected_aspect,
                    "instruction": fix.nl_instruction,
                    "region_hint": fix.region_hint,
                }
                for fix in critique_res.fixes
            ],
        }
        return json.dumps(payload, indent=2)

    @server.tool()
    def fix_image(
        image_path: str,
        op: str,
        params: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """Execute a deterministic photographic adjustment and verify against Identity Guard drift.

        Valid operations: adjust_exposure, adjust_contrast, adjust_white_balance,
        adjust_saturation, unsharp_mask, smart_crop, correct_horizon, denoise, etc.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Source image not found: {image_path!r}")

        clean_op = op.strip().lower()
        if clean_op not in TIER_A_OPERATIONS:
            raise ValueError(
                f"Unknown fix operation {clean_op!r}. Supported operations: {sorted(list(TIER_A_OPERATIONS))}"
            )

        executor = TierAExecutor()
        guard = IdentityGuard(drift_threshold=0.35)
        fix_params = params or {}

        # 1. Apply deterministic Tier-A photographic adjustment
        fix = CriticFix(
            op=clean_op,
            params=fix_params,
            expected_aspect="lighting" if "exposure" in clean_op or "contrast" in clean_op else "composition",
            nl_instruction=f"MCP-initiated adjustment: {clean_op}",
        )
        exec_res = executor.execute_fixes(image_input=image_path, fixes=[fix])

        # 2. Determine target output path
        if not output_path:
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"aesfix_mcp_{clean_op}_{int(time.time()*1000)}.jpg")

        exec_res.image.save(output_path, format="JPEG", quality=95)

        # 3. Check Identity Guard drift to prevent structural corruption or reward hacking
        guard_verdict = guard.evaluate(original=image_path, edited=output_path)

        payload = {
            "source_image": image_path,
            "output_path": output_path,
            "operation": clean_op,
            "params": fix_params,
            "duration_ms": round(exec_res.duration_ms, 2),
            "guard_passed": guard_verdict.is_valid,
            "composite_drift": round(guard_verdict.composite_drift, 4),
            "guard_explanation": guard_verdict.rejection_reason or ("Passed Identity Guard" if guard_verdict.is_valid else "Vetoed"),
            "xmp_sidecar_preview": exec_res.xmp_sidecar[:400] + "...",
        }
        return json.dumps(payload, indent=2)

    @server.tool()
    def verify_gain(
        original_image_path: str,
        edited_image_path: str,
        target_aspect: Optional[str] = None,
    ) -> str:
        """Measure verified Aesthetic Gain (AG) between original and edited photo using frozen judges.

        Calculates baseline and post-edit aesthetic ratings, verifies identity preservation,
        and returns the verified aesthetic gain badge.
        """
        if not os.path.exists(original_image_path):
            raise FileNotFoundError(f"Original image not found: {original_image_path!r}")
        if not os.path.exists(edited_image_path):
            raise FileNotFoundError(f"Edited image not found: {edited_image_path!r}")

        _, _, ensemble = get_demo_components()
        guard = IdentityGuard(drift_threshold=0.35)

        # Perceptual identity check blocks reward hacking before crediting gains
        guard_verdict = guard.evaluate(original=original_image_path, edited=edited_image_path)

        base_res = ensemble.evaluate(original_image_path)
        post_res = ensemble.evaluate(edited_image_path)

        base_score = round(base_res.overall_score, 4)
        post_score = round(post_res.overall_score, 4)
        raw_gain = round(post_score - base_score, 4)

        if guard_verdict.is_valid:
            verified_gain = round(max(0.0, raw_gain), 4)
            gain_badge = f"+{verified_gain:.3f} Verified Aesthetic Gain"
        else:
            verified_gain = 0.0
            gain_badge = "Vetoed by Identity Guard (Structural Drift)"

        # Measure specific aspect delta if requested
        aspect_delta = None
        if target_aspect and base_res.aspects and post_res.aspects:
            clean_aspect = target_aspect.strip().lower()
            if clean_aspect in base_res.aspects and clean_aspect in post_res.aspects:
                aspect_delta = round(post_res.aspects[clean_aspect] - base_res.aspects[clean_aspect], 4)

        payload = {
            "original_image": original_image_path,
            "edited_image": edited_image_path,
            "baseline_score": base_score,
            "post_edit_score": post_score,
            "raw_gain": raw_gain,
            "verified_gain": verified_gain,
            "gain_badge": gain_badge,
            "target_aspect": target_aspect,
            "target_aspect_delta": aspect_delta,
            "guard_passed": guard_verdict.is_valid,
            "guard_composite_drift": round(guard_verdict.composite_drift, 4),
            "disagreement_variance": round(base_res.disagreement, 4),
            "is_uncertain": base_res.is_uncertain,
        }
        return json.dumps(payload, indent=2)

    @server.tool()
    def generate_xmp(
        op: str,
        params: Optional[Dict[str, Any]] = None,
        output_xmp_path: Optional[str] = None,
    ) -> str:
        """Generate Adobe Lightroom-compatible .xmp sidecar metadata for a photographic adjustment."""
        clean_op = op.strip().lower()
        if clean_op not in TIER_A_OPERATIONS:
            raise ValueError(
                f"Unknown fix operation {clean_op!r}. Supported operations: {sorted(list(TIER_A_OPERATIONS))}"
            )

        executor = TierAExecutor()
        fix = CriticFix(
            op=clean_op,
            params=params or {},
            expected_aspect="lighting",
            nl_instruction=f"Generate XMP for {clean_op}",
        )

        # Create 1x1 blank image to cleanly harvest standard XMP sidecar XML
        blank = Image.new("RGB", (64, 64), color=(128, 128, 128))
        res = executor.execute_fixes(image_input=blank, fixes=[fix])

        if output_xmp_path:
            with open(output_xmp_path, "w", encoding="utf-8") as f:
                f.write(res.xmp_sidecar)

        payload = {
            "operation": clean_op,
            "params": params or {},
            "saved_to": output_xmp_path,
            "xmp_xml": res.xmp_sidecar,
        }
        return json.dumps(payload, indent=2)

    @server.tool()
    def run_closed_loop_pipeline(
        image_path: str,
        goal: Optional[str] = None,
        user_critique: Optional[str] = None,
    ) -> str:
        """Run the full end-to-end AesFix closed loop: Critique -> ANFIS -> Tier-A -> Guard -> Gain Verification.

        Returns comprehensive diagnosis, fired rules, and top ranked fixes by verified gain.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path!r}")

        from aesfix.api.billing import QuotaManager
        from aesfix.api.pipeline_queue import PipelineQueueManager

        # Ephemeral quota manager grants execution for agent-driven automated linter workflows
        qm = QuotaManager()
        pq = PipelineQueueManager(quota_mgr=qm)

        job = pq.submit_job(
            user_id="mcp_agent_client",
            image_input=image_path,
            user_critique=user_critique,
            goal=goal,
        )

        result = pq.execute_job_sync(job.job_id)

        payload = {
            "job_id": result.job_id,
            "overall_score": result.overall_score,
            "grade_label": result.grade_label,
            "aspect_scores": result.aspect_scores,
            "anfis_rulebook_summary": result.anfis_rulebook_summary,
            "top_fired_rules": [
                {
                    "rule_index": getattr(r, "rule_index", i),
                    "firing_strength": round(getattr(r, "firing_strength", 0.0), 4),
                    "antecedent_description": getattr(r, "antecedent_description", str(r)),
                    "contribution": round(getattr(r, "contribution", 0.0), 4),
                }
                if hasattr(r, "rule_index")
                else str(r)
                for i, r in enumerate(result.top_fired_rules)
            ],
            "critique_rationale": result.critique_rationale,
            "uncertainty_badge": result.uncertainty_badge,
            "total_latency_seconds": result.total_latency_seconds,
            "verified_fixes": [
                {
                    "fix_id": fix.fix_id,
                    "op": fix.op,
                    "params": fix.params,
                    "expected_aspect": fix.expected_aspect,
                    "reasoning": fix.reasoning,
                    "baseline_score": fix.baseline_score,
                    "post_edit_score": fix.post_edit_score,
                    "aesthetic_gain": fix.aesthetic_gain,
                    "guard_passed": fix.guard_passed,
                    "gain_badge": fix.gain_badge,
                    "execution_time_ms": fix.execution_time_ms,
                }
                for fix in result.verified_fixes
            ],
            "best_fix": (
                {
                    "op": result.best_fix.op,
                    "aesthetic_gain": result.best_fix.aesthetic_gain,
                    "gain_badge": result.best_fix.gain_badge,
                    "params": result.best_fix.params,
                }
                if result.best_fix
                else None
            ),
        }
        return json.dumps(payload, indent=2)

    # =========================================================================
    # 2. MCP Resources: Exposing Standards and Operational Taxonomies
    # =========================================================================

    @server.resource("aesfix://aspects")
    def get_aspect_taxonomy() -> str:
        """Returns the canonical 5-dimensional aesthetic taxonomy and evaluation rubrics."""
        taxonomy = {
            "dimensions": ASPECT_DIMENSIONS,
            "rubrics": {
                "composition": "Framing, balance, subject placement, leading lines, rule of thirds, horizon alignment.",
                "lighting": "Exposure calibration, highlight preservation, shadow detail, dynamic range, tonal contrast.",
                "color": "Color harmony, white balance fidelity, saturation vibrance, palette coherence.",
                "subject_clarity": "Focal sharpness, subject-background separation, depth-of-field isolation.",
                "technical": "Sensor noise, compression artifacts, chromatic aberration, optical clarity.",
            },
        }
        return json.dumps(taxonomy, indent=2)

    @server.resource("aesfix://vocabulary")
    def get_executor_vocabulary() -> str:
        """Returns the supported machine-executable photographic operations and parameter schemas."""
        vocab = {
            "operations": sorted(list(TIER_A_OPERATIONS)),
            "parameter_constraints": {
                "adjust_exposure": {"ev_delta": "float in [-2.0, 2.0]"},
                "adjust_contrast": {"factor": "float in [0.5, 2.0]"},
                "adjust_saturation": {"factor": "float in [0.0, 2.0]"},
                "adjust_white_balance": {"kelvin_delta": "int in [-2000, 2000]"},
                "unsharp_mask": {"radius": "float in [0.5, 3.0]", "amount": "float in [0.5, 2.5]"},
                "smart_crop": {"rule": "rule_of_thirds|center", "tightness": "float in [0.6, 1.0]"},
                "correct_horizon": {"angle_degrees": "float in [-15.0, 15.0]"},
                "denoise": {"strength": "float in [1.0, 15.0]"},
            },
        }
        return json.dumps(vocab, indent=2)

    @server.resource("aesfix://goals")
    def get_goal_profiles() -> str:
        """Returns supported domain goal profiles and aspect priority weightings."""
        from aesfix.api.batch_goal import GOAL_PROFILES
        profiles = {
            (goal.value if hasattr(goal, "value") else str(goal)): {
                "name": getattr(profile, "display_name", str(profile.mode)),
                "description": profile.description,
                "weights": profile.aspect_weights,
                "min_acceptable_score": profile.min_quality_threshold,
            }
            for goal, profile in GOAL_PROFILES.items()
        }
        return json.dumps(profiles, indent=2)

    # =========================================================================
    # 3. MCP Prompts: Structured Guidance for Autonomous Agents
    # =========================================================================

    @server.prompt("diagnose_photo")
    def prompt_diagnose_photo(image_path: str) -> str:
        """Agent guidance prompt for diagnosing and improving a photo."""
        return (
            f"Please inspect the photograph at {image_path} using AesFix tools.\n"
            f"1. Call critique_image(image_path='{image_path}') to get the 5-aspect scores and proposed fixes.\n"
            f"2. For the weakest aspect, call fix_image() to apply the recommended adjustment.\n"
            f"3. Call verify_gain() to confirm that the adjustment produced genuine aesthetic gain without structural drift."
        )

    @server.prompt("prepare_listing_photo")
    def prompt_prepare_listing_photo(image_path: str) -> str:
        """Agent guidance prompt for commercial listing optimization."""
        return (
            f"Optimize the e-commerce listing photo at {image_path}.\n"
            f"1. Run run_closed_loop_pipeline(image_path='{image_path}', goal='listing').\n"
            f"2. Inspect the verified fixes and verify that the best fix achieves verified aesthetic gain.\n"
            f"3. Export the Lightroom .xmp sidecar using generate_xmp() for non-destructive catalog archiving."
        )

    return server


def main() -> None:
    """CLI entry point for running the AesFix MCP server."""
    parser = argparse.ArgumentParser(description="AesFix Model Context Protocol (MCP) Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport protocol: 'stdio' for IDE agents (Claude Desktop, Cursor) or 'sse' for HTTP streaming.",
    )
    parser.add_argument("--port", type=int, default=8001, help="Port for SSE transport (default: 8001).")
    parser.add_argument("--host", default="0.0.0.0", help="Host for SSE transport (default: 0.0.0.0).")

    args = parser.parse_args()
    server = create_mcp_server()

    if args.transport == "stdio":
        # Stdio runs event loop reading from sys.stdin and writing to sys.stdout
        logger.info("Starting AesFix MCP server on stdio transport...")
        asyncio.run(server.run_stdio_async())
    elif args.transport == "sse":
        import uvicorn
        logger.info("Starting AesFix MCP server on SSE transport at %s:%d...", args.host, args.port)
        uvicorn.run(server.sse_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

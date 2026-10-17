"""Unit and integration tests for the AesFix Model Context Protocol (MCP) Server (AF-6.1 & Roadmap §25 F4.6).

Why comprehensive tests for the MCP Server are essential:
1. Agent Protocol Contract: External AI agents (Claude, Cursor, Antigravity) interact
   strictly through the MCP schema. Validates that tool inputs, outputs, resources,
   and prompts conform to the MCP 2.x standard.
2. Tool Execution Rigor: Confirms that `critique_image`, `fix_image`, `verify_gain`,
   `generate_xmp`, and `run_closed_loop_pipeline` produce valid, parseable JSON payloads
   with verifiable aesthetic gains and Lightroom XMP sidecars.
3. Resilience & Defense: Asserts that non-existent image paths and unsupported operations
   fail predictably with informative errors rather than uncaught panics.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aesfix.api.mcp_server import create_mcp_server

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TEST_JPG = str(_PROJECT_ROOT / "test.jpg")
_DARK_JPG = str(_PROJECT_ROOT / "dark.jpg")


@pytest.mark.asyncio
async def test_mcp_server_tools_registered() -> None:
    """Verifies that all 5 core photographic tools are properly exposed via MCP."""
    server = create_mcp_server()
    tools = await server.list_tools()
    tool_names = [t.name for t in tools]

    expected_tools = [
        "critique_image",
        "fix_image",
        "verify_gain",
        "generate_xmp",
        "run_closed_loop_pipeline",
    ]
    for exp in expected_tools:
        assert exp in tool_names, f"Missing expected MCP tool: {exp}"


@pytest.mark.asyncio
async def test_mcp_critique_image() -> None:
    """Verifies that critique_image returns structured scores, ANFIS rulebook trace, and fixes."""
    server = create_mcp_server()
    result = await server.call_tool(
        "critique_image",
        {"image_path": _TEST_JPG, "goal_or_genre": "portrait"},
    )
    assert not result.is_error
    assert len(result.content) >= 1

    payload = json.loads(result.content[0].text)
    assert payload["image_path"] == _TEST_JPG
    assert 0.0 <= payload["overall_score"] <= 1.0
    assert any(g in payload["grade_label"] for g in ("A", "B", "C", "D"))
    assert len(payload["aspect_scores"]) == 5
    assert "composition" in payload["aspect_scores"]
    assert "Overall Aesthetic Score" in payload["anfis_rulebook_summary"]
    assert len(payload["top_fired_rules"]) >= 1
    assert isinstance(payload["proposed_fixes"], list)


@pytest.mark.asyncio
async def test_mcp_fix_image(tmp_path: Path) -> None:
    """Verifies that fix_image applies Tier-A adjustments and checks Identity Guard."""
    server = create_mcp_server()
    out_file = str(tmp_path / "fixed_output.jpg")

    result = await server.call_tool(
        "fix_image",
        {
            "image_path": _DARK_JPG,
            "op": "adjust_exposure",
            "params": {"ev_delta": 0.4},
            "output_path": out_file,
        },
    )
    assert not result.is_error
    payload = json.loads(result.content[0].text)

    assert payload["source_image"] == _DARK_JPG
    assert payload["operation"] == "adjust_exposure"
    assert payload["guard_passed"] is True
    assert 0.0 <= payload["composite_drift"] <= 1.0
    assert Path(out_file).exists()
    assert "<x:xmpmeta" in payload["xmp_sidecar_preview"]


@pytest.mark.asyncio
async def test_mcp_verify_gain(tmp_path: Path) -> None:
    """Verifies that verify_gain calculates aesthetic gain and Identity Guard drift."""
    server = create_mcp_server()
    edited_file = str(tmp_path / "edited_dark.jpg")

    # 1. Apply authentic mild exposure adjustment
    await server.call_tool(
        "fix_image",
        {
            "image_path": _DARK_JPG,
            "op": "adjust_exposure",
            "params": {"ev_delta": 0.3},
            "output_path": edited_file,
        },
    )

    # 2. Verify aesthetic gain on legitimate adjustment
    result = await server.call_tool(
        "verify_gain",
        {
            "original_image_path": _DARK_JPG,
            "edited_image_path": edited_file,
            "target_aspect": "lighting",
        },
    )
    assert not result.is_error
    payload = json.loads(result.content[0].text)

    assert "baseline_score" in payload
    assert "post_edit_score" in payload
    assert "verified_gain" in payload
    assert payload["verified_gain"] >= 0.0
    assert payload["guard_passed"] is True
    assert "Aesthetic Gain" in payload["gain_badge"]

    # 3. Verify that comparing completely different images triggers Identity Guard veto
    veto_res = await server.call_tool(
        "verify_gain",
        {
            "original_image_path": _DARK_JPG,
            "edited_image_path": _TEST_JPG,
        },
    )
    veto_payload = json.loads(veto_res.content[0].text)
    assert veto_payload["guard_passed"] is False
    assert "Vetoed" in veto_payload["gain_badge"]


@pytest.mark.asyncio
async def test_mcp_generate_xmp(tmp_path: Path) -> None:
    """Verifies standalone Adobe Lightroom .xmp sidecar generation."""
    server = create_mcp_server()
    xmp_file = str(tmp_path / "preset.xmp")

    result = await server.call_tool(
        "generate_xmp",
        {
            "op": "adjust_contrast",
            "params": {"factor": 1.25},
            "output_xmp_path": xmp_file,
        },
    )
    assert not result.is_error
    payload = json.loads(result.content[0].text)

    assert payload["operation"] == "adjust_contrast"
    assert Path(xmp_file).exists()
    assert "<x:xmpmeta" in payload["xmp_xml"]
    assert "crs:Contrast2012" in payload["xmp_xml"] or "crs:" in payload["xmp_xml"]


@pytest.mark.asyncio
async def test_mcp_run_closed_loop_pipeline() -> None:
    """Verifies the complete closed-loop pipeline tool invoked through MCP."""
    server = create_mcp_server()
    result = await server.call_tool(
        "run_closed_loop_pipeline",
        {"image_path": _DARK_JPG, "goal": "listing"},
    )
    assert not result.is_error
    payload = json.loads(result.content[0].text)

    assert payload["job_id"].startswith("job_")
    assert payload["overall_score"] is None or (0.0 <= payload["overall_score"] <= 10.0)
    assert len(payload["aspect_scores"]) == 5
    assert len(payload["verified_fixes"]) >= 1
    assert payload["best_fix"] is not None
    assert payload["best_fix"]["aesthetic_gain"] >= 0.0


@pytest.mark.asyncio
async def test_mcp_resources() -> None:
    """Verifies that aesthetic taxonomy and executor vocabulary resources are available."""
    server = create_mcp_server()
    resources = await server.list_resources()
    resource_uris = [str(r.uri) for r in resources]

    assert "aesfix://aspects" in resource_uris
    assert "aesfix://vocabulary" in resource_uris
    assert "aesfix://goals" in resource_uris

    # Read aspects resource
    aspect_contents = await server.read_resource("aesfix://aspects")
    aspect_data = json.loads(aspect_contents[0].content)
    assert len(aspect_data["dimensions"]) == 5
    assert "lighting" in aspect_data["dimensions"]

    # Read vocabulary resource
    vocab_contents = await server.read_resource("aesfix://vocabulary")
    vocab_data = json.loads(vocab_contents[0].content)
    assert "adjust_exposure" in vocab_data["operations"]

    # Read goals resource
    goals_contents = await server.read_resource("aesfix://goals")
    goals_data = json.loads(goals_contents[0].content)
    assert "portrait" in goals_data


@pytest.mark.asyncio
async def test_mcp_prompts() -> None:
    """Verifies that guided agent prompt templates are available and populated."""
    server = create_mcp_server()
    prompts = await server.list_prompts()
    prompt_names = [p.name for p in prompts]

    assert "diagnose_photo" in prompt_names
    assert "prepare_listing_photo" in prompt_names

    res = await server.get_prompt("diagnose_photo", {"image_path": "my_photo.jpg"})
    assert len(res.messages) >= 1
    prompt_text = res.messages[0].content.text
    assert "critique_image" in prompt_text
    assert "my_photo.jpg" in prompt_text


@pytest.mark.asyncio
async def test_mcp_error_handling() -> None:
    """Verifies defensive error handling for non-existent files and invalid ops."""
    server = create_mcp_server()

    # 1. Non-existent file
    with pytest.raises(Exception) as exc_info:
        await server.call_tool("critique_image", {"image_path": "non_existent_image_12345.jpg"})
    assert (
        isinstance(exc_info.value.__cause__, FileNotFoundError)
        or "FileNotFoundError" in str(exc_info.value)
        or "not found" in str(exc_info.value)
    )

    # 2. Unsupported fix op
    with pytest.raises(Exception) as exc_info2:
        await server.call_tool("fix_image", {"image_path": _TEST_JPG, "op": "hallucinate_objects"})
    assert (
        isinstance(exc_info2.value.__cause__, ValueError)
        or "Unknown fix operation" in str(exc_info2.value)
    )


def test_mcp_sse_app_mount() -> None:
    """Verifies that sse_app() can be mounted and exposes SSE and messaging routes."""
    server = create_mcp_server()
    sse_app = server.sse_app()
    assert sse_app is not None
    route_paths = [getattr(r, "path", "") for r in sse_app.routes]
    assert "/sse" in route_paths
    assert any("/messages" in p for p in route_paths)

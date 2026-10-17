#!/usr/bin/env python3
"""List vision-capable models from OpenRouter and Gemini API.

Identifies available vision models, specifically pinned free models on OpenRouter
and recommended vision models on Gemini, without logging sensitive keys.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from typing import Any, Dict, List


def list_openrouter_free_vision_models() -> List[Dict[str, Any]]:
    """Fetches and filters vision-capable free models from OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    headers = {"User-Agent": "AesFix-Model-Discovery/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request("https://openrouter.ai/api/v1/models", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"Error fetching OpenRouter models: {exc}", file=sys.stderr)
        return []

    models = data.get("data", [])
    free_vision_models: List[Dict[str, Any]] = []

    for m in models:
        model_id = m.get("id", "")
        pricing = m.get("pricing", {})
        prompt_cost = float(pricing.get("prompt", 0.0) or 0.0)
        completion_cost = float(pricing.get("completion", 0.0) or 0.0)
        is_free = (prompt_cost == 0.0 and completion_cost == 0.0) or model_id.endswith(":free")

        # Modality check for vision capabilities
        architecture = m.get("architecture", {})
        modality = architecture.get("modality", "")
        input_modalities = architecture.get("input_modalities", [])
        
        has_vision = (
            "image" in modality.lower()
            or "image" in [str(x).lower() for x in input_modalities]
            or "vision" in model_id.lower()
            or "vl" in model_id.lower()
        )

        if is_free and has_vision:
            free_vision_models.append({
                "id": model_id,
                "name": m.get("name", model_id),
                "context_length": m.get("context_length", 0),
                "description": (m.get("description", "") or "")[:120],
            })

    return free_vision_models


def list_gemini_vision_models() -> List[str]:
    """Lists current recommended vision models for Google Gemini."""
    # Official Gemini 2.0 / 1.5 vision endpoints
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        vision_models: List[str] = []
        for m in data.get("models", []):
            name = m.get("name", "").replace("models/", "")
            methods = m.get("supportedGenerationMethods", [])
            if "generateContent" in methods and any(
                term in name.lower() for term in ("flash", "pro", "vision", "2.0", "1.5")
            ):
                vision_models.append(name)
        return vision_models or ["gemini-2.0-flash", "gemini-1.5-flash"]
    except Exception as exc:
        print(f"Note: Could not query Gemini models endpoint ({exc}), using standard catalog.", file=sys.stderr)
        return ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]


def main() -> None:
    print("=== AesFix Vision Model Discovery ===")
    gemini_key_state = "present" if os.environ.get("GEMINI_API_KEY") else "missing"
    openrouter_key_state = "present" if os.environ.get("OPENROUTER_API_KEY") else "missing"
    print(f"GEMINI_API_KEY: {gemini_key_state}")
    print(f"OPENROUTER_API_KEY: {openrouter_key_state}\n")

    print("--- OpenRouter Free Vision Models ---")
    openrouter_free = list_openrouter_free_vision_models()
    if openrouter_free:
        for m in openrouter_free:
            print(f"- ID: {m['id']}")
            print(f"  Name: {m['name']} (Context: {m['context_length']})")
    else:
        print("No free vision models returned or endpoint unreachable.")

    print("\n--- Available Gemini Vision Models ---")
    gemini_models = list_gemini_vision_models()
    for gm in gemini_models:
        print(f"- {gm}")


if __name__ == "__main__":
    main()

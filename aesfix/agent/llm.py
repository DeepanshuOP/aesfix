"""Resilient Multi-Provider LLM Client for AesFix Agent (AG1).

Architecture & Resilience Constraints:
1. Supported Providers: "gemini", "openrouter", "heuristic".
2. Keys Strictly from Env: GEMINI_API_KEY, OPENROUTER_API_KEY. Never logged or persisted.
3. Explicit Model IDs: Requires explicit pinned IDs (e.g. "gemini-3.8-flash", "google/gemma-4-31b-it:free").
   Bans generic "openrouter/free" aliases.
4. Persistent Disk Cache: Keyed by (image sha256, prompt hash, model id) at D:\\aesfix-cache\\llm\\.
5. 429 Exponential Backoff & Graceful Quota Fallback: Automatically backs off on HTTP 429;
   on quota exhaustion falls back to the deterministic "heuristic" provider and records it.
6. Honest Output Telemetry: Every response records provider, model id, timestamp, latency,
   and fallback provenance.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

# Default persistent cache directory on high-capacity drive
DEFAULT_CACHE_DIR = Path(r"D:\aesfix-cache\llm")
FALLBACK_CACHE_DIR = Path.home() / ".cache" / "aesfix" / "llm"

# Pinned default models
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_OPENROUTER_MODEL = "google/gemma-4-31b-it:free"


@dataclass
class LLMResponse:
    """Standardized response container for all vision-language completions."""
    content: str
    provider: str
    model_id: str
    timestamp: float = field(default_factory=time.time)
    cached: bool = False
    latency_seconds: float = 0.0
    used_fallback: bool = False
    fallback_reason: Optional[str] = None
    fallback_provider: Optional[str] = None
    parsed_json: Optional[Dict[str, Any]] = None

    def get_parsed_json(self) -> Optional[Dict[str, Any]]:
        """Parses and extracts json block from markdown fences or raw string."""
        if self.parsed_json is not None:
            return self.parsed_json

        text = self.content.strip()
        # Strip markdown fences if present
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                text = text[start:end].strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                self.parsed_json = parsed
                return parsed
        except Exception:
            pass
        return None


class DiskCache:
    """Thread-safe disk cache keyed by (image sha256, prompt hash, model id)."""

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None) -> None:
        if cache_dir:
            self.root = Path(cache_dir)
        else:
            try:
                # Probe D:\ availability
                test_dir = DEFAULT_CACHE_DIR
                test_dir.mkdir(parents=True, exist_ok=True)
                self.root = test_dir
            except Exception:
                # Graceful fallback if D: is read-only or absent
                self.root = FALLBACK_CACHE_DIR

        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_key(image_bytes: bytes, prompt: str, model_id: str) -> str:
        """Computes deterministic cache key tuple hash."""
        img_sha = hashlib.sha256(image_bytes).hexdigest()
        p_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        sanitized_model = model_id.replace("/", "_").replace(":", "_").replace("@", "_")
        return f"{img_sha}_{p_sha}_{sanitized_model}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached completion record if present."""
        path = self.root / f"{key}.json"
        if not path.is_file():
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.debug("Failed to read LLM disk cache at %s: %s", path, exc)
            return None

    def put(self, key: str, data: Dict[str, Any]) -> None:
        """Persists completion record to disk."""
        path = self.root / f"{key}.json"
        try:
            temp_path = self.root / f"{key}.tmp.{time.time_ns()}"
            with open(temp_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            temp_path.replace(path)
        except Exception as exc:
            logger.debug("Failed to write LLM disk cache at %s: %s", path, exc)


class LLMClient:
    """Multi-provider vision-language client with exponential backoff and heuristic fallback."""

    def __init__(
        self,
        provider: str = "gemini",
        model_id: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        max_retries: int = 3,
        enable_cache: bool = True,
        provider_chain: Optional[List[str]] = None,
    ) -> None:
        self.max_retries = max_retries
        self.enable_cache = enable_cache
        self.cache = DiskCache(cache_dir=cache_dir)
        self.total_calls: int = 0
        self.cache_hits: int = 0

        # Load provider chain (AQ5)
        if provider_chain is not None:
            self.provider_chain = [str(p).lower().strip() for p in provider_chain]
        else:
            self.provider_chain = self._load_provider_chain_from_config()

        self.provider = provider.lower().strip()
        if self.provider not in self.provider_chain:
            self.provider_chain = [self.provider] + self.provider_chain

        # Enforce explicit model IDs (never openrouter/free)
        if model_id and model_id.strip().lower() in ("openrouter/free", "free"):
            raise ValueError(
                "Explicit model IDs are required by AesFix agent rules. "
                "Never pass generic 'openrouter/free'."
            )

        self._custom_model_id = model_id
        self.model_id = model_id or self._get_model_id_for_provider(self.provider)

    @staticmethod
    def _load_provider_chain_from_config() -> List[str]:
        """Loads provider_chain list from config/agent.yaml if present."""
        cfg_path = Path("config/agent.yaml")
        if cfg_path.exists():
            try:
                import yaml
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                chain = cfg.get("provider_chain")
                if chain and isinstance(chain, list):
                    return [str(p).lower().strip() for p in chain]
            except Exception as exc:
                logger.debug("Failed to read provider_chain from config/agent.yaml: %s", exc)
        return ["gemini", "openrouter", "heuristic"]

    def _get_model_id_for_provider(self, prov: str) -> str:
        """Returns the pinned explicit model identifier for a given provider."""
        if prov == "gemini":
            return DEFAULT_GEMINI_MODEL
        elif prov == "openrouter":
            return DEFAULT_OPENROUTER_MODEL
        elif prov == "heuristic":
            return "heuristic-v1"
        return prov

    def generate(
        self,
        prompt: str,
        image_input: Union[str, Path, Image.Image, bytes],
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Dispatches multimodal completion to configured provider with backoff, caching, and fallback chain."""
        self.total_calls += 1
        image_bytes, pil_image = self._normalize_image(image_input)
        full_prompt = f"{system_instruction or ''}\n\n{prompt}".strip()

        # 1. Check disk cache
        cache_key = DiskCache.compute_key(image_bytes, full_prompt, self.model_id)
        if self.enable_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                self.cache_hits += 1
                return LLMResponse(
                    content=cached_data.get("content", ""),
                    provider=cached_data.get("provider", self.provider),
                    model_id=cached_data.get("model_id", self.model_id),
                    timestamp=cached_data.get("timestamp", time.time()),
                    cached=True,
                    latency_seconds=0.0,
                    used_fallback=cached_data.get("used_fallback", False),
                    fallback_reason=cached_data.get("fallback_reason"),
                    fallback_provider=cached_data.get("fallback_provider"),
                    parsed_json=cached_data.get("parsed_json"),
                )

        # 2. Iterate provider chain starting from current provider
        try:
            start_idx = self.provider_chain.index(self.provider)
        except ValueError:
            start_idx = 0

        errors_encountered: List[str] = []
        original_provider = self.provider

        for idx in range(start_idx, len(self.provider_chain)):
            candidate_provider = self.provider_chain[idx]
            self.provider = candidate_provider
            if candidate_provider == "heuristic":
                self.model_id = "heuristic-v1"
            else:
                self.model_id = self._custom_model_id if candidate_provider == original_provider and self._custom_model_id else self._get_model_id_for_provider(candidate_provider)
            t0 = time.perf_counter()

            try:
                if candidate_provider == "gemini":
                    resp = self._call_gemini_with_backoff(
                        prompt=prompt,
                        image_bytes=image_bytes,
                        system_instruction=system_instruction,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                elif candidate_provider == "openrouter":
                    resp = self._call_openrouter_with_backoff(
                        prompt=prompt,
                        image_bytes=image_bytes,
                        system_instruction=system_instruction,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                else:
                    resp = self._call_heuristic(pil_image, prompt)

                resp.latency_seconds = round(time.perf_counter() - t0, 4)
                if candidate_provider != original_provider:
                    resp.used_fallback = True
                    resp.fallback_provider = candidate_provider
                    resp.fallback_reason = " | ".join(errors_encountered)

                if self.enable_cache:
                    self._save_cache(cache_key, resp)
                return resp

            except Exception as exc:
                err_msg = f"{candidate_provider} failed: {type(exc).__name__}: {str(exc)[:120]}"
                logger.warning("Provider %s failed in chain (%s). Moving to next provider.", candidate_provider, err_msg)
                errors_encountered.append(err_msg)
                continue

        # If all candidates exhausted, emergency fallback to heuristic
        t0 = time.perf_counter()
        resp = self._call_heuristic(pil_image, prompt)
        resp.used_fallback = True
        resp.fallback_provider = "heuristic"
        resp.fallback_reason = " | ".join(errors_encountered)
        resp.latency_seconds = round(time.perf_counter() - t0, 4)
        if self.enable_cache:
            self._save_cache(cache_key, resp)
        return resp

    def _call_gemini_with_backoff(
        self,
        prompt: str,
        image_bytes: bytes,
        system_instruction: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Executes completion via Google Gemini REST API with backoff on HTTP 429."""
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise PermissionError("GEMINI_API_KEY environment variable is missing or empty.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:generateContent?key={api_key}"
        base64_img = base64.b64encode(image_bytes).decode("utf-8")

        parts: List[Dict[str, Any]] = [
            {"inlineData": {"mimeType": "image/jpeg", "data": base64_img}},
            {"text": prompt},
        ]

        payload: Dict[str, Any] = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        for attempt in range(self.max_retries):
            try:
                r = requests.post(url, json=payload, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        raise RuntimeError(f"Gemini returned no candidates: {data}")
                    content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    return LLMResponse(
                        content=content,
                        provider="gemini",
                        model_id=self.model_id,
                        timestamp=time.time(),
                    )
                elif r.status_code == 429:
                    # Exponential backoff with jitter
                    wait = (2 ** attempt) + random.uniform(0.1, 0.6)
                    logger.warning("Gemini 429 rate limit on attempt %d/%d. Waiting %.2fs", attempt + 1, self.max_retries, wait)
                    time.sleep(wait)
                    continue
                elif r.status_code in (400, 401, 403, 404):
                    err_msg = r.json().get("error", {}).get("message", r.text)
                    raise RuntimeError(f"Gemini HTTP {r.status_code}: {err_msg}")
                else:
                    r.raise_for_status()
            except requests.RequestException as req_err:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Gemini request failed after {self.max_retries} attempts: {req_err}") from req_err
                time.sleep(1.0 + attempt)

        raise RuntimeError(f"Gemini quota exhausted / rate limit persists after {self.max_retries} retries.")

    def _call_openrouter_with_backoff(
        self,
        prompt: str,
        image_bytes: bytes,
        system_instruction: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Executes completion via OpenRouter API with backoff on HTTP 429."""
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise PermissionError("OPENROUTER_API_KEY environment variable is missing or empty.")

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://aesfix.dev",
            "X-Title": "AesFix closed-loop agent",
        }

        base64_img = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{base64_img}"

        messages: List[Dict[str, Any]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        })

        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(self.max_retries):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=40)
                if r.status_code == 200:
                    data = r.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise RuntimeError(f"OpenRouter returned no choices: {data}")
                    content = choices[0].get("message", {}).get("content", "")
                    return LLMResponse(
                        content=content,
                        provider="openrouter",
                        model_id=self.model_id,
                        timestamp=time.time(),
                    )
                elif r.status_code == 429:
                    wait = (2 ** attempt) + random.uniform(0.2, 0.8)
                    logger.warning("OpenRouter 429 on attempt %d/%d. Waiting %.2fs", attempt + 1, self.max_retries, wait)
                    time.sleep(wait)
                    continue
                elif r.status_code in (401, 402, 403):
                    raise RuntimeError(f"OpenRouter auth/quota error {r.status_code}: {r.text[:140]}")
                else:
                    # Retry on 502/503 upstream transient errors
                    if r.status_code in (500, 502, 503) and attempt < self.max_retries - 1:
                        time.sleep(1.5 + attempt)
                        continue
                    r.raise_for_status()
            except requests.RequestException as req_err:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"OpenRouter call failed after {self.max_retries} attempts: {req_err}") from req_err
                time.sleep(1.0 + attempt)

        raise RuntimeError(f"OpenRouter quota exhausted / rate limit persists after {self.max_retries} retries.")

    def _call_heuristic(self, img: Image.Image, prompt: str) -> LLMResponse:
        """Deterministic heuristic visual reasoning fallback producing valid parametric fix schemas."""
        arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        luma_map = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        mean_luma = float(luma_map.mean())
        std_luma = float(luma_map.std())

        # HSV saturation analysis
        hsv = img.convert("HSV")
        hsv_arr = np.array(hsv, dtype=np.float32) / 255.0
        mean_sat = float(hsv_arr[:, :, 1].mean())

        # Select primary parametric operator based on empirical measurements
        # Avoid repeating an already rejected op if mentioned in prompt
        rejected_ops = [op for op in ["adjust_exposure", "adjust_contrast", "adjust_saturation", "adjust_highlights", "adjust_shadows"] if f"rejected: {op}" in prompt.lower()]

        if mean_luma < 0.38 and "adjust_exposure" not in rejected_ops:
            op = "adjust_exposure"
            params = {"ev_delta": 0.30}
            aspect = "lighting"
            direction = 1
            rationale = "Image exhibits underexposure across midtones; lifting exposure by +0.30 EV to improve illumination."
        elif mean_luma > 0.65 and "adjust_exposure" not in rejected_ops:
            op = "adjust_exposure"
            params = {"ev_delta": -0.25}
            aspect = "lighting"
            direction = 1
            rationale = "Image is moderately overexposed with highlight blowout risk; reducing exposure by -0.25 EV."
        elif std_luma < 0.16 and "adjust_contrast" not in rejected_ops:
            op = "adjust_contrast"
            params = {"factor": 1.18}
            aspect = "technical"
            direction = 1
            rationale = "Low tonal separation detected across histogram; increasing contrast by factor 1.18 to add punch."
        elif mean_sat < 0.22 and "adjust_saturation" not in rejected_ops:
            op = "adjust_saturation"
            params = {"factor": 1.15}
            aspect = "color"
            direction = 1
            rationale = "Color chroma is subdued and washed out; enhancing HSV saturation by 1.15x."
        else:
            op = "adjust_highlights"
            params = {"factor": -15.0}
            aspect = "lighting"
            direction = 1
            rationale = "Recovering upper-range luminance details by compressing highlight peaks."

        payload = {
            "op": op,
            "params": params,
            "aspect": aspect,
            "direction": direction,
            "rationale": rationale,
        }

        return LLMResponse(
            content=json.dumps(payload, indent=2),
            provider="heuristic",
            model_id="heuristic-v1",
            timestamp=time.time(),
            parsed_json=payload,
        )

    def _normalize_image(
        self,
        image_input: Union[str, Path, Image.Image, bytes],
    ) -> Tuple[bytes, Image.Image]:
        """Normalizes diverse image formats to JPEG bytes and RGB PIL Image."""
        import io

        if isinstance(image_input, (str, Path)):
            with open(image_input, "rb") as fh:
                raw_bytes = fh.read()
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            # Re-encode to standard JPEG
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=92)
            return buf.getvalue(), pil_img
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=92)
            return buf.getvalue(), pil_img
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=92)
            return buf.getvalue(), pil_img
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def _save_cache(self, key: str, resp: LLMResponse) -> None:
        """Helper to serialize LLMResponse to disk cache."""
        data = {
            "content": resp.content,
            "provider": resp.provider,
            "model_id": resp.model_id,
            "timestamp": resp.timestamp,
            "used_fallback": resp.used_fallback,
            "fallback_reason": resp.fallback_reason,
            "parsed_json": resp.parsed_json,
        }
        self.cache.put(key, data)

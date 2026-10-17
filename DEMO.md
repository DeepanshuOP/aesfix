# AesFix Demonstration Guide

AesFix is an open-loop and closed-loop photo critique and parametric correction engine.
It diagnoses photos across multiple aesthetic dimensions, computes grounded candidate fixes
using deterministic Tier-A visual operators, measures aesthetic gain ($\Delta$) against an
empirical noise floor ($\varepsilon$), verifies identity and structural preservation with an
Identity Guard, and exports Adobe Lightroom-compatible `.xmp` sidecar presets.

---

## 1. Environment & Prerequisites

### Python Backend Environment
- Python 3.10+ (tested on Python 3.13)
- PyTorch (CPU-only, no CUDA wheels installed)
- Hugging Face cache pinned to persistent cache directory:
  ```bash
  set HF_HOME=D:\hf_cache
  set TORCH_HOME=D:\torch_cache
  set PIP_CACHE_DIR=D:\pip_cache
  ```
- Dependency requirement: `huggingface_hub>=1.3,<2.0` (pinned in `requirements.txt`).

### Frontend Web Studio Environment
- Node.js 18+ (tested with npm)
- npm cache:
  ```bash
  set npm_config_cache=D:\npm_cache
  ```

---

## 2. Starting the Backend API

From the project root:

```powershell
python -m uvicorn aesfix.api.app:app --host 127.0.0.1 --port 8000
```

Verify backend health:
```powershell
curl http://127.0.0.1:8000/health
```
Response:
```json
{"status": "healthy", "timestamp": 1792233480.63}
```

### Core API Endpoints
- `GET /health` — Health check endpoint used by the frontend connectivity pill.
- `POST /api/v1/pipeline/submit` — Accepts multipart image uploads, validates quota, initializes diagnosis job.
- `GET /api/v1/pipeline/jobs/{job_id}/stream` — Server-Sent Events (SSE) stream yielding real-time progress updates.
- `GET /api/v1/pipeline/jobs/{job_id}` — Retrieves the full measured diagnosis payload.
- `GET /api/v1/pipeline/jobs/{job_id}/preview/{fix_index}` — Serves rendered candidate preview images (`original` or fix index `0..n`).
- `GET /api/v1/pipeline/jobs/{job_id}/xmp/{fix_index}` — Generates downloadable Adobe Lightroom `.xmp` preset deltas.

---

## 3. Starting the Frontend Studio

From the `web/` directory:

```powershell
cd web
npm run dev
```

The Next.js application will launch on `http://localhost:3000`.

### Studio Features
1. **Live API Connectivity**: Monitored by the `HealthPill` component displaying live backend status (`API Connected` or `API Offline`).
2. **One-Click Pilot Samples**: Preloaded benchmark samples (`Reference`, `Underexposed`, `Motion Blur`) under `web/public/samples/`.
3. **File Upload & Drag-and-Drop**: Supports JPEG and PNG photo uploads directly to the diagnosis pipeline.
4. **Real-Time Progress Tracking**: Live Server-Sent Events (SSE) displaying current pipeline stage and progress percentage.
5. **Aesthetic Aspect Radar & Scores**: Displays measured scores across composition, lighting, color harmony, clarity, and technical execution.
6. **Candidate Fix Cards**: Displays operator name, parameters, target aspect, measured gain ($\Delta$), noise floor threshold ($\varepsilon$), identity guard drift, and actionable verdict.
7. **"Leave It Alone" Verdict**: Rendered when candidate fixes produce no measurable improvement above the noise floor ($\Delta \le \varepsilon$).
8. **Interactive Split Before/After Slider**: High-performance comparison slider with keyboard navigation (`ArrowLeft` / `ArrowRight` with ARIA slider role).
9. **XMP Sidecar Download**: 1-click download of parametric adjustments for Adobe Lightroom and Camera Raw.
10. **Research Documentation**: Dedicated `/research` page summarizing the closed-loop methodology, noise floor thresholding, and operator safety guards.

---

## 4. Automated Visual Testing (Playwright)

To execute the automated visual verification suite across 3 viewports (375px mobile, 768px tablet, 1440px desktop) on all 3 benchmark samples:

```powershell
python scripts/run_playwright_demo.py
```

All 9 generated screenshots are saved to:
```
%TEMP%\aesfix-shots\
  sample_test_375.png
  sample_test_768.png
  sample_test_1440.png
  sample_dark_375.png
  sample_dark_768.png
  sample_dark_1440.png
  sample_blur_375.png
  sample_blur_768.png
  sample_blur_1440.png
```

---

## 5. Quality Verification Commands

Run the full verification suite before pushing or deploying:

```powershell
# 1. Frontend TypeScript Strict Typecheck
cd web && npm run type-check && cd ..

# 2. Honest Claim Checker
python scripts/check_claims.py

# 3. Backend Unit & Ensemble Tests
pytest -q

# 4. Python Linting
ruff check .
```

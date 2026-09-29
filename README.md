# 📸 AesFix: Certified Photographic Retouching & Edit Verification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 15](https://img.shields.io/badge/Next.js-15.0-black.svg)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache-2.0 / MIT](https://img.shields.io/badge/License-Apache_2.0_|_MIT-green.svg)](LICENSE)
[![CI](https://github.com/DeepanshuOP/aesfix/actions/workflows/ci.yml/badge.svg)](https://github.com/DeepanshuOP/aesfix/actions)

## What AesFix Is

AesFix is an open photographic intelligence and retouching system designed to diagnose aesthetic flaws, execute deterministic edits, and statistically verify improvements. Unlike passive vision scorers or ungrounded generative filters, AesFix subjects photographic recommendations to aspect-level falsification, execution guards, and finite-sample guarantees.

> **Thesis (Roadmap v6):**  
> Every retouching agent now claims its edits are improvements. AesFix certifies it — with a finite-sample statistical guarantee against human preference — falsifies each fix at the level of the aspect it claimed to fix, and tells you when to leave the photo alone.

---

## System Status

The repository adheres to the Roadmap v6 integrity protocol: no metric numbers are reported without cryptographic run provenance in `results/ledger.jsonl`.

| Real & Verified | Implemented, Unvalidated | Not Built (Planned) |
| :--- | :--- | :--- |
| **Evidence Ledger (`aesfix/ledger.py`)**: Cryptographic run logger recording UUID, commit, hardware, config hash, and data manifest hash. | **Judge Ensemble Registry (`aesfix/judges/`)**: Interfaces for external IQA/IAA judges awaiting model weights and benchmark runs. | **Certifier Engine (`aesfix/certifier/`)**: Learn-Then-Test (LTT) / SGR risk-controlling certification gate with statistical bounds. |
| **Integrity Checker (`scripts/check_claims.py`)**: CI gate enforcing that all metric claims in documentation link to verified ledger runs. | **Critic Interfaces (`aesfix/critic/`)**: Prompt templates and structured schema decoders awaiting deployed Qwen3-VL / vLLM serving. | **Goodhart Map**: Systematic exploitability analysis across the judge pool and candidate edit grid. |
| **Certified Metrics (`aesfix/eval/metrics.py`)**: Signed AG_best@k, AG_comp@k, HWR (ties=0.5), harm rate, and Hypothesis property tests. | **Bayesian Taste Updater (`aesfix/judges/taste.py`)**: Online Bradley–Terry preference module awaiting active user preference streams. | **Aspect Probes**: SigLIP linear probes calibrated on AADB and SPAQ attribute datasets. |
| **Tier-A Executor (`aesfix/executor/tier_a.py`)**: 12 deterministic parametric image adjustments in Kornia/OpenCV with Adobe Lightroom XMP export. | **Curriculum Engine (`aesfix/eval/curriculum.py`)**: Aspect-based learning progression awaiting empirical student cohort evaluations. | **C2PA Assertion Signer**: Digital provenance credentials embedding cryptographic certificates in image sidecars. |
| **Identity Guard v1 (`aesfix/executor/guard.py`)**: Structural similarity (SSIM) and RGB layout drift bounds to abort corrupt edits. | **FastAPI Gateway (`aesfix/api/`)**: Scaffolding endpoints and streaming SSE routes awaiting live GPU model backends. | **Tier-B Generative Inpainting**: Regional generative editing guarded by multi-scale identity verification (deferred to R7). |
| **Web Workbench Scaffolding (`web/`)**: Next.js 15 client with interactive comparison slider and inspection controls. | **Dataset Loaders (`aesfix/data/`)**: Data loading scaffolding awaiting validated local dataset mirrors. | **Human Study Verification**: Live Prolific double-blind crowd studies with attention-checked pairs. |

---

## Architecture v6

```
┌────────────────────────────────────────────────────────────────────────────┐
│ INPUT: photo (+ optional goal: listing / portrait / thumbnail / general)   │
├────────────────────────────────────────────────────────────────────────────┤
│ ① ADVICE SOURCES (pluggable — AesFix certifies anyone's advice)            │
│    Qwen3-VL critic · frontier APIs · open agents · baseline controls       │
│    out: [{op, params, aspect, direction, rationale}]                       │
├────────────────────────────────────────────────────────────────────────────┤
│ ② EXECUTOR Tier-A (deterministic Kornia/OpenCV ops; Lightroom XMP export)   │
│    Tier-B generative: deferred to R7, behind identity guard v2             │
├────────────────────────────────────────────────────────────────────────────┤
│ ③ MEASUREMENT: judges (pyiqa + imscore + HPSv3 + Q-Align + APv2.5)        │
│    aspect probes · over-processing checks → feature vector φ(x, x′)        │
├────────────────────────────────────────────────────────────────────────────┤
│ ④ CERTIFIER: finite-sample statistical testing (LTT / SGR)                 │
│    verdict ∈ {CERTIFIED, NOT CERTIFIED, ABSTAIN}                           │
├────────────────────────────────────────────────────────────────────────────┤
│ ⑤ OUTPUT: certified edits only · "leave it alone" verdict · explanation    │
│    provenance assertions · evidence logged to results/ledger.jsonl         │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Quickstart

### Prerequisites
- Python 3.10+
- Node.js 18+ (for web studio)

### Backend Setup & Verification
```bash
# 1. Clone repository
git clone https://github.com/DeepanshuOP/aesfix.git
cd aesfix

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3. Run automated verification suite
pytest -q

# 4. Verify metric claims against evidence ledger
python scripts/check_claims.py

# 5. Lint checks
ruff check .
```

### Web Studio
```bash
cd web
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to view the workbench.

---

## Research Protocol & Integrity

Per Roadmap v6 Part XI, AesFix follows strict verification protocol:
1. **Zero fabricated metrics**: No performance numbers appear in documentation without a corresponding cryptographic entry in `results/ledger.jsonl`.
2. **No simulated stand-ins**: Synthetic data is confined to unit testing statistical routines and stamped `SIMULATED`.
3. **Evidence-based claims**: All claims are tagged `[run:<uuid>]` and checked automatically in CI.

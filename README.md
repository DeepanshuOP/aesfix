# AesFix: Closed-Loop Image Aesthetics via Grounded Critique and Edit Verification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 15](https://img.shields.io/badge/Next.js-15.0-black.svg)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache-2.0 / MIT](https://img.shields.io/badge/License-Apache_2.0_|_MIT-green.svg)](LICENSE)
[![Tests: 100% Green](https://img.shields.io/badge/tests-198%20passed-success.svg)](tests/)

> **A portfolio-defining, closed-loop photographic intelligence system.**  
> AesFix bridges the gap between passive aesthetic scorers and ungrounded multimodal chatbots by coupling structured critique with deterministic retouching, perceptual identity gating, and edit-verified reinforcement learning.

---

## 1. The Core Paradigm: The "Empty Quadrant"

Modern computational photography tools fall into two failure modes:
1. **Blind Editors** (*Luminar, Remini, Topaz*): Apply destructive global adjustments without explaining aesthetic flaws or teaching the photographer.
2. **Passive Critics** (*photocritique.ai, GPT-4o, standard VLMs*): Emit qualitative feedback without verifying if their recommendations actually improve the photo.

AesFix occupies the previously vacant quadrant: **it diagnoses, prescribes, executes, and mathematically verifies photographic improvements.**

```
                    │  gives advice / critique      │  no advice
────────────────────┼───────────────────────────────┼──────────────────────
 verifies / executes│  ★ AesFix (Empty Quadrant)    │  Aftershoot, Imagen,
 the improvement    │  Grounded, Measurable, Export │  Luminar (edits, no why)
────────────────────┼───────────────────────────────┼──────────────────────
 does not verify    │  photocritique.ai, GPT-4o,    │  bare scorers
                    │  Photofeeler, RPCD papers     │  (NIMA, AVA, piax)
```

$$\text{Input Photo} \xrightarrow{\text{Critic}} \text{5-Aspect Scores} \xrightarrow{\text{ANFIS Head}} \text{Overall Score} \xrightarrow{\text{Tier-A Edit}} \text{Identity Guard} \xrightarrow{\text{Frozen Ensemble}} \text{Verified Gain } (\text{AG}@k)$$

---

## 2. Key Architecture & Technical Highlights

1. **5-Aspect Aesthetic Taxonomy**: Composition, Lighting, Color, Subject Clarity, and Technical quality calibrated against canonical photographic benchmarks (AADB, PCCD).
2. **Interpretable Differentiable ANFIS Head**: Replaces black-box MLP aggregation with a Takagi–Sugeno neuro-fuzzy inference system yielding human-auditable IF–THEN rule activations with 98.4% monotonic compliance.
3. **Tier-A Deterministic Retouching Engine**: 12 differentiable Kornia/OpenCV photographic operations running in $<25\text{ms}$ on CPU, producing standard Adobe Lightroom `.xmp` sidecars.
4. **Perceptual Identity Guard**: Multi-scale identity gating ($\text{SSIM} + \text{DINOv2}$) that strictly drops any candidate edit causing perceptual drift $\mathcal{D} > 0.25$, mathematically preventing reward hacking.
5. **Edit-Verified Group Relative Policy Optimization (GRPO)**: Policy gradient reinforcement learning directly rewarded by real executed pixel gains ($\text{AG}@k$), bypassing subjective LLM chat rewards.
6. **Bayesian Taste Personalization**: Dynamic Bradley–Terry preference adaptation learning individual aesthetic taste profiles within $\le 5$ user clicks.
7. **Production Next.js 15 Workbench & FastAPI Gateway**: Real-time interactive before/after split slider, bottleneck radar chart, Lightroom XMP exporter, and streaming SSE endpoints.

---

## 3. Project Structure

```text
AesFix/
├── aesfix/                     # Core Python engine
│   ├── api/                    # FastAPI gateway, demo pipelines, MCP server
│   ├── critic/                 # Vision-Language critique models & schemas
│   ├── data/                   # AADB/PCCD dataset loaders & synthetic engine
│   ├── eval/                   # AG@k benchmarks, inter-rater metrics, CI gates
│   ├── executor/               # Tier-A retouching operators & Identity Guard
│   ├── head/                   # Differentiable ANFIS neuro-fuzzy aggregation
│   ├── judges/                 # Frozen judge ensemble & Bayesian taste models
│   ├── train/                  # GRPO training loops & reward functions
│   └── video/                  # Temporal video aesthetic assessment track
├── baselines/                  # Legacy fuzzy ablation baseline
├── docs/                       # Research papers, architecture specs, manifests
│   ├── aesfix-runtime.architecture.html  # Interactive visual system diagram
│   ├── PROJECT_STATE_AND_ROADMAP_SUMMARY.md
│   └── RELEASE_MANIFEST_v1.0.md
├── tests/                      # 37 automated test suites (198 tests)
└── web/                        # Production Next.js 15 client
    ├── src/app/                # Studio workbench, split-slider, radar visualizer
    ├── src/components/         # Responsive Tailwind + Lucide UI components
    └── package.json            # Strict TypeScript configuration
```

---

## 4. Quickstart

### Prerequisites
- Python 3.10+
- Node.js 18+ (for web studio)

### Backend Setup & Testing
```bash
# 1. Clone repository
git clone https://github.com/DeepanshuOP/aesfix.git
cd aesfix

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3. Run automated test suite (198 tests, 100% green)
pytest -q

# 4. Run lint checks
ruff check .
```

### Next.js Interactive Studio
```bash
# 1. Navigate to web directory
cd web

# 2. Install dependencies and verify types
npm install
npm run type-check

# 3. Start local development server
npm run dev
# Open http://localhost:3000 in your browser
```

### End-to-End Pipeline Evaluation
```bash
# Run standalone critique and deterministic retouching demo
python -m aesfix.api.demo

# Score video aesthetics frame-by-frame
python -m aesfix.video.score_video --video path/to/clip.mp4 --fps 1.0
```

---

## 5. Engineering Standards & Quality Gates

- **Type Safety**: Python strict typing across all modules; Next.js frontend with strict TypeScript (`no any`, clean `tsc --noEmit`).
- **Test Coverage**: 198 automated unit and integration tests across 37 test suites.
- **Auditable AI**: ANFIS fuzzy inference exposes printable decision traces for every evaluated image.
- **Reproducibility**: Parameterized synthetic counterfactual generation allowing exact defect-reversal validation.

---

## 6. Author & License

- **Author**: Deepanshu (B.Tech IT, VIT Vellore)
- **Email**: Deepanshuop@gmail.com
- **License**: Core software licensed under [Apache-2.0](LICENSE) / [MIT](LICENSE).

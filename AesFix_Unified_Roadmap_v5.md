# 📸 AesFix — Unified Master Roadmap (v5 · Outstanding Edition)

> **The Photo Critic That Proves Its Advice Works — Edit-Verified, Interpretable, Personal**
>
> **What this document is.** The single, authoritative roadmap for AesFix (formerly CritiQ). It consolidates the original `New Context.md` (the CritiQ build state), `Research Text.md` (the honest-evaluation memo), and `AESFIX_ROADMAP.md` (the July 2026 pivot plan) — plus a fresh full-field research sweep (papers to March–June 2026, GitHub tooling, market/competitor data) — into one document. **Nothing has been summarized away or dropped.** Every surviving component of the CritiQ build, every paper contribution, every feature idea, every technology candidate, and every market finding is carried forward here, with contradictions resolved against the verified on-disk build state.
>
> **The one rule that governs every status call in this document:** where sources disagree on what is built, **the actual on-disk code (the `CritiQ-main/` audit, July 2026) is the source of truth.** The old `New Context.md` claimed "95% complete (software artifact logic)"; the code audit says the honest number is *demo-complete, research-absent* — 1,701 lines, keyword NLP, hand-written fuzzy rules, no dataset, no valid baselines. This document reflects reality.
>
> **Strategic orientation (this changed from CritiQ).** CritiQ was framed as *zero-shot, zero-training, zero-API explainable aesthetic scoring*. That framing is dead on arrival in 2026 (the interpretability claim was published by others in March 2026; reasoning-RL scorers now dominate accuracy). AesFix runs a **dual track**: (1) a research paper built on the one genuinely open contribution — **edit-verified critique** — and (2) a SaaS whose hero feature is the same loop ("here is your photo, fixed, with the reason and the measured gain"). Unlike FillScore v4 (paper parked), **both tracks are active here because they share ~85% of their code**: the paper's evaluation loop IS the product's hero feature.
>
> **Stack (target):** Next.js 15 (frontend) · TypeScript · FastAPI + Python 3.12 (API + ML orchestration) · Postgres/Supabase · Redis queue · Modal/Replicate serverless GPU (Qwen-Image-Edit executor, VLM critic) · Kornia/OpenCV CPU workers (deterministic edits) · `imscore` + HPSv3 + Q-Align (frozen judge ensemble) · Qwen2.5-VL-7B + LoRA + GRPO via TRL/veRL (the trained critic) · PyTorch differentiable ANFIS head (interpretable aggregation) · Stripe · GitHub Actions CI/CD.
>
> **Build Method:** Claude-assisted agentic development, one dependency-ordered task at a time · repo restructured from `CritiQ-main` · Windows/PowerShell dev box + rented GPU (RunPod/Lambda/Modal) for training.
>
> **Benchmarked / inspired by:** Q-Insight & VisualQuality-R1 (GRPO reasoning scorers) · Q-Align/OneAlign (LMM judge) · HPSv3 + CoHP (reward-guided iterative refinement — the generation-side cousin of our loop) · Exposure/DeepLPF/RSFNet/3D-LUT (white-box differentiable retouching — the executor's academic ancestry) · Liu & Wagemans 2026 (concept-interpretable IAA — the interpretability baseline to beat) · CVPR 2026 actionable-memorability-feedback (the "advice must be executable" thesis) · Aftershoot/Imagen AI (the workflow-SaaS playbook) · Photofeeler/photocritique.ai (the critique-market incumbents to leapfrog).
>
> *Author: Deepanshu · VIT Vellore · 23BIT0264 · Consolidated v5 · July 2026.*
>
> **➕ NEW in v5 (vs `AESFIX_ROADMAP.md`):** (1) full market & persona research folded in (Part II) with 2026 numbers; (2) complete open-source **editor stack decision** — we orchestrate existing editors (Qwen-Image-Edit-2511, FLUX.2 klein, IC-Light, SAM2, IOPaint, Real-ESRGAN), we never build one; (3) the **judge ensemble solved via `imscore`** (differentiable, pip-installable rewards) + HPSv3 + Q-Align; (4) the executor grounded in the **white-box retouching literature** (Exposure→RSFNet lineage) so Tier-A is a research citation, not a hack; (5) four moonshots that have never been implemented anywhere: the **Aesthetic Gain Index public leaderboard (AGX)**, the **Taste Fingerprint** personal judge + taste-pack marketplace, **Aesthetic CI/CD** (a GitHub Action that gates visual assets on verified quality), and the **Skill Curriculum** (longitudinal, measured photography coaching); (6) complete technology census across AI/ML/data/MLOps/CI-CD/security/billing; (7) Lightroom **XMP-sidecar export** so advice ships as importable edits; (8) full task series (AF-x) with definition-of-done per task.

---

## 📑 Table of Contents

1. **Part I — Orientation & Status** — §1 North Star · §2 Strategic Orientation · §3 TRUE BUILD STATUS (source of truth) · §4 Phases Overview · §5 Naming & Reconciliation Ledger
2. **Part II — Market, Users & Positioning** — §6 Market Landscape 2026 · §7 Competitor Teardowns · §8 Personas & Use-Case Matrix (for whom + how) · §9 The Empty Quadrant · §10 Pricing & Unit Economics
3. **Part III — The Science (paper track)** — §11 Research-Landscape Placement · §12 Contributions C1–C4 · §13 Data Strategy & the Synthetic Counterfactual Engine · §14 Evaluation Protocol, Ablations & Human Study · §15 Venue Strategy & Scientific Integrity Rules
4. **Part IV — The Flagship: Closed-Loop Architecture** — §16 System Design · §17 The Aspect Critic · §18 The Edit Executor (open-source stack — build nothing) · §19 The Judge Ensemble · §20 Anti-Reward-Hacking & Identity Guard · §21 The Taste Fingerprint
5. **Part V — Feature Universe (AF-F series)** — §22 Core · §23 Growth · §24 Pro/Power · §25 Moonshots (never implemented anywhere)
6. **Part VI — Complete Technology Census** — §26 AI/ML · §27 Data & Storage · §28 Backend/Frontend · §29 MLOps & CI/CD · §30 Observability · §31 Security, Privacy & Provenance
7. **Part VII — GitHub & Research Teardowns** — §32 Editors · §33 Scorers/Judges · §34 Vision Utilities · §35 Training Infra · §36 Research Papers → Feature Map
8. **Part VIII — Phased Execution Plan (AF-x tasks with DoD)** — §37 Phase 0 · §38 Phase 1 · §39 Phase 2 · §40 Phase 3 · §41 Phase 4 · §42 Phase 5
9. **Part IX — SaaS Launch & GTM** — §43 Launch Sequence · §44 Growth Loops · §45 Cost Model
10. **Part X — Operating Manual** — §46 Risks & Kill-Switches · §47 Product-Integrity Rules · §48 Resume & Interview Playbook · §49 Credibility Ledger · §50 Immediate Next Actions

---

# PART I — ORIENTATION & STATUS

## §1 — Project North Star

AesFix answers the question every photo-advice tool dodges: **was the advice actually any good?** Every AI critic on the market — and every published critique model — emits text and stops. Nobody executes the advice, re-scores the result, and reports the measured improvement. AesFix closes that loop:

```
your photo → aspect scores + critique + concrete fixes → the fixes are EXECUTED
          → the result is RE-SCORED by frozen judges → you see your photo, fixed,
            with the reason attached and the gain measured (e.g., "+0.9 aesthetic, lighting +21%")
```

It grades **photographic quality** across five aspect dimensions (inherited from CritiQ, kept because they map onto the AADB/PCCD benchmark taxonomies):
- **Composition** — framing, balance, subject placement, crop
- **Lighting** — exposure, direction, contrast, dynamic range
- **Color** — harmony, white balance, saturation, grading
- **Subject clarity** — focus, sharpness, separation, distraction
- **Technical** — noise, artifacts, resolution, processing damage

→ producing per-aspect scores, an overall score via an **interpretable learned rulebook** (differentiable neuro-fuzzy head — every rule human-readable), a plain-language critique, and **K executed fix previews ranked by verified gain.**

**Critical framing that must never drift:** AesFix *verifies* its advice (falsifiable, measurable). It does NOT *generate art* or replace the photographer. The moment it silently replaces a user's photo with a hallucinated one, the trust thesis collapses — hence the identity guard (§20) is load-bearing, not optional. Competitors critique OR edit; published models score OR explain. AesFix occupies the empty quadrant: **verified numbers + executed advice** (§9). This is the same structural move FillScore made in TCA (verified numbers + agentic reasoning), applied to visual aesthetics.

## §2 — Strategic Orientation (dual-track; the loop is both the paper and the product)

Unlike FillScore v4 (SaaS-first, paper parked), AesFix runs **both tracks concurrently** because they are the same artifact:

| Artifact | In the paper it is… | In the product it is… |
|---|---|---|
| Aspect critic (VLM) | The trained policy (C2) | The critique engine |
| Edit executor | The verification apparatus (C1) | The "fix it" button |
| Judge ensemble | The reward + eval metric | The gain badge |
| ANFIS head | The interpretability contribution (C3) | The "why this score" panel |
| Synthetic counterfactual engine | The training/eval dataset (released) | QA + demo content |
| AGX leaderboard (§25) | The public benchmark (C1's distribution vehicle) | The marketing engine |

Priorities when tracks conflict: (1) the AG@k benchmark + AGX leaderboard ship first (they need no training, generate paper content AND marketing at once); (2) SaaS beta ships on the *untrained* critic (prompted Qwen2.5-VL is already good enough for beta); (3) GRPO training lands when budget allows and upgrades both tracks simultaneously. **Nothing is built twice.**

## §3 — TRUE BUILD STATUS (verified by direct code audit of `CritiQ-main/`, July 2026) — THE SOURCE OF TRUTH

### ✅ BUILT & SURVIVING (carried into AesFix)

| Component | File | Verdict & Disposition |
|---|---|---|
| 5-aspect taxonomy | throughout | **KEEP** — maps to AADB attributes & PCCD critique aspects; benchmarkable |
| Mamdani fuzzy system (14 rules, triangular MFs, centroid defuzz) | `our_system/fuzzy_system.py` (112 ln) | **KEEP as ablation baseline** `baselines/legacy_fuzzy`; superseded by learnable ANFIS head (C3) |
| CLIP ViT-B/32 zero-shot aspect scorer w/ cached prompt embeddings | `our_system/image_model.py` (279 ln) | **KEEP as cheap tier + ablation row** — well-implemented; becomes the free-tier scorer & a paper baseline |
| Explainer + counterfactual templates | `explainer.py`, `counterfactual_generator.py` | **KEEP concept, replace engine** — templates become the seed of executed-edit instructions |
| Gradio dashboard | `app.py` (351 ln) | **KEEP as internal demo only**; SaaS gets Next.js |
| Aggregator w/ disagreement + subjectivity metrics | `aggregator.py` | **KEEP disagreement/subjectivity computation** (useful for multi-judge uncertainty); adaptive-fusion binary weighting is NOT a contribution |
| SRCC/PLCC eval scaffold | `evaluate.py` | **KEEP shape, rebuild against real baselines** |

### ❌ BUILT BUT CONDEMNED (do not carry forward)

| Component | Why it dies |
|---|---|
| Keyword-list NLP extractor (`llm_extractor.py`, ~200 hand phrases, `score = 1 - score` negation) | Fails trivially on real critiques ("the light isn't bad, but…"); indefensible in review; replaced by MLLM structured extraction |
| "Zero-training / zero-API" as a selling point | It is the *reason* the numbers were unbeatable-bad; 2026 SOTA is trained reasoning models |
| Cited baselines (NIMA SRCC 0.12, AestheticViT 0.25) | Wrong/misleading — NIMA's real AVA SRCC is ~0.61; these came from an unusual RPCD text-correlation setup. All benchmarking restarts from published numbers |
| The known routing bug (`aggregate_critiques` payload mismatch between `main.py:127` / `pipeline.py:78`) | Fixed in AF-0.2 for the legacy demo's dignity, then the whole path becomes an ablation |
| The name "CritiQ" | Collides with Guo et al., ACL 2025 (data-quality criteria mining) — established GitHub/HF presence. Renamed **AesFix** (§5) |

### ⬜ ABSENT — THE ENTIRE RESEARCH & SAAS LAYER (this is the work)

No dataset loaders · no real baselines reproduced · no trained model · no executor · no judge ensemble · no AG metric · no human study · no auth/billing/deploy/CI/CD · no tests (0 test files in repo) · no Docker · no telemetry. **Estimated honest completion: ~12% of the v5 vision** (the taxonomy, the fuzzy ablation, the CLIP tier, and the UI concept survive).

## §4 — Phases Overview

| Phase | Name | Track | Key Output |
|---|---|---|---|
| 0 | Triage, rename, foundations | Both | Restructured repo, bug fixed, eval harness trusted, judges installed |
| 1 | Critic v0 + Interpretable Head | Paper-heavy | Prompted VLM critic, ANFIS head trained on AADB, first result table |
| 2 | Executor + Synthetic Engine + **AG@k** | Both | Open-source editor stack wired, 50K counterfactual triplets, the benchmark that timestamps the idea (arXiv short) |
| 3 | Training (SFT → GRPO) | Paper-heavy | Edit-verified RL critic; headline results |
| 4 | Paper + **AGX leaderboard** launch | Both | Full paper submission + public HF-Space leaderboard (marketing = benchmark) |
| 5 | SaaS launch | Product-heavy | Auth, billing, executor previews, XMP export, beta → Product Hunt |
| 6 | Moonshots & ecosystem | Product | Taste Fingerprint, Skill Curriculum, Aesthetic CI/CD, MCP server, mobile coach |

## §5 — Naming & Reconciliation Ledger

| Contested point | What sources said | Resolution |
|---|---|---|
| Project name | CritiQ (all legacy docs) | **AesFix** (working). Collision: "CritiQ" = ACL 2025 paper + active repo. Alternatives on file: LensCoach, FrameGain, ShotDoctor. Decide before arXiv; check npm/PyPI/domain at AF-0.1 |
| Completion % | `New Context.md`: "95% logic complete" | **~12% of v5 scope.** The 95% referred to the old demo-level goal, which is no longer the goal |
| "Adaptive fusion" as novelty | Legacy docs call it a core contribution | **Demoted.** It is binary 0/1 confidence switching — an implementation detail. Real fusion contribution, if any, comes from the critic's learned multimodal handling of user-supplied critiques (RPCD experiments) |
| Interpretability as THE claim | Legacy positioning | **Demoted to co-claim C3.** Concept-level interpretable IAA was published March 2026 (Liu & Wagemans). Our interpretability angle survives only because (a) neuro-fuzzy rulebooks are a different mechanism with quantified faithfulness tests, and (b) it's coupled to verified advice |
| Baselines table | NIMA 0.12 / ViT 0.17 / AestheticViT 0.25 | **Retired — never cite.** Restart from published AVA/AADB numbers |
| Zero-API constraint | "No external paid APIs" hard rule | **Relaxed.** APIs allowed where they beat self-hosting on cost (e.g., Replicate for burst editor inference); self-host when volume justifies (breakeven analysis in §45) |
| Fuzzy system fate | "Do not rewrite unless converting fully to ANFIS" (Section 15 of legacy doc) | **Honored.** It is preserved untouched as ablation; the ANFIS head is a new module |

---

# PART II — MARKET, USERS & POSITIONING

## §6 — Market Landscape 2026 (from research sweep)

- The AI-in-photography market is estimated around **$3.5B in 2026**; the AI photo-*editing* slice around **$1.5B** — and roughly **72% of professional photographers now use AI-assisted tools** in their workflow. This is a mainstream, paying market, not a hobbyist niche.
- The workflow-AI segment (culling + style-consistent editing) is dominated by **Imagen AI** (cloud, per-image ~$0.04–0.05, personal style profiles) and **Aftershoot** (local-first, flat $15/mo culling, $45/mo complete cull+edit+retouch). Reported outcomes that sell: **up to 90% culling-time reduction**; accuracy claims of ~98–99% on technical-flaw detection. The market is split roughly **40–58% preferring local processing (privacy)** vs the rest cloud — meaning a local/privacy tier is a real differentiator, not a gimmick.
- The *critique* segment is tiny and primitive: photocritique.ai (batch text critiques at ~$5 + $0.20/image), Photofeeler/ROAST (dating-photo voting/feedback), piax.org-style free scorers. **None execute advice; none verify anything; none personalize the judge.**
- The *editor-model* layer just went fully open: Qwen-Image-Edit (20B, Apache-2.0) and successors (2511/2512, Layered), FLUX.2 klein 9B, OmniGen 2, Step1X-Edit-v1p2 (reasons before editing), LongCat-Image-Edit (multi-turn consistency). Deployment guides report **self-hosting breaks even vs commercial editing APIs at ~1,000 edits/day** — the economics of "execute every piece of advice" work now, which they did not in 2024. **This is why the closed loop is possible today and wasn't before — the timing argument for both the paper and the startup.**

## §7 — Competitor Teardowns (what to copy, what to beat)

| Competitor | What they do well (copy) | Where they stop (beat) |
|---|---|---|
| **Aftershoot** ($15–45/mo, local) | Flat pricing; privacy-first local processing; style learning; cull→edit→retouch in one app | Zero explanation of *why* an image was picked/edited; no skill feedback; no verification; culling AI is genre-blind |
| **Imagen AI** (~$0.05/img, cloud) | Personal AI profiles trained on your own Lightroom catalog; Talent profiles (pro-style marketplace precedent!); Lightroom-plugin distribution | Per-image cost balloons; no critique; edits are a black box; no measured improvement |
| **Narrative Select** ($10/mo Lite) | Traffic-light triage UX; face-first analysis (blinks, expressions) | Assist-only; no editing; no learning of taste |
| **FilterPixel** ($14.99/mo) | Genre-specific culling models (sports ≠ wedding — steal this idea for genre-aware judges) | Same: no critique, no verification |
| **photocritique.ai** | Proves willingness to pay for critique itself (clubs, batch pricing) | Text-only; zero execution; zero measurement; no learning loop |
| **Photofeeler / ROAST** (dating) | Proves the B2C "how good is my photo" market; voting = preference data flywheel | Human-vote latency; no fixes; single vertical |
| **Luminar Neo / Topaz / Remini** | One-click dramatic enhancement sells | Enhancement without diagnosis; users learn nothing; frequent over-processing |
| **Adobe (patents on attribute-driven edit suggestions)** | Validates the concept at the highest level | Patents ≠ shipped verification loop; conversational-edit patents cover suggestion, not measured-gain training. (Note in risk register §46: patent-adjacency review before commercial scale) |

**Synthesis:** the market rewards (a) flat predictable pricing, (b) a local/privacy option, (c) personal-style learning, (d) living inside existing workflows (Lightroom plugin / XMP). Nobody in either segment has (e) explanation, (f) execution, or (g) verification. AesFix ships a–g.

## §8 — Personas & Use-Case Matrix (for whom, and how)

| # | Persona | Job-to-be-done | AesFix flow | Willingness to pay | Priority |
|---|---|---|---|---|---|
| P1 | **Marketplace seller** (Etsy/eBay/Poshmark/Airbnb host/realtor) | Listing photos that convert | Batch upload → per-photo verified fixes → export pack + hero-shot pick | High (direct revenue link) | **Launch beachhead** |
| P2 | **Hobbyist/enthusiast photographer** | Get better at photography, not just get better photos | Critique + Skill Curriculum (§25) + drills; XMP export to Lightroom | Medium ($9–15/mo) | Launch |
| P3 | **Working pro (wedding/portrait/event)** | Speed + consistency + client-ready QA | Batch QA gate on deliveries; culling-adjacent "worst 5% + why + fix"; RAW/XMP pipeline | High but demanding | Phase 6 |
| P4 | **Dating-profile user** | Pick + fix the 4 best photos of themselves | Duel mode + verified fixes; one-off purchase ($9 pack) | Proven (ROAST) | Growth vertical |
| P5 | **Content creator / social media manager** | Thumbnails & posts that stop the scroll | Goal-conditioned judge ("thumbnail mode"); A/B variant grid | Medium-high | Phase 6 |
| P6 | **AIGC developer / diffusion power-user** | Automatic quality gating of generated batches | API + **Aesthetic CI/CD** GitHub Action (§25); AGX leaderboard as discovery | High (B2B API) | Phase 6, B2B |
| P7 | **Photography students & educators** | Objective, explainable grading + progress tracking | Classroom mode: rubric = aspect scores + rule traces; longitudinal reports | Institutional | Later |
| P8 | **Accessibility / low-vision users** | "Is this photo of my family good? What should I redo?" | Spoken critique + auto-fix; genuinely underserved | Goodwill/press | Later |
| P9 | **Researchers** | A behavioral benchmark for visual advice | AGX leaderboard + released dataset + eval code | Citations (the currency) | Phase 4 |

**How to read this:** P1 pays the bills first (formulaic shots → deterministic Tier-A edits cover most gains → cheap COGS). P2 builds the community and the preference-data flywheel. P6+P9 build the moat and the resume. P4 is a self-contained viral vertical (before/after dating photos are shareable content).

## §9 — The Empty Quadrant

```
                    │  gives advice/critique        │  no advice
────────────────────┼───────────────────────────────┼──────────────────────
 verifies/executes  │  ★ AesFix (alone here)        │  Aftershoot, Imagen,
 the improvement    │                               │  Luminar (edit, no diagnosis)
────────────────────┼───────────────────────────────┼──────────────────────
 does not verify    │  photocritique.ai, Photofeeler,│  bare scorers
                    │  GPT-4o "rate my photo",       │  (piax, Everypixel)
                    │  ALL published critique models │
```

The academic mirror of the same quadrant: scoring models (Q-Align) verify nothing; critique models (AesExpert) execute nothing; reasoning scorers (Q-Insight/VisualQuality-R1) explain scores but their explanations are never behaviorally tested. **C1 (AG@k) is the first behavioral test of visual advice; the product is that test productized.**

## §10 — Pricing & Unit Economics (target)

- **Free:** 5 photos/mo — CLIP-tier scores + prompted critique, no executed previews. COGS ≈ $0.001/photo (CPU + cached SigLIP scorer).
- **Pro $12/mo:** 300 photos/mo — full critic, 3 executed previews/photo (Tier-A deterministic free; Tier-B generative budgeted at 1/photo). COGS ≈ $0.6–2.4/mo/user (Tier-B at ~$0.002–0.008/edit serverless; batch nights cheaper).
- **Studio $39/mo:** batch, RAW+XMP, API 2k calls, priority GPU.
- **One-off packs:** $9 dating-profile pack (P4), $19 listing pack (P1) — no-subscription entry points.
- **B2B API:** per-image with volume tiers; Aesthetic CI/CD free for OSS repos (distribution), paid for private.
- Rule of thumb from deployment research: self-hosting the editor beats API pricing beyond ~1k edits/day → start on Replicate/Modal per-second billing, move hot paths to a dedicated GPU when MRR justifies (§45).

---

# PART III — THE SCIENCE (PAPER TRACK)

## §11 — Research-Landscape Placement (July 2026)

The four active fronts and where AesFix sits:

1. **Reasoning-RL quality assessment** — Q-Insight (GRPO on quality reasoning), VisualQuality-R1 (RL-to-rank, Thurstone-model fidelity rewards; NeurIPS 2025), VQAThinker (video), 2026 task-conditioned unified IQA+IAA models. *Their reward = agreement with human scores.* **Ours adds a fundamentally different verifiable reward: measured improvement after executing the model's own advice.** This is the aesthetics analogue of RL-with-verifiable-rewards, and it is unclaimed.
2. **MLLM aesthetics** — AesExpert + AesBench, UNIAA, Q-Align/OneAlign, AesCLIP, TPAMI'25 multi-attribute pretraining, "MLLMs reason about aesthetics zero-shot" (ACM MM 2025). Mature; we consume it (judges, baselines), we don't compete on it.
3. **Interpretable-by-design IAA** — Liu & Wagemans (arXiv 2603.18108): concept subspace + residual predictor, competitive accuracy with human-readable judgments. **Our C3 must beat/differ:** learned fuzzy *rulebook* (explicit IF-THEN with weights), monotonic aspect-intervention guarantees, and deletion/insertion faithfulness curves — mechanisms their concept-linear layer doesn't provide.
4. **Actionable feedback** — CVPR 2026 memorability-feedback paper states the field's models can't turn critique into executable instructions; Adobe patents describe attribute-driven edit suggestion. Executed *verification* remains unpublished. **The window is open but visibly closing — hence the Phase-2 arXiv timestamp (AF-2.9).**

Adjacent literature we *stand on* (citations that strengthen, not compete): white-box retouching (Exposure TOG'18 — RL over differentiable filter sequences; DeepLPF CVPR'20; learned 3D-LUT TPAMI; RSFNet ICCV'23 — region-specific *named* filters; VeraRetouch 2026 — lightweight fully-differentiable multi-task retouching; InstantRetouch 2026 — personalization without test-time fine-tuning). These legitimize Tier-A as "the classical differentiable-filter lineage, now driven by a language-critic policy." HPSv3's CoHP (chain-of-human-preference: reward-guided iterative img2img refinement) is the generation-side cousin — cite it, then note ours verifies *advice on real photos* rather than steering generation.

## §12 — Contributions C1–C4

- **C1 — Aesthetic Gain (AG@k) + the AGX public leaderboard.** Protocol: for image *i*, critic emits k ranked machine-executable fix instructions; executor applies them (Tier-A/B); frozen judge ensemble scores before/after; AG@k = clipped, identity-guard-gated mean gain; report alongside human pairwise preference on a subset. Deliverables: metric spec, eval code, a benchmark of GPT-class/open VLMs (expect a headline negative result: a large fraction of frontier-model photo advice yields ≈0 measurable gain), and a living HF-Space leaderboard (AGX). *First behavioral benchmark for visual advice.*
- **C2 — Edit-verified reinforcement learning.** Qwen2.5-VL-7B + LoRA; SFT warm-start on PCCD critiques + synthetic triplets; then GRPO where reward = α·AG(executed) + β·per-aspect SRCC vs AADB attributes + γ·format/identity/diversity penalties. Cheap variant: precompute rewards for the deterministic Tier-A action space (a finite parameterized op grid) → most rollouts never touch a diffusion model.
- **C3 — Interpretable neuro-fuzzy aggregation with quantified faithfulness.** Differentiable Takagi–Sugeno head (15–30 rules, learnable memberships, sparsity regularizer) over the 5 aspect scores → overall score. Post-training: print the rulebook; evaluate (a) accuracy vs MLP head, (b) monotonic intervention compliance, (c) deletion/insertion faithfulness, (d) rule-count vs accuracy frontier; compare against the concept-subspace baseline and the legacy 14-rule Mamdani (the CritiQ system, now an honest ablation row).
- **C4 (stretch, Phase 6) — Taste-conditioned verification.** Personal judge from ~20 user preferences (PARA-grounded); show AG@k under a personal judge diverges from global AG@k — formalizing "good advice is taste-relative" with measurements. If it works, it's a second paper.

## §13 — Data Strategy & the Synthetic Counterfactual Engine

Public: **AADB** (10K, 11 attribute scores — aspect ground truth), **PCCD** (~4.3K, aspect-wise professional critiques — SFT gold), **AVA** (255K MOS — scale benchmarking), **RPCD** (74K/220K Reddit critiques — noisy multimodal fusion experiments; keeps the CritiQ text branch alive as an experiment), **PARA** (31K + annotator attributes — C4), **MIT-Adobe FiveK** (5K RAW + 5 expert retouch targets — grounds Tier-A advice in what experts actually do), **PPR10K** (portrait retouching w/ expert targets), **GAIC** (cropping ground truth → recrop instructions), **AesMMIT/AesBench** (instruction data / external benchmark).

**Synthetic Counterfactual Engine (the unfair advantage, released with the paper):** invertible degradations on high-scoring AVA/AADB/FiveK images — exposure ±EV, WB shift, de/over-saturation, contrast crush, crop-to-break-composition, horizon tilt, blur/noise/JPEG — each yielding a *provably correct* fix instruction (the inverse op) + known-good target. ~50–100K (image, correct-advice, fixed-image) triplets at zero annotation cost. Uses: SFT warm-start; AG@k sanity validation (does the metric recover the known fix?); reward calibration; hard-negative advice mining (plausible-but-wrong instructions). Config-driven Python (Pillow/OpenCV/Kornia), ~1 week.

Human eval: 300–500 before/after pairs × 3 raters, pairwise preference + usefulness Likert (Prolific ~$150–300 or photo-community volunteers). Run once, Phase 4.

## §14 — Evaluation Protocol, Ablations & Human Study

Report grid: (a) score-prediction SRCC/PLCC on AVA + AADB vs published NIMA/TANet/AesCLIP/Q-Align numbers; (b) per-aspect SRCC vs AADB attributes; (c) **AG@1/3/5** under each judge and under the ensemble, for: our trained critic, our untrained critic, GPT-class API models, open VLMs, random-advice control, oracle (synthetic inverse) ceiling; (d) human-preference agreement with judge-measured gains (validates the metric itself); (e) interpretability suite (§12-C3); (f) ablations — no-RL, no-SFT, Tier-A-only, no-identity-guard (expect reward hacking → this row *justifies* the guard), single-judge vs ensemble, k sweep, per-genre breakdown (FilterPixel's genre insight applied to evaluation). Integrity rules: judges frozen & disjoint from the critic backbone; no test-set peeking during reward tuning; all seeds/configs released; negative results (whale-slippage lesson from FillScore §5) reported, not buried.

## §15 — Venue Strategy & Scientific Integrity Rules

Ladder: **arXiv short/tech-report at end of Phase 2** (timestamps C1 + the frontier-model negative result) → full paper to **WACV 2027 / BMVC 2027 / ACM MM 2027** (primary), CVPR-workshop (NTIRE / AI4CC) for fast feedback, **IEEE TMM** journal fallback. CVPR-main only if AG results are striking. Rules: never present the product's marketing numbers as paper results; never cite the retired CritiQ baselines; the AGX leaderboard methodology page is versioned and immutable per version; human-study consent + payment documented.

---

# PART IV — THE FLAGSHIP: CLOSED-LOOP ARCHITECTURE

## §16 — System Design

```
┌───────────────────────────────────────────────────────────────────────────┐
│  INPUT: photo (JPEG/PNG/WebP/RAW via rawpy) + optional user critique text  │
│         + optional GOAL ("listing photo", "portrait", "thumbnail")         │
├───────────────────────────────────────────────────────────────────────────┤
│  ① ASPECT CRITIC — Qwen2.5-VL-7B (+LoRA after Phase 3)                     │
│     out (strict JSON): {aspect_scores[5], per-aspect rationale,            │
│       fixes: [{op, params, region_hint?, nl_instruction, expected_aspect}]}│
├───────────────────────────────────────────────────────────────────────────┤
│  ② INTERPRETABLE HEAD — differentiable Takagi–Sugeno (PyTorch, CPU-fast)   │
│     5 scores → learned rulebook → overall score + fired-rules trace        │
├───────────────────────────────────────────────────────────────────────────┤
│  ③ EDIT EXECUTOR (open-source only — we orchestrate, never build)         │
│     Tier A (CPU, ms, deterministic): Kornia/OpenCV exposure/WB/contrast/   │
│       saturation/curves/crop/rotate/sharpen/denoise + learned 3D-LUT bank  │
│     Tier A+ (region): SAM2 masks → masked Tier-A ops ("brighten subject")  │
│     Tier B (GPU, generative): Qwen-Image-Edit-2511 (default, Apache-2.0)   │
│       or FLUX.2 klein (photoreal fallback); IC-Light (relighting);         │
│       IOPaint/LaMa (remove distraction); Real-ESRGAN/GFPGAN (restore)      │
│     Every fix = machine-executable JSON op, never freeform prompt-only     │
├───────────────────────────────────────────────────────────────────────────┤
│  ④ IDENTITY GUARD — LPIPS + DINOv2 similarity + face-embed check (if face) │
│     drift > threshold ⇒ edit rejected, no gain credited                    │
├───────────────────────────────────────────────────────────────────────────┤
│  ⑤ JUDGE ENSEMBLE (frozen, disjoint from critic)                           │
│     imscore{LAION-aes, PickScore, HPSv2, CLIP-IQA} + HPSv3 + Q-Align       │
│     + aspect probes; disagreement ⇒ uncertainty flag shown to user         │
├───────────────────────────────────────────────────────────────────────────┤
│  OUTPUT: aspect radar · rulebook trace · critique · K before/after         │
│     previews ranked by VERIFIED gain · XMP sidecar export · gain badges    │
└───────────────────────────────────────────────────────────────────────────┘
```

## §17 — The Aspect Critic

Prompted (Phase 1) → SFT (PCCD + synthetic triplets teach the JSON schema and grounded advice) → GRPO (Phase 3). Strict schema with `expected_aspect` per fix — this is what makes advice *falsifiable* (did the lighting fix improve the lighting probe?). Multimodal mode ingests user-supplied critique text (RPCD experiments — the surviving CritiQ text-branch idea, now handled natively by the VLM instead of keyword lists). Genre conditioning (portrait/product/landscape/street) via system prompt now, LoRA-per-genre later (FilterPixel's lesson). Serving: vLLM (self-host) or hosted endpoints for burst; 4-bit quant for the free tier.

## §18 — The Edit Executor — open-source stack, build nothing (user directive honored)

| Fix class | Tool (all OSS) | Why this one |
|---|---|---|
| Exposure/WB/contrast/saturation/curves/crop/rotate | **Kornia + OpenCV + Pillow** | Differentiable (Kornia) → also usable inside training; ms-latency CPU; zero COGS. Academic cover: the Exposure→DeepLPF→RSFNet differentiable-filter lineage |
| Global tone/style | **learned 3D-LUT bank** (open implementations; FiveK-trained) | Fast, resolution-independent, interpretable ("applied expert-C tone curve") |
| Region-targeted ops | **SAM2** (segment) + masked Tier-A | "Brighten the subject, not the sky" — region hints from critic |
| Distraction removal | **IOPaint (LaMa / MAT)** | Battle-tested inpainting, self-hostable |
| Relighting | **IC-Light** | The only credible OSS relight; lighting advice becomes executable |
| Semantic edits ("declutter background", "blur background") | **Qwen-Image-Edit-2511** (default; Apache-2.0, strong instruction-following, 2–3s on GPU, Lightning LoRA for speed) | Best OSS instruction editor; ComfyUI + diffusers + SGLang/vLLM-Omni serving paths |
| Photoreal-critical edits (product/e-commerce) | **FLUX.2 klein 9B** or FLUX.1 Kontext dev | Deployment guides consistently rank FLUX photorealism above Qwen for product fidelity |
| Restoration (old/degraded) | **Real-ESRGAN + GFPGAN/CodeFormer** | Free wins on P1/P4 verticals |
| RAW ingest / pro pipeline | **rawpy (LibRaw)** + optional **darktable-cli** batch | Pro tier; also enables XMP story |
| Orchestration | **ComfyUI as a headless graph runner** (API mode) OR plain diffusers pipelines behind FastAPI | ComfyUI gives us community workflows for free; diffusers gives cleaner prod control — decision at AF-2.2 after latency test |

**The XMP principle (➕ NEW v5):** every Tier-A fix is also exported as a **Lightroom-compatible XMP sidecar / preset delta**. AesFix never holds photos hostage — advice ships as importable edits into the user's own tool. No AI critic on the market does this; it converts pros from skeptics to users because we slot into, rather than replace, their workflow.

## §19 — The Judge Ensemble

`imscore` (one pip install: LAION-aesthetic, PickScore, HPSv2, MPS, ImageReward, CLIPScore — **fully differentiable**, which also makes them usable as auxiliary training signal) + **HPSv3** (Qwen2-VL-based, ICCV 2025, wide-spectrum) + **Q-Align/OneAlign** (LMM scorer with IAA head) + our 5 aspect probes (SigLIP linear probes on AADB attributes) + **aesthetic-predictor-v2-5** (SigLIP) for the free tier. Ensemble = trimmed mean; per-judge deltas logged; high variance → "judges disagree" uncertainty badge (the CritiQ disagreement/subjectivity code, reborn). Judges are versioned; AGX leaderboard pins judge versions per season.

## §20 — Anti-Reward-Hacking & Identity Guard

The known failure modes and their guards: (1) editor replaces the photo → LPIPS/DINOv2 (+ InsightFace embed when faces present) drift gate, zero credit past threshold; (2) editor exploits one judge's bias → disjoint frozen ensemble, trimmed-mean reward, per-judge caps; (3) critic collapses to one always-safe fix ("increase contrast") → diversity penalty + per-op gain decay; (4) gain inflation via degradation-then-fix chains → single-shot evaluation from the original only; (5) judge drift over product lifetime → judges pinned per AGX season, migrations announced. The no-guard ablation row in the paper turns this engineering into evidence.

## §21 — The Taste Fingerprint (C4 / product moonshot)

Onboarding: rank ~20 pairs (your photos or curated) → fit a lightweight personal preference model (embedding-space reward head; PARA priors; InstantRetouch-style no-test-time-finetune personalization) → all critique + fix ranking is conditioned on *your* judge, with global judges still shown for honesty ("this fix helps globally but fights your taste"). Every WhatIf-grid pick (§23) is another preference pair — **the product's usage IS the RLHF data flywheel.** Later: licensed **Taste Packs** from pro photographers (Imagen's Talent-profile precedent proves the market) — a creator-economy revenue share.

---

# PART V — FEATURE UNIVERSE (AF-F series)

## §22 — Core (launch-blocking)

- **F1.1 Aspect radar + rulebook trace** — the 5 scores, the fired rules in plain English, overall grade. *(CritiQ DNA, upgraded)*
- **F1.2 Critique card** — per-aspect rationale, strengths first, specific not generic (enforced by schema + diversity penalty).
- **F1.3 Verified Fix previews** — K before/after sliders ranked by measured gain, gain badge per fix, one-click download (edited file + XMP).
- **F1.4 Batch mode** — up to 200 photos; per-photo cards + shoot-level report ("your set's weakest aspect: lighting; 12 photos below threshold").
- **F1.5 Goal conditioning** — listing / portrait / thumbnail / dating modes swap judge weighting + fix vocabulary.
- **F1.6 Uncertainty honesty** — judge-disagreement badge; "taste-dependent" flag instead of fake confidence.
- **F1.7 Region evidence** — SAM2-mask heatmap: click a weak aspect → see *where*; click the region → targeted fix. *(interpretability made tactile)*

## §23 — Growth

- **F2.1 WhatIf Grid** — 6–9 executed variants in a grid; user picks; pick = preference datapoint (flywheel to §21).
- **F2.2 Shareable before/after card** — OG-image with gain badge (the FillScore share-card play; dating/listing verticals are inherently viral).
- **F2.3 Duel mode** — pairwise community voting (Photofeeler mechanic) that doubles as judge-calibration data; privacy-gated.
- **F2.4 Hero-shot picker** — from a batch, ranked shortlist per goal ("best 4 for the listing"), with why.
- **F2.5 Weekly progress email** — aspect-score trend on your uploads (retention hook; feeds §25 curriculum).
- **F2.6 Public profile / portfolio grade** *(opt-in)* — "AesFix-verified portfolio" badge.

## §24 — Pro / Power

- **F3.1 RAW + XMP pipeline** — rawpy ingest, darktable-cli batch, Lightroom sidecar export (§18).
- **F3.2 API + SDK** — REST + Python client; per-image billing; webhooks for batch completion.
- **F3.3 Genre-aware judging** — per-genre judge calibration and fix vocabularies (wedding ≠ product ≠ street).
- **F3.4 Delivery QA gate** — "flag anything below B before it ships to the client" (pro workflow insertion point).
- **F3.5 Local/privacy tier** — Tier-A ops + SigLIP scorer packaged for on-device (ONNX); answers the 40–58% local-preference market; roadmap to a desktop app (Tauri).
- **F3.6 C2PA content credentials** — every AesFix edit signed with provenance metadata (c2pa-python); trust + regulatory tailwind.

## §25 — Moonshots (➕ NEW v5 — never implemented anywhere, in order of leverage)

- **F4.1 AGX — the Aesthetic Gain Index leaderboard.** A public, living HF-Space leaderboard where *any* advice-giving model (GPT-class, Claude, Qwen-VL, LLaVA, fine-tunes) is benchmarked by AG@k on a fixed pool: advice in → executed → verified gain out. Chatbot-Arena-for-photo-advice, but with objective verification instead of votes. It is simultaneously: paper C1's distribution vehicle, a citation magnet, a B2B lead generator (P6), and a permanent PR engine ("which AI gives the best photo advice? — measured"). *Nothing like it exists for any creative-advice domain.*
- **F4.2 Skill Curriculum ("Duolingo for photography, but measured").** Longitudinal aspect-score tracking on a user's own uploads → skill graph → weekly personalized drills ("your composition +18% in 3 months; lighting flat → this week: backlight exercises, here are 3 of YOUR photos where better light was the fix, with the fixed versions"). Photo-education apps teach generically; nobody coaches from *your measured trajectory with executed counterexamples.* Retention engine + genuinely improves humans (the "highly impact everybody" answer: the tool that makes users *not need it* is the tool they evangelize).
- **F4.3 Aesthetic CI/CD.** A GitHub Action + pre-commit hook: any image asset in a PR (marketing site, app store screenshots, docs, generated-content pipelines) is scored; below-threshold assets block the merge with a critique comment + auto-fix suggestion commit. "Lint your images." Free for OSS (distribution wedge), paid for private repos. Nobody has shipped quality-gated visual CI; it converts AesFix from an app into infrastructure.
- **F4.4 Taste Fingerprint + Taste-Pack marketplace** (§21) — personalization as moat, creators as channel.
- **F4.5 Live shooting coach (mobile, later).** Distilled aspect scorer (MobileCLIP-class) over the viewfinder: real-time composition/lighting nudges *before* the shutter — grounded in the CVPR 2026 actionable-feedback line. The endgame: advice at capture time, verification after.
- **F4.6 MCP server.** Expose `critique(image)`, `fix(image, aspect)`, `verify(before, after)` as MCP tools so agents (Claude, IDE agents, AIGC pipelines) call AesFix natively — the FillScore agent-native-gateway play, ported. Positions AesFix as the aesthetic-QA primitive of the agent ecosystem.


---

# PART VI — COMPLETE TECHNOLOGY CENSUS (everything we might integrate, by layer)

## §26 — AI / ML

| Layer | Technology | Role | Status |
|---|---|---|---|
| Critic VLM | **Qwen2.5-VL-7B** (+3B for cheap tier) | Aspect scores + critique + fix JSON | Core |
| Critic training | **LoRA/QLoRA (PEFT)** · **TRL GRPOTrainer** or **veRL** · bitsandbytes 4-bit | SFT → GRPO edit-verified RL | Core, Phase 3 |
| Legacy scorer | CLIP ViT-B/32 zero-shot (CritiQ `image_model.py`) | Free tier + ablation baseline | Kept |
| Aggregation | **PyTorch differentiable Takagi–Sugeno/ANFIS** (custom, ~300 ln) | Interpretable overall score (C3) | Core |
| Legacy fuzzy | scikit-fuzzy Mamdani (14 rules) | Ablation row only | Frozen |
| Judges | **imscore** (LAION-aes, PickScore, HPSv2, MPS, ImageReward, CLIPScore — differentiable) · **HPSv3** · **Q-Align/OneAlign** · **aesthetic-predictor-v2-5** (SigLIP) · CLIP-IQA | Frozen ensemble: rewards + gain badges | Core |
| Aspect probes | SigLIP linear probes trained on AADB attributes | Falsify per-aspect claims | Core |
| Editors (Tier B) | **Qwen-Image-Edit-2511** (default) · **FLUX.2 klein 9B / FLUX.1 Kontext dev** (photoreal) · Step1X-Edit-v1p2 (reasoning edits, watch) · OmniGen 2 (unified, watch) · Lightning LoRAs (speed) | Semantic executable fixes | Core |
| Editors (Tier A) | **Kornia** (differentiable ops) · OpenCV · Pillow · learned **3D-LUT** bank (FiveK) | Deterministic ms-latency fixes | Core |
| Region & masks | **SAM2** · Grounding-DINO (open-vocab region hints) | Targeted fixes, evidence heatmaps | Core |
| Specialist editors | **IC-Light** (relight) · **IOPaint/LaMa** (remove) · **Real-ESRGAN/GFPGAN/CodeFormer** (restore) | Fix-class coverage | Phase 5 |
| Identity guard | **LPIPS** · **DINOv2** embeddings · InsightFace | Drift gate | Core |
| RAW | **rawpy/LibRaw** · darktable-cli | Pro pipeline | Phase 6 |
| Serving | **vLLM** (VLM) · SGLang / vLLM-Omni (Qwen-Image family) · ONNX Runtime + TensorRT (probes/scorers) · torch.compile | Latency & cost | Phase 5 |
| Distillation (mobile) | MobileCLIP-class student of the aspect probes | F4.5 live coach | Later |
| Personalization | Embedding-space reward head; InstantRetouch-style asymmetric AE ideas | Taste Fingerprint (C4) | Phase 6 |

## §27 — Data & Storage

Datasets: AADB · PCCD · AVA · RPCD · PARA · MIT-Adobe FiveK · PPR10K · GAIC · AesMMIT/AesBench (all §13). Infra: **Postgres (Supabase)** for app data · **S3/Cloudflare R2** for images (R2 = zero egress fees, matters for before/after serving) · **Redis** (queue + cache) · **DVC** or lakeFS for dataset versioning · **Parquet + DuckDB** for eval artifacts (the FillScore student-path lesson) · content-hash dedup on uploads · signed URLs everywhere · auto-delete originals ≤30 days unless opted in.

## §28 — Backend / Frontend

**Backend:** FastAPI (Python 3.12) API + orchestration · Celery or RQ workers over Redis (Tier-A CPU pool; Tier-B GPU pool via Modal/Replicate) · Pydantic v2 strict schemas (the fix-JSON contract) · WebSocket/SSE progress streaming for batch jobs · Stripe (subscriptions + one-off packs + metered API) · Postmark/Resend (progress emails). **Frontend:** Next.js 15 + React Server Components · Tailwind + shadcn/ui · before/after slider (react-compare-slider) · Recharts radar/trends · TanStack Table (batch blotter) · UploadThing/tus resumable uploads · PWA now, Tauri desktop for the local tier later, React Native/Expo for F4.5 later.

## §29 — MLOps & CI/CD

GitHub Actions pipelines: lint (ruff) + type (mypy/pyright) + test (pytest, coverage gate ≥80% on scoring/guard code) → Docker build (multi-stage, CPU & CUDA images) → push GHCR → deploy (Railway/Fly.io for API; Vercel for frontend; Modal deploy for GPU functions) · pre-commit hooks · **eval-as-CI**: every PR touching critic/judges/executor re-runs a 200-image golden set; AG@k regression >2% blocks merge (this is FillScore's eval-harness lesson, institutionalized) · **W&B or MLflow** experiment tracking · HF Hub as model registry (private) · dataset versioning via DVC · k6 load tests before launch · Playwright e2e on the 5 golden user flows · feature flags (PostHog) · canary deploys for judge/critic version bumps · reproducibility pack per paper table (config + seed + commit hash).

## §30 — Observability

Sentry (errors, both stacks) · OpenTelemetry traces API→queue→GPU · **Langfuse** for VLM call traces/cost (the LLM-native observability layer FillScore v4 promoted — same move) · PostHog product analytics (activation: "first verified fix viewed"; retention: weekly upload) · GPU cost telemetry per feature per user (unit-economics dashboard from day one — FillScore's Groq-budget lesson generalized) · uptime probes `/health` `/ready` `/version` · structured logs with correlation IDs.

## §31 — Security, Privacy & Provenance

JWT + OAuth (Google/Apple) via Supabase Auth · signed, expiring image URLs · per-user object prefixes + row-level security · rate limits per tier · **untrusted-input rule:** user critique text is data, never concatenated into system prompts undelimited (FillScore T2.16 prompt-injection lesson, verbatim) · prompt-injection red-team pass before API launch (FillScore F5, ported) · GDPR: export + delete flows, no training on user photos without explicit opt-in, EU data residency option via R2 location hints · NSFW/CSAM screening on upload (open classifiers + provider tooling; mandatory before public launch) · **C2PA signing** of edited outputs + visible "AI-edited preview" watermark on previews (trust + platform-policy compliance) · face-photo handling: biometric-adjacent care — no face embeddings stored beyond the guard check unless user opts into Taste Fingerprint · secrets scanning in CI (gitleaks) — the FillScore hardcoded-secrets P0, pre-empted.

---

# PART VII — GITHUB & RESEARCH TEARDOWNS (what we take from each)

## §32 — Editors

| Repo/model | Take |
|---|---|
| **QwenLM/Qwen-Image** (Qwen-Image-Edit, -2511, -2512, -Layered; Apache-2.0) | Default Tier-B executor. Semantic + appearance edit split matches our fix taxonomy; Layered opens future layer-aware fixes; Lightning LoRA + SGLang/vLLM-Omni for cheap serving; 2–3s/edit reported on standard GPUs |
| **FLUX.2 klein / FLUX.1 Kontext dev** (Black Forest Labs) | Photoreal-critical fallback (product photography fidelity); klein 9B runs on consumer GPUs — candidate for the local tier's generative option |
| **Step1X-Edit-v1p2** | Reasoning-before-editing — watch for executor upgrade; their eval style informs fix-instruction phrasing |
| **OmniGen 2** | Unified gen+edit — if it matures, collapses two pipelines into one |
| **IC-Light** | Relighting = the executable form of most lighting advice |
| **Sanster/IOPaint** (LaMa/MAT) | Distraction-removal endpoint, self-hostable, proven |
| **Real-ESRGAN / GFPGAN / CodeFormer** | Restoration vertical (P4 old photos, P1 low-res listings) |
| **ComfyUI** (headless API mode) | Free orchestration graphs + community workflows; decision vs plain diffusers at AF-2.2 |

## §33 — Scorers / Judges

| Repo | Take |
|---|---|
| **RE-N-Y/imscore** | The judge ensemble in one dependency — differentiable PickScore/HPSv2/MPS/ImageReward/LAION/CLIPScore with benchmark tables; differentiability also unlocks auxiliary training signal |
| **MizzenAI/HPSv3** (ICCV 2025) | Wide-spectrum preference judge (Qwen2-VL-based, pip-installable); their **CoHP** loop (reward-guided iterative refinement) is the closest published cousin of our loop — cite and differentiate |
| **Q-Align/OneAlign** | LMM judge with IQA+IAA+VQA heads + LoRA code; discrete-level scoring trick informs our critic's score elicitation |
| **discus0434/aesthetic-predictor-v2-5** | SigLIP scorer — free-tier + probe backbone |
| **VisualQuality-R1 / Q-Insight** repos | GRPO-for-quality training recipes to adapt, not reinvent |

## §34 — Vision Utilities

**SAM2** (masks/evidence) · **Grounding-DINO** (text→region for "the cluttered background") · **Kornia** (differentiable Tier-A) · **LPIPS / DINOv2 / InsightFace** (guard) · **rawpy** (RAW) · **pyexiv2/exiftool** (EXIF preservation — users hate metadata loss) · **c2pa-python** (provenance).

## §35 — Training Infra

**TRL** (GRPOTrainer; simplest path) vs **veRL** (scales better; more setup) — decide at AF-3.2 by rollout-throughput test · **PEFT/QLoRA** · **Flash-Attention 2** · **W&B** · **Modal** (serverless GPU functions — also serves prod) · RunPod/Lambda/Vast for long training runs · the Tier-A reward precompute trick (§12-C2) keeps GRPO affordable: a parameterized op grid means most rollouts score against cached edits.

## §36 — Research Papers → Feature Map (the "papers that add features" index)

| Paper | Feature it powers |
|---|---|
| Q-Insight / VisualQuality-R1 (2025) | C2 training recipe; reasoning-trace UI in critique card |
| HPSv3 + CoHP (ICCV 2025) | Judge; iterative-refinement pattern for multi-step fixes |
| Q-Align (ICML 2024) | Judge; discrete-level score elicitation in the critic |
| Liu & Wagemans (2026) concept-interpretable IAA | C3 baseline; concept-evidence UI comparison |
| CVPR 2026 actionable memorability feedback | F4.5 live coach design; "advice must be executable" framing |
| Exposure (TOG 2018) / DeepLPF / RSFNet / learned 3D-LUT / VeraRetouch (2026) | Tier-A legitimacy + the named-filter idea (RSFNet names its filters → our fixes carry human names) |
| InstantRetouch (2026) | Taste Fingerprint without test-time fine-tuning |
| PARA (CVPR 2022) + AesBiasBench (EMNLP 2025) | C4 personalization + a bias-audit section (does advice quality differ across subject demographics? — an ethics section reviewers reward and competitors ignore) |
| AesExpert/AesBench/UNIAA | External benchmarks; instruction-data seasoning |
| GAIC (cropping) | Recrop fix ground truth |
| MIT-Adobe FiveK / PPR10K | Expert-edit grounding; 3D-LUT bank training |

---

# PART VIII — PHASED EXECUTION PLAN (AF-x, dependency-ordered, DoD each)

## §37 — Phase 0: Triage & Foundations (Wk 1–2)
- **AF-0.1 Rename & restructure** — `aesfix/{critic,head,executor,judges,data,eval,api}`; name/domain/PyPI check. DoD: CI green on empty test suite; legacy demo runs under `baselines/legacy_fuzzy`.
- **AF-0.2 Fix the routing bug** — explicit `FeaturePayload` dataclass; `main.py`/`pipeline.py` conform. DoD: multimodal demo produces different output with vs without critique text (the original bug's symptom, gone).
- **AF-0.3 Loaders + trusted eval harness** — AADB, PCCD loaders; reproduce ONE published baseline (CLIP linear probe on AADB) within tolerance. DoD: our SRCC matches literature ±0.02.
- **AF-0.4 Judge ensemble install** — imscore + apv2.5 + CLIP-IQA scoring the 5 legacy test images (blur/dark/bright/temp/test.jpg). DoD: sane ordering (blur.jpg lowest technical).
- **AF-0.5 CI/CD skeleton** — GH Actions lint/type/test/docker; pre-commit; gitleaks. DoD: PR gate live.

## §38 — Phase 1: Critic v0 + Head (Wk 3–6)
- **AF-1.1 Prompted critic** — Qwen2.5-VL strict-JSON aspect scoring. DoD: per-aspect SRCC vs AADB attributes reported; beats CLIP tier on ≥4/5 aspects.
- **AF-1.2 ANFIS head** — differentiable TS rules, sparsity reg, train on AADB. DoD: ≥ MLP-head accuracy −1pt with ≤30 printable rules; beats legacy Mamdani.
- **AF-1.3 Faithfulness suite** — intervention monotonicity, deletion/insertion. DoD: table 1 exists; legacy fuzzy included as row.
- **AF-1.4 Gradio v2 internal demo** — radar + rulebook trace. DoD: end-to-end <8s/image on dev box.

## §39 — Phase 2: Executor + Engine + AG@k (Wk 7–10) — the timestamp phase
- **AF-2.1 Tier-A executor** — Kornia op grid + JSON op schema + XMP writer. DoD: 12 ops round-trip; XMP opens in Lightroom.
- **AF-2.2 Tier-B executor decision** — Qwen-Image-Edit-2511 via diffusers vs ComfyUI-headless latency/cost test on Modal. DoD: ≤$0.01 and ≤6s per edit at 1MP; identity guard wired.
- **AF-2.3 Synthetic Counterfactual Engine** — 50K triplets, config-driven. DoD: AG@1 with oracle advice recovers ≥90% of known degradation on the synthetic set (metric sanity).
- **AF-2.4 SAM2 region path** — masked Tier-A ops. DoD: "brighten subject" demo.
- **AF-2.5 AG@k harness** — full protocol §12-C1. DoD: reproducible run on 1K images, seeds pinned.
- **AF-2.6 Frontier-model benchmark** — GPT-class + open VLMs advice → AG@k. DoD: the comparison table (the expected negative result documented).
- **AF-2.7 Human-agreement mini-study** — 100 pairs, 3 raters. DoD: judge-gain ↔ human-preference correlation reported.
- **AF-2.8 arXiv short** — C1 + benchmark + engine release. DoD: preprint live (the idea is timestamped).
- **AF-2.9 AGX leaderboard v0** — HF Space, fixed pool, submission protocol. DoD: 5 models on the board, methodology page versioned.

## §40 — Phase 3: Training (Wk 11–15)
- **AF-3.1 SFT** — LoRA on PCCD + synthetic triplets. DoD: schema-valid ≥99%; aspect SRCC +≥5pt vs prompted.
- **AF-3.2 GRPO infra decision** — TRL vs veRL throughput test. DoD: ≥N rollouts/hr on 2×A100 budget plan.
- **AF-3.3 Edit-verified GRPO** — reward §12-C2, Tier-A-precompute first, Tier-B budgeted. DoD: AG@3 beats SFT by significant margin on held-out; hacking dashboards clean (guard-violation rate, advice diversity, judge variance).
- **AF-3.4 Ablation grid** — §14 rows. DoD: every paper table reproducible from configs.

## §41 — Phase 4: Paper + AGX season 1 (Wk 16–20)
- **AF-4.1 Full human study** (§13). **AF-4.2 Paper draft** (venue ladder §15). **AF-4.3 Code/dataset/model release** (HF + GH, licenses checked — AVA/RPCD redistribution rules respected: release scripts + IDs, not images). **AF-4.4 AGX season 1 launch** with the trained critic on top of the board (marketing = benchmark). DoD: submission uploaded; leaderboard has external submissions.

## §42 — Phase 5: SaaS Launch (Wk 18–24, overlaps)
- **AF-5.1 Auth + billing + tiers** (Supabase, Stripe). **AF-5.2 Upload→critique→fix flow** (F1.1–1.3) with queue + progress SSE. **AF-5.3 Batch + goal modes** (F1.4–1.5). **AF-5.4 Trust layer** — watermarks, C2PA, delete flows, NSFW gate. **AF-5.5 Observability + cost dashboards** (§30). **AF-5.6 Beta** — 20 users from photo/Etsy/dating communities; instrument activation. **AF-5.7 Launch** — Product Hunt + r/photocritique + IndieHackers + the arXiv paper + AGX as credibility artillery. DoD: first paying user; COGS/user within §10 envelope. Phase 6 (moonshots F4.x, MCP, mobile, marketplace) is gated on launch learnings.

---

# PART IX — SAAS LAUNCH & GTM

## §43 — Launch Sequence
Waitlist page during Phase 3 (before/after GIFs from the synthetic engine are launch-content for free) → private beta (AF-5.6) → public launch stacked with the paper + AGX ("we didn't build another photo app; we built the first AI whose photo advice is *measured* — here's the leaderboard, here's the paper, here's your photo fixed").

## §44 — Growth Loops
(1) Share-card loop (F2.2, before/after + gain badge); (2) AGX loop (every model comparison article links us); (3) XMP loop (Lightroom users evangelize tools that respect their workflow); (4) Duel/WhatIf data flywheel → better personal judges → visibly better fixes; (5) Aesthetic CI/CD OSS wedge → B2B API; (6) SEO programmatic pages per fix-class ("fix underexposed listing photo") with live demo embeds.

## §45 — Cost Model
Free tier ≈ $0.001/photo (CPU SigLIP + cached CLIP) · Pro photo with 3 previews ≈ 1 Tier-B edit ($0.002–0.01 serverless) + Tier-A (≈0) + judges (≈$0.0005) → gross margin >85% at $12/mo·300 photos · training one-off: SFT ~$50–150, GRPO ~$200–500 (Tier-A precompute keeps rollouts cheap) · self-host switch at ~1k Tier-B edits/day (deployment-guide breakeven), i.e., ~350 daily-active Pro users — a good problem.

---

# PART X — OPERATING MANUAL

## §46 — Risks & Kill-Switches
| Risk | Guard |
|---|---|
| Reward hacking | §20 (frozen disjoint judges, trimmed mean, identity gate, diversity penalty, single-shot eval) + no-guard ablation as evidence |
| Editor destroys identity/trust | Guard threshold strict by default; previews watermarked; original always one click away; C2PA |
| Scooped on C1 | AF-2.8 arXiv timestamp is the whole point of Phase 2's ordering |
| Compute budget dies | Paper stands on C1+C3 (no training needed); product ships on prompted critic |
| Judge bias/staleness | Versioned judges, AGX seasons, human-agreement study validates the metric itself |
| Aesthetic subjectivity attacks | Distributions not points; uncertainty badges; C4 personalization; PARA/AesBiasBench-grounded bias audit |
| Adobe patent adjacency | We verify + train on gains (unpatented territory) vs suggest-only patents; counsel review before B2B scale |
| Photo privacy/regulatory | §31 wholesale; local tier as structural answer |
| Dataset licenses | Release scripts+IDs not images; AVA/RPCD terms respected |
| Solo-founder scope creep | Phase gates; moonshots locked behind launch; this document is the scope fence |

## §47 — Product-Integrity Rules (never drift)
1. Never present an edited preview as the user's original; 2. Never credit gain past the identity guard; 3. Judges never share weights with the critic; 4. User photos never train anything without explicit opt-in; 5. Advice is coaching, not guaranteed outcomes — no "this WILL increase sales" claims; 6. The legacy CritiQ baselines are never cited; 7. AGX methodology changes only between seasons, publicly.

## §48 — Resume & Interview Playbook
Headline: *"Built AesFix — the first AI photo critic whose advice is executed and verified; introduced Aesthetic Gain, a behavioral benchmark for visual advice (arXiv/[venue]); trained a GRPO edit-verified VLM critic; shipped it as a SaaS with a public leaderboard."* STAR stories banked: (1) the pivot decision (found the interpretability claim published mid-project → repositioned around verification — judgment story); (2) reward hacking caught by the guard dashboard (evidence-driven engineering); (3) the metric-validation study (does the judge agree with humans — scientific humility); (4) Tier-A reward precompute (10× RL cost reduction — systems cleverness); (5) XMP export (user-empathy beats feature-count). Skills demonstrated: VLMs, RL (GRPO), diffusion editors, interpretable ML, benchmark design, MLOps/CI-CD, unit economics, GTM.

## §49 — Credibility Ledger
Claims we can defend on request: every paper number ↔ config+seed+commit; AG@k human-validated; ensemble compositions pinned; COGS from telemetry not guesses; the CritiQ→AesFix pivot documented honestly in §3/§5 (interviewers respect the kill decision more than the demo).

## §50 — Immediate Next Actions (this week)
1. AF-0.1 rename/restructure + CI skeleton; 2. AF-0.2 bug fix (30 min, closes the CritiQ chapter with dignity); 3. AF-0.3 AADB+PCCD loaders + reproduce one published number; 4. AF-0.4 pip-install the judge ensemble and score the five legacy test images; 5. AF-1.1 prompt Qwen2.5-VL with the fix-JSON schema on those same images — the moment the first structured, executable critique prints, AesFix exists.

---

*End of AesFix Unified Master Roadmap v5. Supersedes `New Context.md`, `Research Text.md`, and `AESFIX_ROADMAP.md`. Next consolidation (v6) after Phase 2 exit.*

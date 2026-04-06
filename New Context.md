================================================================================
CritiQ — HYBRID NEURO-FUZZY MULTIMODAL FRAMEWORK
FOR EXPLAINABLE IMAGE AESTHETIC EVALUATION
COMPLETE PROJECT CONTEXT DOCUMENT (UPDATED — POST CLIP FIX)
Generated: 2026-04-06
Purpose: Single source of truth for continuation, research publication,
and LLM handoff
===============

================================================================================
SECTION 1 — PROJECT OVERVIEW
============================

PROJECT TITLE:
CritiQ — A Hybrid Neuro-Fuzzy and Multimodal Framework for Explainable
Image Aesthetic Evaluation

PROBLEM STATEMENT:
Image aesthetic evaluation is inherently subjective, uncertain, and
multi-dimensional. Existing deep learning approaches such as NIMA,
AestheticViT, and ViT-based regressors suffer from:

```
1. Black-box decision making (no interpretability)
2. Only numeric outputs (no explanations)
3. Lack of uncertainty modeling
4. No actionable feedback for users
5. Heavy dependency on labeled datasets and GPU training
6. Poor handling of multi-modal inputs (text + image)
```

PURPOSE AND MOTIVATION:
The goal of CritiQ is to build a fully explainable, lightweight, and
multimodal aesthetic evaluation system that:

```
- Combines human critique understanding (text)
- Uses vision intelligence (CLIP)
- Models uncertainty (fuzzy logic)
- Generates explanations and improvements (XAI)
- Works without training or APIs (zero-cost inference)
```

EXPECTED FINAL OUTCOME:
A complete system that:

```
1. Accepts image and/or critique text
2. Extracts 5 aesthetic dimensions:
   - composition
   - lighting
   - color
   - subject_clarity
   - technical quality
3. Uses zero-shot CLIP for image understanding
4. Uses rule-based NLP for critique understanding
5. Fuses both modalities intelligently
6. Evaluates using Mamdani fuzzy logic
7. Outputs:
   - Aesthetic score
   - Label (HIGH / MEDIUM / LOW)
   - Explanation
   - Counterfactual improvements
   - Subjectivity metrics
```

UNIQUE CONTRIBUTION:
This project is novel because it combines:

```
✔ Zero-shot CLIP aesthetic decomposition
✔ Rule-based aspect-level NLP
✔ Fuzzy logic for uncertainty modeling
✔ Explainable AI (template + counterfactual)
✔ Multi-critic aggregation + disagreement scoring
✔ Zero API + zero training requirement
```

This combination is **publishable-level novelty**.

================================================================================
SECTION 2 — CURRENT APPROACH AND ARCHITECTURE
=============================================

OVERALL SYSTEM DESIGN:

INPUT → FEATURE EXTRACTION → FUSION → FUZZY INFERENCE → EXPLANATION

PIPELINE:

+------------------+
| Input Layer      |
| (Image/Text)     |
+--------+---------+
|
+------+------+
|             |
v             v
+-----------+   +----------------+
| NLP Model |   | CLIP Model     |
| Rule-based|   | Zero-shot      |
+-----------+   +----------------+
|             |
+------+------+
|
FEATURE FUSION
|
AGGREGATION
|
FUZZY SYSTEM
|
+---------+----------+
| Explanation + XAI  |
+--------------------+

---

KEY TECHNOLOGIES:

1. Rule-Based NLP

   * Keyword matching
   * Aspect scoring (0–1)
   * Cached results

2. CLIP (ViT-B/32)

   * Zero-shot aesthetic scoring
   * Uses text-image similarity

3. Mamdani Fuzzy Inference System

   * 5 inputs → 1 output
   * 14 rules
   * Centroid defuzzification

4. Aggregator

   * Multi-critic fusion
   * Confidence weighting
   * Disagreement scoring

5. Explainability Layer

   * Template explanations
   * Counterfactual suggestions

---

DATA FLOW:

Step 1: Input

* Image and/or critique text

Step 2: Feature Extraction

* Text → NLP scores
* Image → CLIP scores

Step 3: Fusion
IF text present:
0.85 * text + 0.15 * image
ELSE:
image only (CRITICAL FIX APPLIED)

Step 4: Aggregation

* Weighted averaging
* Disagreement computation

Step 5: Fuzzy Inference

* Apply rules
* Output score + label

Step 6: Explanation + Counterfactual

---

KEY TECHNICAL DECISIONS:

1. Zero-shot CLIP instead of training
2. Mamdani fuzzy (interpretability > precision)
3. Template NLG (consistent, no API cost)
4. Confidence-weighted aggregation
5. Modular pipeline architecture

================================================================================
SECTION 3 — IMPLEMENTED COMPONENTS (DONE)
=========================================

[1] FIXED CLIP IMAGE MODEL (CRITICAL UPGRADE)

NEW IMPLEMENTATION:

* Uses:
  _model.vision_model()
  _model.text_model()

* Extracts:
  pooler_output (CLS token)

* Applies:
  visual_projection
  text_projection

* Computes:
  cosine similarity + softmax

RESULT:
✔ Fully working zero-shot aesthetic scoring
✔ No more fallback to 0.5
✔ Semantically meaningful outputs

---

[2] FUSION BUG FIXED

OLD:
round(...) → integer truncation

NEW:
round(..., 4)

RESULT:
✔ Continuous values preserved

---

[3] IMAGE-ONLY MODE FIXED

OLD:
0.85 * 0.5 + 0.15 * image → always ~0.5

NEW:
If no text → use image only

RESULT:
✔ Correct behavior in image mode

---

[4] NLP EXTRACTOR

* Keyword-based scoring
* MD5 caching
* Sentiment estimation
* 5 dimensions extracted

---

[5] FUZZY SYSTEM

* 5 inputs, 1 output
* 14 rules
* Triangular membership functions

---

[6] AGGREGATOR

* Confidence-weighted averaging
* Cosine disagreement
* Subjectivity labeling

---

[7] EXPLAINER

* Template-based NLG
* Uses strongest + weakest dimensions

---

[8] COUNTERFACTUAL GENERATOR

* Suggests improvements
* Targets weakest features

---

[9] GRADIO DASHBOARD

* Image upload
* Feature visualization
* Score display

---

[10] PIPELINE

* Batch processing
* Dataset support
* JSON output

================================================================================
SECTION 4 — COMPONENTS IN PROGRESS (DOING)
==========================================

[1] NLP INTELLIGENCE (MAJOR LIMITATION)

Current issue:

* "bad composition" → still medium score
* No negation handling
* No context understanding

Status:
Needs upgrade to aspect-aware parsing

---

[2] REAL DATASET

* Currently using synthetic dataset
* RPCD dataset not integrated yet

---

[3] CLIP PROMPT OPTIMIZATION

* Prompts are static
* Can be improved for better alignment

================================================================================
SECTION 5 — PLANNED COMPONENTS (NEXT LEVEL RESEARCH)
====================================================

[1] NEGATION-AWARE NLP

* Detect "not sharp", "not good"
* Reverse sentiment

[2] ASPECT-LEVEL PARSING

* Map adjectives to dimensions
* Use dependency parsing

[3] ADAPTIVE FUSION

* Dynamic weights per dimension

[4] COUNTERFACTUAL XAI

* Minimal changes to reach HIGH score

[5] RULE OPTIMIZATION

* Genetic algorithm / PSO for fuzzy tuning

[6] ABLATION STUDY

* Text-only vs Image-only vs Hybrid

[7] ADVANCED UI

* Radar charts
* Comparison mode

================================================================================
SECTION 6 — FILE STRUCTURE
==========================

CritiQ/
│
├── main.py                → CLI entry point
├── app.py                 → Gradio UI
├── our_system/
│   ├── llm_extractor.py   → NLP
│   ├── image_model.py     → CLIP (FIXED)
│   ├── fuzzy_system.py    → FIS
│   ├── aggregator.py      → Aggregation
│   ├── explainer.py       → Explanation
│   ├── counterfactual_generator.py
│   ├── pipeline.py        → Batch runner
│   ├── evaluate.py        → Metrics
│
├── data/
├── results/
├── cache/

---

EXECUTION FLOW:

main.py → pipeline/app → modules → output

================================================================================
SECTION 7 — MODELS AND DEPENDENCIES
===================================

MODELS:

* CLIP ViT-B/32 (HuggingFace)
* Rule-based NLP
* Fuzzy system (scikit-fuzzy)

DEPENDENCIES:

* torch
* transformers
* numpy
* scipy
* Pillow
* gradio

DATA:

* Synthetic dataset (current)
* RPCD dataset (planned)

================================================================================
SECTION 8 — EXECUTION INSTRUCTIONS
==================================

SETUP:

pip install -e .

RUN:

# Single image

python main.py --mode single --image test.jpg

# Single critique

python main.py --mode single --critique "text"

# Batch

python main.py --mode run

# UI

python app.py

================================================================================
SECTION 9 — CURRENT STATUS SUMMARY
==================================

COMPLETION: ~90%

WORKING:

✔ CLIP zero-shot extraction
✔ Fuzzy inference
✔ NLP extraction (basic)
✔ Fusion logic
✔ Explanation + counterfactual
✔ UI + pipeline

ISSUES:

⚠ NLP lacks intelligence
⚠ No real dataset evaluation

================================================================================
SECTION 10 — NEXT EXACT STEPS
=============================

STEP 1:
Implement negation-aware NLP

STEP 2:
Improve aspect mapping

STEP 3:
Run full evaluation

STEP 4:
Perform ablation study

STEP 5:
Write research paper

================================================================================
SECTION 11 — PROJECT CONTEXT FOR LLM CONTINUATION
=================================================

ASSUMPTIONS:

* System must remain zero-API
* Must be explainable
* Must be lightweight

CONSTRAINTS:

* 5 fixed feature keys
* Output range [0,1]
* Modular architecture

CRITICAL NOTES:

* CLIP now uses vision_model + text_model
* Fusion logic differs for image-only mode
* Fuzzy system is core evaluator
* NLP must be improved next

WHAT NEXT LLM MUST DO:

* Upgrade NLP (priority)
* Optimize prompts
* Improve evaluation

================================================================================
END OF DOCUMENT
===============

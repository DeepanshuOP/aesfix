# AesFix Implementation & Validation Status

Per the Roadmap v6 Integrity Protocol (Part XI), this document tracks the codebase at the file level across three strict verification categories. No performance metrics are claimed or reported without an associated run entry in `results/ledger.jsonl`.

---

## File-Level Status Matrix

| Real & Verified (Tested & Truthful) | Implemented, Unvalidated (Awaiting Weights / Data / Study) | Not Built (Planned in Roadmap v6) |
| :--- | :--- | :--- |
| `aesfix/ledger.py`<br>Evidence ledger with SHA-256 provenance, UUID4 run tags, and JSONL appending. Tested in `tests/test_ledger.py`. | `aesfix/critic/critic.py`<br>Pydantic schemas and critic prompt generation. Awaiting real Qwen3-VL serving backend. | `aesfix/certifier/ltt.py`<br>Learn-Then-Test (LTT) risk-controlling calibration procedure (Part IV.4). |
| `aesfix/eval/metrics.py`<br>Certified signed AG_best@k, AG_comp@k, clipped variants, HWR (ties=0.5), harm rate. Tested with hand-computed toys and Hypothesis property tests. | `aesfix/judges/ensemble.py`<br>Multi-judge normalization and trimmed ensemble logic. Awaiting real frozen judge weights (pyiqa, imscore, HPSv3). | `aesfix/certifier/sgr.py`<br>Selective Gain Risk (SGR) statistical certification bounds. |
| `aesfix/eval/ag_benchmark.py`<br>Behavioral benchmark harness evaluating candidate and composed fixes using certified signed metrics. | `aesfix/judges/taste.py`<br>Online Bayesian Bradley–Terry preference module. Awaiting live user interaction telemetry. | `aesfix/probes/aspect_probes.py`<br>SigLIP linear probes for 5-aspect falsification (TAG / COL) calibrated on AADB & SPAQ. |
| `scripts/check_claims.py`<br>Automated integrity checker enforcing that all metric claims in `docs/` and `paper/` link to valid ledger runs. | `aesfix/eval/frontier_benchmark.py`<br>Runner for external model advice JSON. Awaiting real model inference files. | `aesfix/eval/goodhart_map.py`<br>Judge exploitability index and hack zone analysis across candidate edits (Part IV.5). |
| `scripts/make_tables.py`<br>CLI tool generating Markdown tables directly from verified ledger entries. | `aesfix/eval/ablation_grid.py`<br>Component ablation harness. Awaiting real model and judge executions. | `aesfix/executor/guard_v2.py`<br>Identity Guard v2 incorporating LPIPS (AlexNet), DINOv2 cosine similarity, and over-processing checks (Part IV.7). |
| `aesfix/executor/tier_a.py`<br>12 deterministic parametric image adjustments in Kornia/OpenCV and Adobe Lightroom `.xmp` exporter. | `aesfix/eval/aesthetic_ci.py`<br>Aesthetic regression CI checks. Awaiting calibrated baseline scores. | `aesfix/c2pa/signer.py`<br>C2PA digital provenance credential assertion generator. |
| `aesfix/executor/guard.py`<br>Deterministic identity guard with SSIM and RGB layout statistics. | `aesfix/eval/curriculum.py`<br>Aesthetic learning trajectory engine. Awaiting human trial validation. | `aesfix/data/manifest.py`<br>Data pool manifest generator verifying checksums and licenses (scaffolded in R1.1). |
| `aesfix/eval/leaderboard.py`<br>Clean leaderboard loader reading exclusively from `results/ledger.jsonl`. | `aesfix/eval/faithfulness.py`<br>Deletion/insertion curves for attribution. Awaiting model weights. | `aesfix/judges/registry.py`<br>Judge environment registry with version pins and install diagnostics (scaffolded in R1.2). |
| `aesfix/eval/paper_export.py`<br>Clean research table export engine rendering strictly from ledger records. | `aesfix/eval/full_human_study.py`<br>Simulation cohort testing harness. Awaiting live Prolific crowd study data. | `aesfix/executor/tier_b.py`<br>Generative inpainting and regional generative editing (deferred to R7). |
| `aesfix/head/anfis.py`<br>Differentiable Takagi–Sugeno neuro-fuzzy inference network forward pass and rule activation. | `aesfix/eval/human_study.py`<br>Synthetic pair simulation generator. Awaiting live human annotations. | |
| `web/src/app/*`<br>Next.js 15 workbench UI, interactive before/after split slider, and inspection controls. | `aesfix/data/dataset.py`<br>Dataset loader interfaces. Awaiting verified local image mirrors. | |
| `web/src/components/*`<br>Modular UI components adhering to Tailwind and Lucide icon standards. | `aesfix/data/synthetic.py`<br>Synthetic degradation pipelines for test fixtures. | |
| `tests/*`<br>Automated test suite verifying deterministic invariants without asserting on fabricated numbers. | `aesfix/api/app.py`<br>FastAPI web service routes. Awaiting live model execution pipelines. | |
| | `aesfix/api/mcp_server.py`<br>Model Context Protocol bridge. Awaiting connected live backends. | |
| | `aesfix/train/grpo.py`<br>Group Relative Policy Optimization loop. Awaiting GPU training runs. | |
| | `aesfix/train/reward.py`<br>RL reward functions. Awaiting verified judge feedback loop. | |
| | `aesfix/video/temporal.py`<br>Temporal video aesthetic consistency module. | |

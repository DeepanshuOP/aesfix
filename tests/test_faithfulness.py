"""Tests for Faithfulness Suite and Table 1 Benchmarking (AF-1.3).

Why these tests verify the scientific interpretability layer:
1. Verifies monotonic compliance rate calculation on controlled functions (strictly
   increasing yields 100%, strictly decreasing yields 0%).
2. Validates Deletion and Insertion AUC calculations within theoretical [0, 1] bounds.
3. Fulfills roadmap §38 DoD: verifies Paper Table 1 is generated comparing ANFIS
   directly against the legacy Mamdani fuzzy baseline under identical protocols.
"""
from aesfix.eval import (
    FaithfulnessMetrics,
    benchmark_table1_interpretability,
    evaluate_deletion_insertion,
    evaluate_monotonic_compliance,
)
from aesfix.head import ANFISHead


def test_monotonic_compliance_rate_logic():
    # Monotonic function: sum of inputs
    def _mono_inc(x):
        return sum(x) / len(x)

    samples = [
        [0.2, 0.4, 0.5, 0.6, 0.3],
        [0.5, 0.5, 0.5, 0.5, 0.5],
    ]
    rate = evaluate_monotonic_compliance(_mono_inc, samples, delta=0.2)
    assert rate == 100.0

    # Inverted function: negative sum
    def _mono_dec(x):
        return 1.0 - (sum(x) / len(x))

    inv_rate = evaluate_monotonic_compliance(_mono_dec, samples, delta=0.2)
    assert inv_rate == 0.0


def test_deletion_insertion_auc_bounds():
    def _linear_model(x):
        return sum(x) / len(x)

    samples = [
        [0.8, 0.2, 0.7, 0.3, 0.9],
        [0.1, 0.9, 0.2, 0.8, 0.5],
    ]

    del_auc, ins_auc = evaluate_deletion_insertion(_linear_model, samples, baseline_value=0.5)

    assert 0.0 <= del_auc <= 1.0
    assert 0.0 <= ins_auc <= 1.0


def test_af13_table1_interpretability_dod():
    """DoD: Paper Table 1 exists, benchmarks ANFIS, and includes legacy fuzzy as an ablation row."""
    head = ANFISHead(num_inputs=5, num_rules=20, monotonic=True)

    result = benchmark_table1_interpretability(head)

    # 1. Structure check
    assert "metrics" in result
    assert "markdown_table" in result
    metrics = result["metrics"]
    assert len(metrics) == 3

    # 2. Verify all models are present (DoD mandates legacy fuzzy row)
    model_names = [m.model_name for m in metrics]
    assert "ANFIS Head (C3, Ours)" in model_names
    assert "Legacy Mamdani (CritiQ)" in model_names
    assert "Linear Baseline" in model_names

    # 3. Check metric validity
    for m in metrics:
        assert isinstance(m, FaithfulnessMetrics)
        assert 0.0 <= m.monotonic_compliance_rate <= 100.0
        assert 0.0 <= m.deletion_auc <= 1.0
        assert 0.0 <= m.insertion_auc <= 1.0

    # 4. Check Markdown Table 1 formatting
    table_str = result["markdown_table"]
    assert "| Architecture | Rules | Monotonic Compliance |" in table_str
    assert "Legacy Mamdani (CritiQ)" in table_str
    assert "ANFIS Head (C3, Ours)" in table_str

    # 5. ANFIS monotonic compliance must be high due to non-negative softplus weights
    anfis_metric = next(m for m in metrics if "ANFIS" in m.model_name)
    assert anfis_metric.monotonic_compliance_rate >= 90.0

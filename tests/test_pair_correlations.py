"""Unit tests for pairwise Spearman sanity check and margin distribution analysis (I4)."""
from pathlib import Path

from scripts.recompute_pair_correlations import recompute_pair_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_recompute_pair_metrics():
    pilot_path = REPO_ROOT / "results" / "pilot_pair_scores.json"
    assert pilot_path.exists(), "results/pilot_pair_scores.json must exist"

    res = recompute_pair_metrics(pilot_path)

    assert res["n_pairs"] == 300
    assert "spearman" in res["correlations"]
    assert "kendall_tau" in res["correlations"]
    assert "pearson_r" in res["correlations"]

    # Contingency table sums to N
    table = res["contingency_table_2x2"]
    total_table = (
        table["x_gt_0_y_gt_0_both_edit"]
        + table["x_gt_0_y_le_0_judge_edit_human_orig"]
        + table["x_le_0_y_gt_0_judge_orig_human_edit"]
        + table["x_le_0_y_le_0_both_orig"]
    )
    assert total_table == 300
    assert 0.0 <= table["sign_agreement_accuracy"] <= 1.0

    # Margin stratification is monotonic in threshold
    strat = res["stratified_by_margin"]
    assert len(strat) >= 4
    for i in range(len(strat) - 1):
        assert strat[i]["count"] >= strat[i + 1]["count"]

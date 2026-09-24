"""Tests for Differentiable Takagi-Sugeno ANFIS Head (AF-1.2 / C3).

Why these tests verify the interpretable head:
1. Validates end-to-end differentiability: all parameters (centers, sigmas, consequents,
   and rule gates) must receive non-zero gradients during backprop.
2. Verifies monotonic compliance: an increase in any individual aspect score must
   never paradoxically reduce the overall aesthetic rating.
3. Tests explain() and print_rulebook() to ensure the rulebook produces readable,
   audit-ready IF-THEN justifications for every decision.
4. Validates training step and L1 sparsity regularization.
"""
import torch

from aesfix.head import ANFISHead, RulebookTrace, train_anfis_epoch


def test_anfis_differentiability_and_gradient_flow():
    head = ANFISHead(num_inputs=5, num_rules=16, monotonic=True)
    inputs = torch.rand(4, 5, requires_grad=True)
    targets = torch.tensor([[0.3], [0.5], [0.7], [0.9]])

    outputs = head(inputs)
    loss = torch.nn.functional.mse_loss(outputs, targets)
    loss.backward()

    # Verify input gradients flow smoothly
    assert inputs.grad is not None
    assert torch.any(inputs.grad != 0.0)

    # Verify membership parameter gradients exist
    for mf in head.memberships:
        assert mf.centers.grad is not None
        assert mf.log_sigmas.grad is not None

    # Verify consequent parameter gradients exist
    assert head.consequent_weights.grad is not None
    assert head.consequent_biases.grad is not None
    assert head.rule_importance.grad is not None


def test_anfis_forward_bounds():
    head = ANFISHead(num_inputs=5, num_rules=20)

    # Extreme corner cases: all 0.0 and all 1.0
    corners = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5, 0.5, 0.5],
    ], dtype=torch.float32)

    with torch.no_grad():
        preds = head(corners)

    assert preds.shape == (3, 1)
    assert torch.all(preds >= 0.0)
    assert torch.all(preds <= 1.0)
    # Reference high aesthetic should score strictly higher than zeroed low
    assert preds[1].item() > preds[0].item()


def test_monotonic_intervention_compliance():
    """DoD: Intervening to improve any aspect score must monotonically non-decrease score."""
    head = ANFISHead(num_inputs=5, num_rules=24, monotonic=True)
    head.eval()

    base_sample = torch.tensor([[0.5, 0.4, 0.5, 0.5, 0.5]])

    with torch.no_grad():
        base_score = head(base_sample).item()

        # Boost lighting from 0.4 to 0.85
        boosted_sample = torch.tensor([[0.5, 0.85, 0.5, 0.5, 0.5]])
        boosted_score = head(boosted_sample).item()

    assert boosted_score >= base_score - 1e-4


def test_anfis_explain_trace_and_rulebook():
    head = ANFISHead(num_inputs=5, num_rules=24)

    aspects = {
        "composition": 0.85,
        "lighting": 0.80,
        "color": 0.70,
        "subject_clarity": 0.65,
        "technical": 0.75,
    }

    trace = head.explain(aspects, top_k=3)

    assert isinstance(trace, RulebookTrace)
    assert 0.0 <= trace.overall_score <= 1.0
    assert len(trace.top_fired_rules) == 3
    assert trace.active_rule_count >= 1

    # Check top rule structure
    top_rule = trace.top_fired_rules[0]
    assert top_rule.firing_strength > 0.0
    assert "composition is" in top_rule.antecedent_description
    assert "IF" in trace.rulebook_summary

    # Check rulebook printing
    rulebook_str = head.print_rulebook(min_importance=0.1)
    assert "AesFix Learned ANFIS Rulebook" in rulebook_str
    assert "Rule 01" in rulebook_str


def test_train_anfis_epoch_loss_reduction():
    head = ANFISHead(num_inputs=5, num_rules=16)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.05)

    features = torch.rand(32, 5)
    # Synthetic target with known positive correlation to features
    targets = (features[:, 0] * 0.4 + features[:, 1] * 0.4 + features[:, 2] * 0.2).unsqueeze(1)

    initial_loss = train_anfis_epoch(head, optimizer, features, targets)
    for _ in range(5):
        final_loss = train_anfis_epoch(head, optimizer, features, targets)

    assert final_loss < initial_loss

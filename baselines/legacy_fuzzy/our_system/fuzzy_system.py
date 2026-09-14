import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def build_fuzzy_system():
    # ── Input variables ──────────────────────────────────────
    composition     = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'composition')
    lighting        = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'lighting')
    color           = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'color')
    subject_clarity = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'subject_clarity')
    technical       = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'technical')

    # ── Output variable ──────────────────────────────────────
    aesthetic = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'aesthetic')

    # ── Membership functions (triangular) ────────────────────
    for var in [composition, lighting, color, subject_clarity, technical, aesthetic]:
        var['low']    = fuzz.trimf(var.universe, [0.0, 0.0, 0.5])
        var['medium'] = fuzz.trimf(var.universe, [0.0, 0.5, 1.0])
        var['high']   = fuzz.trimf(var.universe, [0.5, 1.0, 1.0])

    # ── Rule base (Mamdani) ──────────────────────────────────
    rules = [
        # Strong positive signals
        ctrl.Rule(composition['high'] & lighting['high'],
                  aesthetic['high']),
        ctrl.Rule(composition['high'] & lighting['high'] & color['high'],
                  aesthetic['high']),
        ctrl.Rule(composition['high'] & subject_clarity['high'],
                  aesthetic['high']),

        # Mixed positive
        ctrl.Rule(composition['high'] & lighting['medium'],
                  aesthetic['high']),
        ctrl.Rule(composition['medium'] & lighting['high'] & color['high'],
                  aesthetic['high']),
        ctrl.Rule(composition['medium'] & color['high'] & subject_clarity['high'],
                  aesthetic['medium']),

        # Neutral / medium
        ctrl.Rule(composition['medium'] & lighting['medium'],
                  aesthetic['medium']),
        ctrl.Rule(composition['medium'] & lighting['medium'] & color['medium'],
                  aesthetic['medium']),
        ctrl.Rule(composition['high'] & lighting['low'],
                  aesthetic['medium']),
        ctrl.Rule(composition['low'] & lighting['high'] & color['high'],
                  aesthetic['medium']),

        # Negative signals
        ctrl.Rule(composition['low'],
                  aesthetic['low']),
        ctrl.Rule(lighting['low'] & color['low'],
                  aesthetic['low']),
        ctrl.Rule(composition['low'] & subject_clarity['low'],
                  aesthetic['low']),
        ctrl.Rule(technical['low'] & lighting['low'],
                  aesthetic['low']),
    ]

    system = ctrl.ControlSystem(rules)
    sim    = ctrl.ControlSystemSimulation(system)
    return sim


def evaluate_aesthetics(sim, composition, lighting, color,
                         subject_clarity, technical):
    """
    Takes 5 feature scores (0-1 each), returns (label, confidence).
    """
    # Clamp inputs to valid range
    def clamp(v):
        return float(max(0.01, min(0.99, v)))

    sim.input['composition']     = clamp(composition)
    sim.input['lighting']        = clamp(lighting)
    sim.input['color']           = clamp(color)
    sim.input['subject_clarity'] = clamp(subject_clarity)
    sim.input['technical']       = clamp(technical)

    sim.compute()

    score = round(sim.output['aesthetic'], 4)

    if score >= 0.65:
        label = "HIGH"
    elif score >= 0.40:
        label = "MEDIUM"
    else:
        label = "LOW"

    return label, score


# ── Quick self-test ───────────────────────────────────────────
if __name__ == "__main__":
    print("Building fuzzy system...")
    sim = build_fuzzy_system()

    tests = [
        ("All HIGH",   1.0, 1.0, 1.0, 1.0, 1.0),
        ("All LOW",    0.0, 0.0, 0.0, 0.0, 0.0),
        ("All MEDIUM", 0.5, 0.5, 0.5, 0.5, 0.5),
        ("Good comp, bad light", 0.9, 0.1, 0.5, 0.7, 0.6),
        ("Mixed",      0.6, 0.7, 0.3, 0.5, 0.8),
    ]

    print("\n{:<25} {:<10} {:<10}".format("Test", "Label", "Score"))
    print("-" * 45)
    for name, c, l, co, s, t in tests:
        label, score = evaluate_aesthetics(sim, c, l, co, s, t)
        print("{:<25} {:<10} {:<10}".format(name, label, score))
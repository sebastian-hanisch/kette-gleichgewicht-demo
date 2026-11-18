"""Die Irrfahrt auf dem Kreis mit fünf Zuständen und Drift p (Levin/Peres/Wilmer, 2. Auflage, Beispiel 1.22).

Vorwärts mit Wahrscheinlichkeit p, rückwärts mit 1 − p; die Gleichverteilung π = 1/5 ist stationär, Detailed Balance gilt nur für p = ½. Im Hard-Core-Ring mit
fünf Plätzen ist das die Bahn der fünf Einzelnen unter der Drehung um einen Platz. Zufall nur über kgw_rng (SplitMix64, vektorisiert in kgw_simulation).
"""

import numpy as np

import kgw_simulation as S

N5 = 5
PI5 = np.full(N5, 1.0 / N5)


def kernel(p: float) -> np.ndarray:
    P = np.zeros((N5, N5))
    for i in range(N5):
        P[i, (i + 1) % N5] += p
        P[i, (i - 1) % N5] += 1.0 - p
    return P


def deviation(p: float) -> float:
    """Detailed-Balance-Abweichung max |π(x) P(x, y) − π(y) P(y, x)| = π · |2p − 1| (exakt)."""
    return float(abs(2.0 * p - 1.0) / N5)


def net_flow_exact(p: float) -> float:
    """Nettofluss je Kante und Schritt in Vorwärtsrichtung: π · (p − (1 − p)) (exakt, auf jeder der fünf Kanten gleich)."""
    return float((2.0 * p - 1.0) / N5)


def kolmogorov(p: float):
    """Produkt der Übergangswahrscheinlichkeiten einmal vorwärts und einmal rückwärts um den Kreis: (p⁵, (1 − p)⁵). Gleich genau dann, wenn Detailed Balance (Kolmogorov-Kriterium)."""
    return float(p ** N5), float((1.0 - p) ** N5)


def simulate_net_flow(p: float, steps: int, seed: int) -> np.ndarray:
    """Gemessener Nettofluss je Kante i → i+1 und Schritt über `steps` Schritte einer einzigen Kette (SplitMix64, Start gleichverteilt aus dem ersten Wert)."""
    if steps < 1:
        raise ValueError("Die Simulation braucht mindestens einen Schritt.")
    u64 = S.u64_stream(seed, steps + 1)
    x0 = int(u64[0] % np.uint64(N5))
    u = S.to_unit(u64[1:])
    step = np.where(u < p, 1, -1)
    pos = (x0 + np.cumsum(step)) % N5
    prev = np.concatenate(([x0], pos[:-1]))
    forward = step == 1
    up = np.bincount(prev[forward], minlength=N5)                 # Schritte i → i+1 zählen bei i
    down = np.bincount(pos[~forward], minlength=N5)               # Schritte i+1 → i zählen bei i
    return (up - down) / float(steps)


def net_flow_sd(steps: int) -> float:
    """Näherung für die Standardabweichung des gemessenen Nettoflusses je Kante: √(0,2 / T). Ein Schritt überquert eine gegebene Kante mit Wahrscheinlichkeit 0,2
    (für jedes p); die Schritte werden als unabhängig behandelt, die Abhängigkeit zwischen aufeinanderfolgenden Schritten bleibt unberücksichtigt."""
    if steps < 1:
        raise ValueError("Die Simulation braucht mindestens einen Schritt.")
    return float((1.0 / N5 / steps) ** 0.5)

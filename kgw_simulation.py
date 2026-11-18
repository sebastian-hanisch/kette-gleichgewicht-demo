"""Simulation mit dem ganzzahligen SplitMix64 (kgw_rng): vektorisierte Zufallszahlen, Zeitanteile einer Kette, Rauschgrenze unabhängiger Ziehungen.

SplitMix64 ist zählerbasiert: die k-te Ausgabe hängt nur von seed + k · 0x9E3779B97F4A7C15 ab. `u64_stream` rechnet deshalb viele Ausgaben auf einmal
mit numpy (Ganzzahlen modulo 2^64, ohne Gleitkomma-Bibliothek) und liefert genau die Folge von `kgw_rng.SplitMix64.next_u64()`; tests/test_simulation.py
vergleicht beide. Zufallsergebnisse der App sind Bänder, keine Zahlen: gleiche Zufalls-Seeds geben auf jeder Plattform dieselben Folgen.
"""

from bisect import bisect_right

import numpy as np

from kgw_rng import MASK  # noqa: F401  (Maske des Generators: dieselbe 2^64-Arithmetik)

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def u64_stream(seed: int, count: int) -> np.ndarray:
    """Die ersten `count` Ausgaben von SplitMix64(seed) als uint64-Feld (Ausgabe k = 1 … count)."""
    k = np.arange(1, count + 1, dtype=np.uint64)
    z = np.uint64(seed & MASK) + k * _GOLDEN
    z = (z ^ (z >> np.uint64(30))) * _M1
    z = (z ^ (z >> np.uint64(27))) * _M2
    return z ^ (z >> np.uint64(31))


def to_unit(u64: np.ndarray) -> np.ndarray:
    """Gleichverteilt in [0, 1) mit 53 Bit, wie `SplitMix64.random()`."""
    return (u64 >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def uniforms(seed: int, count: int) -> np.ndarray:
    return to_unit(u64_stream(seed, count))


def time_fractions(P: np.ndarray, pi: np.ndarray, steps: int, seed: int) -> np.ndarray:
    """Anteil der Zeit in jedem Zustand über `steps` Schritte einer einzigen Kette; Start aus π gezogen (der erste Zufallswert), danach ein Wert je Schritt."""
    if steps < 1:
        raise ValueError("Die Simulation braucht mindestens einen Schritt.")
    u = uniforms(seed, steps + 1).tolist()
    cum_pi = np.cumsum(pi)
    x = min(bisect_right(cum_pi.tolist(), u[0]), len(pi) - 1)
    cum = np.cumsum(P, axis=1)
    cum[:, -1] = 2.0                       # der letzte Eintrag fängt Rundungsreste ab
    rows = [c.tolist() for c in cum]
    counts = np.zeros(len(pi))
    for t in range(1, steps + 1):
        x = bisect_right(rows[x], u[t])
        counts[x] += 1
    return counts / steps


def iid_time_fraction_noise(pi: np.ndarray, steps: int, seed: int, reps: int) -> float:
    """Rauschgrenze: größter Totalvariationsabstand zu π, den `reps` Stichproben aus je `steps` unabhängigen Ziehungen aus π erreichen."""
    if steps < 1:
        raise ValueError("Die Simulation braucht mindestens einen Schritt.")
    cum = np.cumsum(pi)
    cum[-1] = 2.0
    worst = 0.0
    for r in range(reps):
        draws = np.searchsorted(cum, uniforms(seed + r, steps), side="right")
        freq = np.bincount(draws, minlength=len(pi)) / steps
        worst = max(worst, float(0.5 * np.abs(freq - pi).sum()))
    return worst

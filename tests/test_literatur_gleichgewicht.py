"""Literaturwerte (ORACLE-PLAYBOOK §5.1): nur Werte aus Quellen, die gelesen sind; je Wert Quelle, Seite und Stand der Einsicht.

* Levin/Peres/Wilmer, *Markov Chains and Mixing Times*, 2. Auflage (AMS 2017), ganzes Buch gelesen (Stufe W), Autorenseite Levin: Detailed Balance (1.29) und Satz 1.20 (Seite 13:
  Detailed Balance ⇒ stationär), Beispiel 1.22 (Seite 14: die Irrfahrt auf dem n-Kreis mit Drift p hat die Gleichverteilung als stationäre Verteilung, erfüllt (1.29) aber für p ≠ 1/2 nicht),
  Metropolis-Matrix (3.5, Seite 41) und die größtmögliche Annahme b(x, y) = π(x) ∧ π(y) (Seite 39), Definition von d(t) und t_mix (4.30 bis 4.31).
* OEIS A000032 (Lucas-Zahlen; Folge 2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322): die Zahl der Zustände des Rings mit n Plätzen ist L_n für n ≥ 3.

Nicht gelesen und deshalb ohne Test: Peskun 1973 und Tierney 1998 (Ordnung nach Leitwerten), Diaconis/Holmes/Neal 2000 im Original, die Nummer des Konvergenzsatzes in Levin/Peres/Wilmer."""

import numpy as np
import pytest

import kgw_checks as H
import kgw_cycle5 as Y
import kgw_kernels as K
from kgw_rng import SplitMix64
from kgw_ring import make_ring, normalisation

LUCAS = {3: 4, 4: 7, 5: 11, 6: 18, 7: 29, 8: 47, 9: 76, 10: 123, 11: 199, 12: 322}      # OEIS A000032


def test_beispiel_1_22_gleichverteilung_stationaer_detailed_balance_nur_bei_p_halb():
    """LPW Beispiel 1.22 (Seite 14): π gleichverteilt stationär für jedes p, Gleichung (1.29) nur für p = 1/2."""
    for p in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
        P = Y.kernel(p)
        assert H.stationary_residual(P, Y.PI5) < 1e-15, p
        assert (H.db_deviation(P, Y.PI5) < 1e-15) == (p == 0.5), p


def test_satz_1_20_detailed_balance_impliziert_stationaer_an_zufaelligen_symmetrischen_leitwerten():
    """LPW Satz 1.20 (Seite 13): jede Verteilung mit Detailed Balance ist stationär. Geprüft an 40 zufälligen symmetrischen Leitwerten auf dem Zustandsgraphen (Zeilensumme ≤ π(x))."""
    rng = SplitMix64(2026)
    for n in (4, 5, 6, 7):
        ring = make_ring(n, 2.0)
        for _ in range(10):
            c = np.array([rng.random() for _ in range(ring.n_edges)])
            row = np.zeros(ring.size)                         # Summe der Leitwerte je Zustand
            np.add.at(row, ring.edges[:, 0], c)
            np.add.at(row, ring.edges[:, 1], c)
            c = c * np.min(ring.pi / row)                     # so skaliert, dass jede Zeilensumme ≤ π(x)
            P = K.kernel_from_flows(ring, c, c)
            assert H.db_deviation(P, ring.pi) < 1e-15 and H.stationary_residual(P, ring.pi) < 1e-14


def test_metropolis_ist_der_groesstmoegliche_zulaessige_leitwert():
    """LPW Seite 39: b(x, y) = π(x) ∧ π(y) ist die größtmögliche Lösung. Mit Platzwahl 1/n ist min(π(x), π(y))/n der größte symmetrische Leitwert; mehr verletzt eine Zeilensumme."""
    ring = make_ring(6, 2.0)
    base = K.metropolis_conductance(ring)
    zeile = np.zeros(ring.size)
    np.add.at(zeile, ring.edges[:, 0], base)
    np.add.at(zeile, ring.edges[:, 1], base)
    assert np.all(zeile <= ring.pi + 1e-15), "Metropolis ist zulässig"
    # jede Vergrößerung eines einzelnen Leitwerts über min(π(x), π(y))/n hinaus bricht die Annahme a ≤ 1: P(x, y) > 1/n
    k = 0
    a, b = ring.edges[k]
    assert (1.001 * base[k]) / ring.pi[a] > 1 / ring.n or (1.001 * base[k]) / ring.pi[b] > 1 / ring.n


def test_zustandszahlen_sind_lucas_zahlen_und_normierung_ist_mu_plus_hoch_n_plus_mu_minus_hoch_n():
    """OEIS A000032: die Zahl der Zustände ist L_n; Z = μ₊ⁿ + μ₋ⁿ (Transfermatrix), bei λ = 1 gerade L_n."""
    for n, count in LUCAS.items():
        assert make_ring(n, 2.0).size == count, n
        assert normalisation(n, 1.0) == pytest.approx(count, rel=1e-12), n


def test_definition_d_t_und_t_mix_an_einem_zwei_zustands_kern():
    """LPW (4.30 bis 4.31): d(t) = max_x TV(P^t(x, ·), π). Für P = [[1−a, a], [b, 1−b]] ist d(t) = max(π₁, π₂) · |1 − a − b|^t mit π = (b, a)/(a + b) (zwei Zustände, gerechnet von Hand)."""
    a, b = 0.3, 0.1
    P = np.array([[1 - a, a], [b, 1 - b]])
    pi = np.array([b, a]) / (a + b)
    for t in range(0, 12):
        assert H.d_curve(P, pi, 11)[t] == pytest.approx(max(pi) * abs(1 - a - b) ** t, abs=1e-14), t
    assert H.t_mix(P, pi) == next(t for t in range(100) if max(pi) * 0.6 ** t <= 0.25)

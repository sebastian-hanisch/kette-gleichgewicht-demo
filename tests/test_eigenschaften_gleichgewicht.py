"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen, die ohne Kenntnis der richtigen Zahl gelten müssen. Je Eigenschaft 20 bis 50 Zufallsinstanzen mit festem Zufalls-Seed.

Zufall nur über kgw_rng (SplitMix64). Jede Eigenschaft ist eine belegte Aussage über Kerne und Flüsse, nicht das heutige Verhalten des Codes:
Leitwerte mit Zeilensumme ≤ π(x) geben Zeilensumme 1, Einträge ≥ 0, πP = π und Detailed Balance; gemeinsames Skalieren der Gewichte ändert den Kern nicht; Umnummerieren permutiert
die Matrix; Drehen oder Spiegeln des Rings ändert t_mix nicht; (P*)* = P; P* hat dasselbe π; P* = P genau dann, wenn Detailed Balance gilt; ein Kreisstrom bis zur größten Stärke erhält
πP = π; d(t) fällt nicht; die Dimension der Flüsse wächst mit der Zahl der Kanten."""

import numpy as np
import pytest

import kgw_checks as H
import kgw_constants as C
import kgw_kernels as K
from kgw_ring import Ring, make_ring
from kgw_rng import SplitMix64


def _zufaellige_leitwerte(ring, rng):
    c = np.array([rng.random() for _ in range(ring.n_edges)])
    row = np.zeros(ring.size)
    np.add.at(row, ring.edges[:, 0], c)
    np.add.at(row, ring.edges[:, 1], c)
    return c * float(np.min(ring.pi / row)) * (0.3 + 0.7 * rng.random())


def test_jeder_zulaessige_symmetrische_leitwert_gibt_einen_kern_mit_pi():
    rng = SplitMix64(1)
    for n in (3, 4, 5, 6, 7):
        ring = make_ring(n, 0.5 + 4.5 * rng.random())
        for _ in range(8):
            P = K.kernel_from_flows(ring, c := _zufaellige_leitwerte(ring, rng), c)
            assert H.row_sum_error(P) < 1e-14 and H.min_entry(P) >= 0
            assert H.stationary_residual(P, ring.pi) < 1e-14 and H.db_deviation(P, ring.pi) < 1e-15


def test_gemeinsames_skalieren_der_gewichte_aendert_den_kern_nicht():
    """π(x) = Gewicht / Z: ein gemeinsamer Faktor kürzt sich. Geprüft über die Leitwert-Regeln mit unnormierten Gewichten."""
    for n in (4, 5, 6):
        ring = make_ring(n, 2.0)
        w = 2.0 ** ring.count.astype(float)
        for faktor in (0.1, 3.0, 1000.0):
            pi = faktor * w / np.sum(faktor * w)
            assert pi == pytest.approx(ring.pi, abs=1e-15)
            a, b = ring.edges[:, 0], ring.edges[:, 1]
            P = np.zeros((ring.size, ring.size))
            P[a, b] = np.minimum(1.0, (faktor * w[b]) / (faktor * w[a])) / n      # Metropolis-Regel mit unnormierten Gewichten
            P[b, a] = np.minimum(1.0, (faktor * w[a]) / (faktor * w[b])) / n
            P[np.diag_indices_from(P)] = 1 - P.sum(axis=1)
            assert P == pytest.approx(K.metropolis(ring), abs=1e-14)


def test_umnummerieren_der_zustaende_permutiert_die_matrix():
    rng = SplitMix64(2)
    ring = make_ring(6, 2.0)
    for _ in range(10):
        perm = list(range(ring.size))
        rng.shuffle(perm)
        perm = np.array(perm)
        inv = np.argsort(perm)
        neu = Ring(ring.n, ring.lam, tuple(ring.masks[i] for i in perm), ring.count[perm], ring.pi[perm],
                   np.array([[inv[a], inv[b]] for a, b in ring.edges]), ring.deg[perm], inv[ring.rot[perm]], {ring.masks[i]: k for k, i in enumerate(perm)})
        P_neu = K.metropolis(neu)
        assert P_neu == pytest.approx(K.metropolis(ring)[np.ix_(perm, perm)], abs=1e-14)


def test_drehen_des_rings_aendert_t_mix_nicht():
    """Die Drehung ist eine Symmetrie des Rings: t_mix von Metropolis ist für Kerne auf dem gedrehten Ring gleich (Zustände um einen Platz verschoben)."""
    for n in (4, 5, 6, 7):
        ring = make_ring(n, 2.0)
        P = K.metropolis(ring)
        Pr = P[np.ix_(ring.rot, ring.rot)]
        assert H.t_mix(P, ring.pi) == H.t_mix(Pr, ring.pi[ring.rot])
        assert Pr == pytest.approx(P, abs=1e-14), "Metropolis kommutiert mit der Drehung"


def test_zeitumkehr_ist_eine_involution_hat_dasselbe_pi_und_ist_gleich_p_genau_bei_detailed_balance():
    rng = SplitMix64(3)
    ring = make_ring(5, 2.0)
    for q, p in [(0.2, 0.7), (0.5, 1.0), (0.1, 0.5), (0.3, 0.5), (0.0, 0.7)]:
        P = K.mixture(ring, q, p)
        Pstar = H.time_reversal(P, ring.pi)
        assert H.time_reversal(Pstar, ring.pi) == pytest.approx(P, abs=1e-14)
        assert H.stationary_residual(Pstar, ring.pi) < 1e-14 and H.row_sum_error(Pstar) < 1e-14
        assert (np.abs(Pstar - P).max() < 1e-13) == (H.db_deviation(P, ring.pi) < 1e-13), (q, p)
    for _ in range(20):
        P = K.kernel_from_flows(ring, c := _zufaellige_leitwerte(ring, rng), c)
        assert H.time_reversal(P, ring.pi) == pytest.approx(P, abs=1e-14), "reversibel: P* = P"


def test_kreisstrom_beliebiger_staerke_bis_zum_maximum_erhaelt_pi():
    for n in (4, 5, 6):
        ring = make_ring(n, 2.0)
        for j in range(20):
            for anteil in (0.0, 0.3, 1.0):
                P, Prev, smax = K.cycle_kernels(ring, 0.5, anteil, 1000 + j)
                assert H.stationary_residual(P, ring.pi) < 1e-13 and H.row_sum_error(P) < 1e-13 and H.min_entry(P) >= -1e-15 and smax > 0
                if anteil == 0.0:
                    assert P == pytest.approx(Prev, abs=1e-15), "ohne Strom ist der Kern der Basiskern"
                else:
                    assert H.db_deviation(P, ring.pi) > 1e-9, "der Kreisstrom verletzt Detailed Balance"


def test_d_von_t_faellt_nicht_fuer_reversible_und_nichtreversible_kerne():
    ring = make_ring(5, 2.0)
    kerne = [K.metropolis(ring), K.heat_bath(ring), K.equal_conductance(ring), K.mixture(ring, 0.2, 1.0), K.mixture(ring, 0.5, 0.7), K.cycle_kernels(ring, 0.5, 1.0, 7)[0]]
    for P in kerne:
        d = H.d_curve(P, ring.pi, 60)
        assert np.all(np.diff(d) <= 1e-12)


def test_die_dimension_der_fluesse_waechst_mit_der_zahl_der_kanten():
    """Eine zusätzliche Kante (ein weiterer erlaubter Zug) erhöht die Dimension der stationären Flüsse um 2 und die der reversiblen um 1, solange der Graph zusammenhängend bleibt (Rang V − 1)."""
    ring = make_ring(5, 1.0)
    pairs = [(int(a), int(b)) for a, b in ring.edges]
    base_rank = np.linalg.matrix_rank(H.conservation_matrix(ring.size, pairs + [(b, a) for a, b in pairs]))
    # alle weiteren Paare, die noch keine Kante sind
    vorhanden = set(pairs) | {(b, a) for a, b in pairs}
    extra = [(x, y) for x in range(ring.size) for y in range(x + 1, ring.size) if (x, y) not in vorhanden][:6]
    for k in range(1, len(extra) + 1):
        gewaehlt = pairs + extra[:k]
        gerichtet = gewaehlt + [(b, a) for a, b in gewaehlt]
        rang = np.linalg.matrix_rank(H.conservation_matrix(ring.size, gerichtet))
        assert rang == base_rank == ring.size - 1
        assert len(gerichtet) - rang == 2 * (ring.n_edges + k) - (ring.size - 1)


def test_t_mix_hoeht_sich_nicht_wenn_man_den_leitwertfaktor_vergroessert():
    """Messung, kein Satz: f = 0,1; 0,25; 0,5; 0,75; 1 für n = 6 und n = 8, λ = 2 und 5."""
    for n in (6, 8):
        for lam in (2.0, 5.0):
            ring = make_ring(n, lam)
            zeiten = [H.t_mix(K.metropolis(ring, f), ring.pi) for f in (0.1, 0.25, 0.5, 0.75, 1.0)]
            assert all(zeiten[i] >= zeiten[i + 1] for i in range(4)), (n, lam, zeiten)


def test_gleicher_zufalls_seed_gleiche_kreisstroeme_andere_seeds_andere():
    ring = make_ring(6, 2.0)
    a, b, c = (K.random_cycle_current(ring, s) for s in (5, 5, 6))
    assert np.array_equal(a, b) and not np.allclose(a, c)


def test_gewicht_lambda_eins_macht_pi_gleichverteilt_und_die_fehlerfaelle_wirkungslos():
    for n in (4, 5, 6):
        ring = make_ring(n, 1.0)
        assert ring.pi == pytest.approx(np.full(ring.size, 1 / ring.size), abs=1e-15)
        for name in C.KERN_FEHLER:
            assert H.stationary_residual(K.build_kernel(ring, name), ring.pi) < 1e-13, name

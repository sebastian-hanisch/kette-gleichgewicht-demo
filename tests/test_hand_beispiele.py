"""Handbeispiele: jede Zahl ist von Hand (Bruch, Ganzzahl, Menge) nachgerechnet und steht hier als Rechnung, nicht als Abschrift des Codes (DEMO-PLAYBOOK §1b und §5).

Das Hauptbeispiel ist der Ring mit 4 Plätzen und λ = 2: 7 Zustände (leer, vier Einzelne, zwei gegenüberliegende Paare), Gewichte 1, 2, 2, 2, 2, 4, 4, Normierung 17.
Die exakte Gegenseite (`exakt.py`) rechnet in Brüchen und baut Metropolis und Heat-bath aus den Regeln auf; die App-Matrizen müssen bis auf Rundung gleich sein."""

from fractions import Fraction as F

import numpy as np
import pytest

import exakt
import kgw_checks as H
import kgw_cycle5 as Y
import kgw_evaluation as E
import kgw_kernels as K
from kgw_ring import make_ring, rotation_matrix, rotation_orbits


# ---- H1: Raum, Gewichte, Normierung ---------------------------------------------------------------------------------------------------------
def test_h1_raum_gewichte_normierung(ring4, raum4):
    assert ring4.size == 7 == raum4.S, "leer, vier Einzelne, zwei gegenüberliegende Paare"
    assert sorted(raum4.w) == [1, 2, 2, 2, 2, 4, 4] and raum4.Z == 17
    assert raum4.Z == 2 ** 4 + (-1) ** 4, "Transfermatrix: μ± = 2 und −1, Z = μ₊⁴ + μ₋⁴"
    assert raum4.pi == [F(1, 17), F(2, 17), F(2, 17), F(2, 17), F(2, 17), F(4, 17), F(4, 17)]
    assert sorted(np.round(ring4.pi * 17, 12).tolist()) == [1, 2, 2, 2, 2, 4, 4]
    assert len(ring4.edges) == 8 == len(raum4.edges) and sorted(ring4.deg.tolist()) == [2, 2, 2, 2, 2, 2, 4]


def test_h1_zweiter_raum_n5(ring5, raum5):
    assert ring5.size == 11 and raum5.Z == 31 == 2 ** 5 + (-1) ** 5
    assert sorted(raum5.w) == [1] + [2] * 5 + [4] * 5
    assert len(ring5.edges) == 15


# ---- H2: gleiche Leitwerte ------------------------------------------------------------------------------------------------------------------
def test_h2_gleiche_leitwerte(ring4, raum4, perm4):
    assert K.equal_conductance_value(ring4) == pytest.approx(1 / 68, abs=1e-15), "c = min π(x)/Grad(x) = (1/17)/4"
    P = K.equal_conductance(ring4)
    exact, c = raum4.gleiche_leitwerte()
    assert c == F(1, 68)
    assert exakt.umnummeriert(P, perm4, exact) < 1e-15
    assert all(P[0, j] == pytest.approx(1 / 4) for j in range(1, 5)) and P[0, 0] == pytest.approx(0, abs=1e-15), "aus ∅: jeder Einzelne mit 1/4, nichts bleibt"
    e = ring4.index[1]
    assert P[e, 0] == pytest.approx(1 / 8) and P[e, ring4.index[5]] == pytest.approx(1 / 8) and P[e, e] == pytest.approx(3 / 4)
    pair = ring4.index[5]
    assert P[pair, e] == pytest.approx(1 / 16) and P[pair, pair] == pytest.approx(7 / 8)
    assert H.stationary_residual(P, ring4.pi) < 1e-15 and H.db_deviation(P, ring4.pi) < 1e-17


# ---- H3: Metropolis und Heat-bath als Leitwerte ---------------------------------------------------------------------------------------------
def test_h3_metropolis_und_heat_bath(ring4, raum4, perm4):
    Pm, Ph = K.metropolis(ring4), K.heat_bath(ring4)
    assert exakt.umnummeriert(Pm, perm4, raum4.metropolis()) < 1e-15
    assert exakt.umnummeriert(Pm, perm4, raum4.metropolis_nach_regel()) < 1e-15, "Leitwert-Sicht und Regel-Sicht geben dieselbe Matrix"
    assert exakt.umnummeriert(Ph, perm4, raum4.waermebad_nach_regel()) < 1e-15
    assert exakt.umnummeriert(Ph, perm4, raum4.metropolis(F(2, 3))) < 1e-15, "Heat-bath = Metropolis-Leitwerte mal max(1, λ)/(1 + λ) = 2/3 (Kreuzprobe)"
    cm = K.metropolis_conductance(ring4)
    assert sorted(set(np.round(cm * 68, 12).tolist())) == [1.0, 2.0], "Leitwerte: leer–Einzelner 1/68, Einzelner–Paar 2/68"


# ---- H4: gleiches π, andere Zeit --------------------------------------------------------------------------------------------------------------
def test_h4_mischzeiten_und_d1(ring4, raum4):
    kerne = {"M": (K.metropolis(ring4), raum4.metropolis()), "G": (K.equal_conductance(ring4), raum4.gleiche_leitwerte()[0]),
             "W": (K.heat_bath(ring4), raum4.waermebad_nach_regel())}
    assert {k: (H.t_mix(v[0], ring4.pi), exakt.t_mix(v[1], raum4.pi)) for k, v in kerne.items()} == {"M": (15, 15), "G": (18, 18), "W": (22, 22)}
    assert {k: exakt.d_t(v[1], raum4.pi, 1) for k, v in kerne.items()} == {"M": F(10, 17), "G": F(95, 136), "W": F(67, 102)}
    for k, v in kerne.items():
        assert H.d_curve(v[0], ring4.pi, 1)[1] == pytest.approx(float(exakt.d_t(v[1], raum4.pi, 1)), abs=1e-14), k


def test_h4_verteilung_nach_t_schritten_von_leer(ring4, raum4, perm4):
    Pm = raum4.metropolis()
    mu1 = Pm[0]
    assert mu1 == [0, F(1, 4), F(1, 4), F(1, 4), F(1, 4), 0, 0], "μ₁ = Zeile ∅ des Kerns"
    assert exakt.tv(mu1, raum4.pi) == F(9, 17) and exakt.tv([F(1)] + [F(0)] * 6, raum4.pi) == F(16, 17)
    mu2 = exakt.vm(mu1, Pm)
    assert mu2 == [F(1, 8), F(5, 32), F(5, 32), F(5, 32), F(5, 32), F(1, 8), F(1, 8)], "leer 1/8, Einzelne je 5/32 (= 1/4 · 5/8), Paare je 1/8 (= 2 · 1/4 · 1/4)"
    assert exakt.tv(mu2, raum4.pi) == F(15, 68)
    curve = H.mu_curve(K.metropolis(ring4), ring4.pi, 0, 2)
    assert curve[:3] == pytest.approx([16 / 17, 9 / 17, 15 / 68], abs=1e-14)


# ---- H5: 5-Kreis mit Drift ------------------------------------------------------------------------------------------------------------------
def test_h5_kreis_stationaer_ohne_detailed_balance():
    p = F(7, 10)
    P = exakt.kreis5(p)
    pi5 = [F(1, 5)] * 5
    assert exakt.zeilensumme_eins(P) and exakt.stationaer(P, pi5)
    assert exakt.db_abw(P, pi5) == F(2, 25) == F(1, 5) * (2 * p - 1), "Abweichung π(2p − 1) = 1/5 · 2/5 = 2/25 = 0,08"
    assert p ** 5 == F(16807, 100000) and (1 - p) ** 5 == F(243, 100000)
    assert pi5[0] * P[0][1] - pi5[1] * P[1][0] == F(2, 25), "Nettofluss je Kante und Schritt"
    assert Y.deviation(0.7) == pytest.approx(0.08, abs=1e-15) and Y.net_flow_exact(0.7) == pytest.approx(0.08, abs=1e-15)
    assert Y.kolmogorov(0.7) == pytest.approx((0.16807, 0.00243), abs=1e-15)
    assert np.abs(Y.kernel(0.7) - exakt.als_float(P)).max() < 1e-16


def test_h5_zeitumkehr_dreht_die_drift():
    P = exakt.kreis5(F(7, 10))
    pi5 = [F(1, 5)] * 5
    Pstar = [[pi5[j] * P[j][i] / pi5[i] for j in range(5)] for i in range(5)]
    assert Pstar[1][0] == F(7, 10) and Pstar[0][1] == F(3, 10) and exakt.stationaer(Pstar, pi5)
    assert np.abs(H.time_reversal(Y.kernel(0.7), Y.PI5) - exakt.als_float(Pstar)).max() < 1e-15


def test_h5_ankunft_des_kreises():
    pi5 = [F(1, 5)] * 5
    P7 = exakt.kreis5(F(7, 10))
    assert exakt.d_t(exakt.kreis5(F(1, 2)), pi5, 0) == F(4, 5)
    assert (exakt.d_t(P7, pi5, 1), exakt.d_t(P7, pi5, 2)) == (F(3, 5), F(51, 100))
    assert (exakt.t_mix(P7, pi5), exakt.t_mix(exakt.kreis5(F(1, 2)), pi5)) == (6, 5)
    assert (H.t_mix(Y.kernel(0.7), Y.PI5), H.t_mix(Y.kernel(0.5), Y.PI5)) == (6, 5)
    assert all(exakt.d_t(exakt.kreis5(F(1)), pi5, t) == F(4, 5) for t in range(12)), "p = 1: π stationär und eindeutig, aber d(t) = 4/5 für alle t"
    assert exakt.t_mix(exakt.kreis5(F(1)), pi5, tmax=30) is None and H.t_mix(Y.kernel(1.0), Y.PI5) is None
    lazy = [[(F(int(i == j)) + exakt.kreis5(F(1))[i][j]) / 2 for j in range(5)] for i in range(5)]
    assert exakt.t_mix(lazy, pi5) == 5 == H.t_mix(exakt.als_float(lazy), Y.PI5), "träge reine Drehung: t_mix = 5"
    assert H.second_eigenvalue_abs(Y.kernel(0.7)) == pytest.approx(0.8425, abs=5e-5)
    assert abs(np.linalg.eigvals(Y.kernel(0.7))[1].imag) > 0.1, "komplexes Eigenwertpaar: die Verteilung nach t Schritten schwingt"


# ---- H6: Freiheitsgrade ---------------------------------------------------------------------------------------------------------------------
def test_h6_freiheitsgrade_von_hand(raum4):
    S = raum4.S
    alle = [(i, j) for i in range(S) for j in range(S) if i != j]
    assert len(alle) == 42 and exakt.rang(exakt.erhaltung(S, alle)) == 6
    assert 42 - 6 == 36 == (S - 1) ** 2 and 42 // 2 == 21 == S * (S - 1) // 2, "ohne Graphbeschränkung: 36 stationär gegen 21 reversibel"
    gerichtet = [(a, b) for a, b in raum4.edges] + [(b, a) for a, b in raum4.edges]
    assert len(gerichtet) == 16 and exakt.rang(exakt.erhaltung(S, gerichtet)) == 6
    assert 16 - 6 == 10 and len(raum4.edges) == 8 and 10 - 8 == 2 == 8 - 7 + 1


def test_h6_freiheitsgrade_der_app(ring4):
    d = H.flow_dimensions(ring4, numeric=True)
    assert (d["stationaer"], d["reversibel"], d["kreise"], d["rang"]) == (10, 8, 2, 6)
    assert H.flow_dimensions_all_moves(7, numeric=True)["stationaer"] == 36 and H.flow_dimensions_all_moves(7, numeric=True)["reversibel"] == 21
    assert H.flow_dimensions_all_moves(11, numeric=True)["stationaer"] == 100 and H.flow_dimensions_all_moves(11, numeric=True)["reversibel"] == 55


def test_h6_n3_ist_ein_baum_und_n5_hat_fuenf_kreise(raum5):
    r3 = exakt.Raum(3, 2)
    assert len(r3.edges) == r3.S - 1 == 3
    assert len(raum5.edges) - raum5.S + 1 == 5 == 15 - 11 + 1
    assert H.flow_dimensions(make_ring(3, 2.0), numeric=True)["kreise"] == 0


# ---- H7: ein Kreisstrom von Hand ------------------------------------------------------------------------------------------------------------
def test_h7_ein_kreisstrom_von_hand(ring4, raum4):
    h = E.hand_cycle_example()
    assert h.s == pytest.approx(1 / 136) and h.deviation == pytest.approx(1 / 68, abs=1e-15), "Abweichung 2s = 1/68 auf den vier Kreiskanten"
    assert (h.p_empty_to_0, h.p_empty_to_2, h.p_empty_stay) == pytest.approx((1 / 4, 0.0, 1 / 2), abs=1e-15)
    assert (h.tmix_base, h.tmix_cycle) == (29, 29), "der Kreisstrom ändert die Mischzeit in diesem Beispiel nicht"
    # exakter Aufbau von Hand in Brüchen
    i0, i2, i02 = raum4.idx[(0,)], raum4.idx[(2,)], raum4.idx[(0, 2)]
    s = F(1, 136)
    fl = {}
    for a, b in raum4.edges:
        c = F(1, 2) * min(raum4.pi[a], raum4.pi[b]) / 4
        fl[(a, b)] = fl[(b, a)] = c
    for x, y in ((0, i0), (i0, i02), (i02, i2), (i2, 0)):
        fl[(x, y)] += s
        fl[(y, x)] -= s
    assert all(v >= 0 for v in fl.values())
    P = [[F(0)] * raum4.S for _ in range(raum4.S)]
    for (x, y), v in fl.items():
        P[x][y] = v / raum4.pi[x]
    for i in range(raum4.S):
        P[i][i] = 1 - sum(P[i])
    assert exakt.zeilensumme_eins(P) and exakt.stationaer(P, raum4.pi) and exakt.db_abw(P, raum4.pi) == 2 * s == F(1, 68)
    assert (P[0][i0], P[0][i2], P[0][0], P[i0][0]) == (F(1, 4), 0, F(1, 2), 0)
    assert exakt.t_mix(P, raum4.pi) == 29 == exakt.t_mix(raum4.metropolis(F(1, 2)), raum4.pi)


# ---- H8: falsch gebaut ----------------------------------------------------------------------------------------------------------------------
def test_h8_feste_annahme_halb(ring4, raum4, perm4):
    P = raum4.feste_annahme_halb()
    unif = [F(1, 7)] * 7
    assert exakt.zeilensumme_eins(P) and not exakt.stationaer(P, raum4.pi) and exakt.stationaer(P, unif)
    assert exakt.tv(unif, raum4.pi) == F(22, 119)
    assert exakt.umnummeriert(K.fixed_half(ring4), perm4, P) < 1e-15
    stat = H.stationary_solve(K.fixed_half(ring4))
    assert H.tv(stat, ring4.pi) == pytest.approx(22 / 119, abs=1e-14) and stat == pytest.approx(np.full(7, 1 / 7), abs=1e-14)


def test_h8_umgekehrtes_verhaeltnis(ring4, raum4, perm4):
    P = raum4.umgekehrtes_verhaeltnis()
    inv = [1 / w for w in raum4.w]
    pinv = [x / sum(inv) for x in inv]
    assert pinv == [F(2, 7), F(1, 7), F(1, 7), F(1, 7), F(1, 7), F(1, 14), F(1, 14)]
    assert exakt.zeilensumme_eins(P) and exakt.stationaer(P, pinv) and not exakt.stationaer(P, raum4.pi)
    assert exakt.tv(pinv, raum4.pi) == F(39, 119)
    assert exakt.umnummeriert(K.inverse_ratio(ring4), perm4, P) < 1e-15
    assert H.tv(H.stationary_solve(K.inverse_ratio(ring4)), ring4.pi) == pytest.approx(39 / 119, abs=1e-14)


# ---- H9: reine Drehung ----------------------------------------------------------------------------------------------------------------------
def test_h9_reine_drehung_n5(ring5, raum5):
    R = raum5.drehung()
    assert exakt.zeilensumme_eins(R) and exakt.stationaer(R, raum5.pi)
    leer = [F(1)] + [F(0)] * 10
    mu = leer
    for _ in range(12):
        mu = exakt.vm(mu, R)
    assert mu == leer and exakt.tv(mu, raum5.pi) == F(30, 31), "Start leer bleibt leer: d(t) ≥ 30/31 für alle t"
    b5 = exakt.bahnen(R)
    assert len(b5) == 3 and sorted(len(b) for b in b5) == [1, 5, 5]
    zweite = [F(0)] * 11
    zweite[0] = F(1, 3)
    for j in b5[1] + b5[2]:
        zweite[j] = F(1, 15)
    assert sum(zweite) == 1 and exakt.stationaer(R, zweite) and zweite != raum5.pi, "eine zweite stationäre Verteilung"
    assert rotation_orbits(ring5) == 3 == H.n_classes(rotation_matrix(ring5))
    assert H.d_curve(rotation_matrix(ring5), ring5.pi, 20) == pytest.approx(np.full(21, 30 / 31), abs=1e-14)


def test_h9_reine_drehung_n4(ring4, raum4):
    b4 = exakt.bahnen(raum4.drehung())
    assert len(b4) == 3 and sorted(len(b) for b in b4) == [1, 2, 4] and rotation_orbits(ring4) == 3


# ---- H10: Kern gegen Ankunft ----------------------------------------------------------------------------------------------------------------
def test_h10_kern_gegen_ankunft(ring4, raum4):
    Pm = raum4.metropolis()
    mu = [F(1)] + [F(0)] * 6
    tvs = []
    for _ in range(6):
        tvs.append(exakt.tv(mu, raum4.pi))
        assert exakt.db_abw(Pm, raum4.pi) == 0, "die Abweichung des Kerns ist bei jedem t dieselbe (null)"
        mu = exakt.vm(mu, Pm)
    assert tvs == [F(16, 17), F(9, 17), F(15, 68), F(69, 544), F(297, 4352), F(1305, 34816)]
    assert all(tvs[i] > tvs[i + 1] for i in range(5))
    assert H.mu_curve(K.metropolis(ring4), ring4.pi, 0, 5) == pytest.approx([float(x) for x in tvs], abs=1e-14)

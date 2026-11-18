"""Orakel-Tests: die Kerne, Prüfungen und Zahlen der App gegen einen anderen Rechenweg (ORACLE-PLAYBOOK §5, §6).

Die Gegenseite rechnet anders als die App: Brüche statt Gleitkomma, Metropolis und Heat-bath aus den Regeln (Platz, Vorschlag, Annahme) statt aus Leitwerten, die stationäre
Verteilung aus der Formel λ^|x|/Z, aus einem linearen System und aus dem linken Eigenvektor, Mischzeiten aus `numpy.linalg.matrix_power` und Schritt-für-Schritt-Vektoriteration,
Klassen aus der Zahl der Bahnen der Drehung, der Rang der Flüsse aus der Formel 2E − (V − 1). Zufallsergebnisse stehen als Band gegen die exakte Größe.
(a) Kerne gegen Brüche und Regeln, (b) stationäre Verteilung auf drei Wegen, (c) Abweichung und Nettofluss gegen Formeln, (d) t_mix und d(t) gegen Brüche und Potenzen,
(e) Freiheitsgrade, (f) Klassen, (g) Zufallsergebnisse als Band."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import exakt
import kgw_checks as H
import kgw_constants as C
import kgw_cycle5 as Y
import kgw_kernels as K
import kgw_simulation as S
from kgw_ring import make_ring, normalisation, rotation_matrix, rotation_orbits


# ---- (a) Kerne gegen den exakten Aufbau --------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n,lam", [(4, 2), (5, 2), (4, 5), (5, 1), (6, 3)])
def test_a_kerne_stimmen_mit_den_bruechen_und_den_regeln_ueberein(n, lam):
    ring, raum = make_ring(n, float(lam)), exakt.Raum(n, lam)
    perm = exakt.app_nach_exakt(ring, raum)
    assert exakt.umnummeriert(K.metropolis(ring), perm, raum.metropolis_nach_regel()) < 1e-14
    assert exakt.umnummeriert(K.heat_bath(ring), perm, raum.waermebad_nach_regel()) < 1e-14
    assert exakt.umnummeriert(K.equal_conductance(ring), perm, raum.gleiche_leitwerte()[0]) < 1e-14
    assert exakt.umnummeriert(K.fixed_half(ring), perm, raum.feste_annahme_halb()) < 1e-14
    assert exakt.umnummeriert(K.inverse_ratio(ring), perm, raum.umgekehrtes_verhaeltnis()) < 1e-14
    assert exakt.umnummeriert(K.heat_bath_inverse(ring), perm, raum.waermebad_nach_regel(F(1, 1 + lam))) < 1e-14
    assert exakt.umnummeriert(K.metropolis(ring, 0.5), perm, raum.metropolis(F(1, 2))) < 1e-14


def test_a_gestufte_leitwerte_gegen_exakten_aufbau():
    ring, raum = make_ring(6, 2.0), exakt.Raum(6, 2)
    perm = exakt.app_nach_exakt(ring, raum)
    f0, f1 = F(1, 4), F(3, 4)
    exakt_gestuft = raum.kern_aus_leitwert(lambda a, b: (f0 if min(len(raum.st[a]), len(raum.st[b])) == 0 else f1) * min(raum.pi[a], raum.pi[b]) / raum.n)
    assert exakt.umnummeriert(K.metropolis_staged(ring, 0.25, 0.75), perm, exakt_gestuft) < 1e-14


def test_a_mischung_gegen_exakten_aufbau():
    ring, raum = make_ring(5, 2.0), exakt.Raum(5, 2)
    perm = exakt.app_nach_exakt(ring, raum)
    q, p = F(1, 5), F(7, 10)
    R = exakt.als_float(raum.drehung())
    expected = float(1 - q) * exakt.als_float(raum.metropolis()) + float(q) * (float(p) * R + float(1 - p) * R.T)
    assert np.abs(K.mixture(ring, 0.2, 0.7) - expected[np.ix_(perm, perm)]).max() < 1e-14


# ---- (b) stationäre Verteilung auf drei Wegen ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n,lam", [(4, 2.0), (6, 2.0), (6, 0.5), (8, 5.0), (7, 1.0)])
def test_b_stationaere_verteilung_auf_drei_wegen(n, lam):
    ring = make_ring(n, lam)
    formel = np.array([lam ** int(k) for k in ring.count]) / normalisation(n, lam)
    assert ring.pi == pytest.approx(formel, abs=1e-14), "π = λ^|x| / Z mit Z = μ₊ⁿ + μ₋ⁿ"
    for name in (C.K_METRO, C.K_HEAT, C.K_EQUAL):
        P = K.build_kernel(ring, name)
        assert H.stationary_solve(P) == pytest.approx(ring.pi, abs=1e-11), (name, "lineares System")
        w, v = np.linalg.eig(P.T)
        vec = np.real(v[:, np.argmin(np.abs(w - 1))])
        assert vec / vec.sum() == pytest.approx(ring.pi, abs=1e-11), (name, "linker Eigenvektor")
        assert H.stationary_residual(P, ring.pi) < 1e-13 and H.row_sum_error(P) < 1e-14 and H.min_entry(P) >= 0


def test_b_normierung_ueber_transfermatrix_gegen_aufzaehlung_n3_bis_12():
    for n in range(3, 13):
        for lam in (0.5, 1.0, 2.0, 5.0):
            ring = make_ring(n, lam)
            assert float(np.sum(lam ** ring.count.astype(float))) == pytest.approx(normalisation(n, lam), rel=1e-12), (n, lam)


# ---- (c) Abweichung und Nettofluss gegen Formeln -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("p", [0.5, 0.6, 0.7, 0.9, 1.0])
def test_c_kreis_abweichung_und_nettofluss_gegen_formel(p):
    P, pi = Y.kernel(p), Y.PI5
    assert H.stationary_residual(P, pi) < 1e-15
    assert H.db_deviation(P, pi) == pytest.approx(abs(2 * p - 1) / 5, abs=1e-15)
    flow = pi[:, None] * P
    assert flow[0, 1] - flow[1, 0] == pytest.approx((2 * p - 1) / 5, abs=1e-15)


@pytest.mark.parametrize("n,lam", [(4, 2.0), (6, 5.0), (7, 2.0)])
def test_c_mischung_abweichung_gegen_formel(n, lam):
    ring = make_ring(n, lam)
    R = rotation_matrix(ring)
    bewegt = [ring.pi[i] for i in range(ring.size) if not math.isclose((R @ R)[i, i], 1.0)]
    for q in (0.05, 0.2, 0.5):
        for p in (0.5, 0.7, 1.0):
            P = K.mixture(ring, q, p)
            assert H.stationary_residual(P, ring.pi) < 1e-13 and H.row_sum_error(P) < 1e-14 and H.min_entry(P) >= -1e-15
            assert H.db_deviation(P, ring.pi) == pytest.approx(q * abs(2 * p - 1) * max(bewegt), abs=1e-13), (q, p)


# ---- (d) t_mix und d(t) ------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n,lam", [(4, 2), (5, 2), (4, 5), (5, 5)])
def test_d_t_mix_gegen_brueche(n, lam):
    ring, raum = make_ring(n, float(lam)), exakt.Raum(n, lam)
    for name, P_app, P_ex in ((C.K_METRO, K.metropolis(ring), raum.metropolis_nach_regel()), (C.K_HEAT, K.heat_bath(ring), raum.waermebad_nach_regel()),
                              (C.K_EQUAL, K.equal_conductance(ring), raum.gleiche_leitwerte()[0])):
        assert H.t_mix(P_app, ring.pi) == exakt.t_mix(P_ex, raum.pi, tmax=400), (name, n, lam)


@pytest.mark.parametrize("n,lam", [(4, 2.0), (6, 2.0), (6, 5.0), (8, 5.0), (10, 5.0), (10, 0.5)])
def test_d_t_mix_ist_die_kleinste_schwelle_nach_matrix_potenz(n, lam):
    """Der Wert t des Quadrierens und Hebens ist das erste t mit d(t) ≤ 1/4: d(t − 1) > 1/4 ≥ d(t), gerechnet mit numpy.linalg.matrix_power."""
    ring = make_ring(n, lam)
    for name in (C.K_METRO, C.K_HEAT, C.K_EQUAL):
        P = K.build_kernel(ring, name)
        t = H.t_mix(P, ring.pi)
        assert H.d_of(np.linalg.matrix_power(P, t), ring.pi) <= 0.25 + 1e-12, (name, t)
        assert H.d_of(np.linalg.matrix_power(P, t - 1), ring.pi) > 0.25 - 1e-12, (name, t)


def test_d_t_mix_gegen_schrittweise_vektoriteration():
    """Unabhängig vom Quadrieren: von jedem Start die Verteilung Schritt für Schritt vorrechnen, d(t) als Maximum über die Starts."""
    ring = make_ring(6, 2.0)
    for name in (C.K_METRO, C.K_HEAT, C.K_EQUAL):
        P = K.build_kernel(ring, name)
        V = np.eye(ring.size)
        t = 0
        while 0.5 * np.abs(V - ring.pi).sum(axis=1).max() > 0.25 + 1e-12:
            V = V @ P
            t += 1
        assert H.t_mix(P, ring.pi) == t, name


def test_d_zwanzig_zellen_der_tabelle():
    expected = {(4, 2.0): (15, 22, 18), (6, 2.0): (24, 36, 50), (6, 5.0): (87, 104, 502), (8, 5.0): (127, 153, 2675), (10, 5.0): (173, 208, 13655)}
    for (n, lam), times in expected.items():
        assert tuple(H.t_mix(K.build_kernel(make_ring(n, lam), name), make_ring(n, lam).pi) for name in (C.K_METRO, C.K_HEAT, C.K_EQUAL)) == times, (n, lam)


def test_d_t_mix_nie_bei_der_reinen_drehung():
    ring = make_ring(5, 2.0)
    assert H.t_mix(rotation_matrix(ring), ring.pi) is None and H.t_mix(Y.kernel(1.0), Y.PI5) is None


# ---- (e) Freiheitsgrade -------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", range(3, 10))
def test_e_rang_der_erhaltung_gegen_formel_n3_bis_9(n):
    ring = make_ring(n, 1.0)
    d = H.flow_dimensions(ring, numeric=True)
    assert d["rang"] == ring.size - 1, "zusammenhängender Graph: Rang V − 1"
    assert d["stationaer"] == 2 * ring.n_edges - (ring.size - 1) == d["stationaer_formel"]
    assert d["stationaer"] - d["reversibel"] == ring.n_edges - ring.size + 1 == d["kreise"]
    assert H.graph_connected(ring)


def test_e_basiskreise_haben_die_richtige_zahl_und_erhalten_die_bilanz():
    for n in (4, 5, 6, 7):
        ring = make_ring(n, 2.0)
        basis = K.cycle_basis(ring)
        assert len(basis) == ring.n_edges - ring.size + 1
        for vec in basis:
            net = np.zeros(ring.size)       # Nettoabfluss je Zustand
            for k, (a, b) in enumerate(ring.edges):
                net[a] += vec[k]
                net[b] -= vec[k]
            assert np.abs(net).max() < 1e-12
        assert np.linalg.matrix_rank(basis) == len(basis), "unabhängig"


# ---- (f) Klassen --------------------------------------------------------------------------------------------------------------------------------
def test_f_klassen_der_reinen_drehung_gegen_bahnenzahl():
    expected = [2, 3, 3, 5, 5, 8, 10, 15]
    assert [rotation_orbits(make_ring(n, 2.0)) for n in range(3, 11)] == expected
    assert [H.n_classes(rotation_matrix(make_ring(n, 2.0))) for n in range(3, 11)] == expected


def test_f_stationaere_verteilungen_der_reinen_drehung_haben_dieselbe_dimension():
    for n in (3, 4, 5, 6):
        ring = make_ring(n, 2.0)
        R = rotation_matrix(ring)
        assert ring.size - np.linalg.matrix_rank(R.T - np.eye(ring.size)) == rotation_orbits(ring)


# ---- (g) Zufallsergebnisse als Band gegen die exakte Größe ----------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [11001, 11002])
def test_g_nettofluss_innerhalb_vier_standardabweichungen(seed):
    steps = 1_000_000
    net = Y.simulate_net_flow(0.7, steps, seed)
    assert np.abs(net - 0.08).max() < 4 * Y.net_flow_sd(steps)


def test_g_zeitanteile_liegen_im_vielfachen_der_rauschgrenze():
    ring = make_ring(6, 2.0)
    P = K.metropolis(ring)
    noise = S.iid_time_fraction_noise(ring.pi, 100_000, 5, 4)
    for seed in (11081, 11082):
        assert H.tv(S.time_fractions(P, ring.pi, 100_000, seed), ring.pi) < 6 * noise, seed


def test_g_falscher_kern_wird_von_der_simulation_nicht_gerettet():
    """Die Zeitanteile eines falsch gebauten Kerns laufen zum falschen Gleichgewicht: ihr Abstand zu π liegt beim Abstand der stationären Verteilung, weit über der Rauschgrenze."""
    ring = make_ring(6, 2.0)
    P = K.fixed_half(ring)
    stat_tv = H.tv(H.stationary_solve(P), ring.pi)
    frac = S.time_fractions(P, ring.pi, 100_000, 11001)
    assert abs(H.tv(frac, ring.pi) - stat_tv) < 0.03 and H.tv(frac, ring.pi) > 5 * S.iid_time_fraction_noise(ring.pi, 100_000, 7, 4)

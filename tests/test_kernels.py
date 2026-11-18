"""Kerne und Prüfungen als Einheiten: Zweige, Fehler bei unzulässigen Leitwerten, Prüfer an Mini-Instanzen (DEMO-PLAYBOOK §1b)."""

import numpy as np
import pytest

import kgw_checks as H
import kgw_constants as C
import kgw_kernels as K
from kgw_ring import make_ring, rotation_matrix


# ---- Aufbau aus Flüssen -------------------------------------------------------------------------------------------------------------------------
def test_kern_aus_fluessen_ergaenzt_die_diagonale_und_lehnt_zu_grosse_fluesse_ab(ring4):
    c = np.full(ring4.n_edges, 1 / 68)
    P = K.kernel_from_flows(ring4, c, c)
    assert P.sum(axis=1) == pytest.approx(np.ones(7), abs=1e-15)
    with pytest.raises(ValueError, match="Zeilensumme über 1"):
        K.kernel_from_flows(ring4, 10 * c, 10 * c)


def test_negativer_fluss_ist_ein_fehler_kein_zahlenwert(ring4):
    c = np.full(ring4.n_edges, 1 / 68)
    bad = c.copy()
    bad[0] = -0.01
    with pytest.raises(ValueError, match="negativ"):
        K.kernel_from_flows(ring4, bad, c)


def test_nicht_symmetrische_leitwerte_verletzen_detailed_balance_und_meist_auch_pi(ring4):
    """Ein Fluss, der nur in einer Richtung läuft, ist kein Leitwert: die Prüfung zeigt es als Zahl. Eine einzelne Kante mit Zusatzfluss verletzt die Zustandsbilanz."""
    c = np.full(ring4.n_edges, 1 / 136)
    c2 = c.copy()
    c2[0] += 0.005
    P = K.kernel_from_flows(ring4, c2, c)
    assert H.db_deviation(P, ring4.pi) > 1e-3 and H.stationary_residual(P, ring4.pi) > 1e-3


# ---- Zweige: jede Variante weicht von ihren Nachbarn ab ---------------------------------------------------------------------------------------
def test_jede_regel_und_jeder_fehlerfall_ist_ein_eigener_kern():
    ring = make_ring(6, 2.0)
    kerne = {name: K.build_kernel(ring, name, 0.5, 0.5) for name in C.KERN_OPTIONS}
    namen = list(kerne)
    for i in range(len(namen)):
        for j in range(i + 1, len(namen)):
            assert np.abs(kerne[namen[i]] - kerne[namen[j]]).max() > 1e-6, (namen[i], namen[j])


def test_nur_die_gebauten_regeln_haben_pi_die_fehlerfaelle_nicht():
    ring = make_ring(6, 2.0)
    for name in C.KERN_OPTIONS:
        res = H.stationary_residual(K.build_kernel(ring, name, 0.5, 0.5), ring.pi)
        assert (res > 1e-3) == (name in C.KERN_FEHLER), name


def test_gestufte_leitwerte_f0_nur_auf_den_kanten_ab_dem_leeren_ring(ring4):
    f = K.staged_factors(ring4, 0.2, 0.9)
    stufe = np.minimum(ring4.count[ring4.edges[:, 0]], ring4.count[ring4.edges[:, 1]])
    assert (f[stufe == 0] == 0.2).all() and (f[stufe == 1] == 0.9).all() and (stufe == 0).sum() == 4
    assert K.metropolis_staged(ring4, 1.0, 1.0) == pytest.approx(K.metropolis(ring4), abs=1e-16)


def test_gestufte_leitwerte_aendern_die_mischzeit(ring4):
    assert H.t_mix(K.metropolis_staged(ring4, 0.2, 1.0), ring4.pi) != H.t_mix(K.metropolis_staged(ring4, 1.0, 1.0), ring4.pi)
    assert H.t_mix(K.metropolis_staged(ring4, 1.0, 0.2), ring4.pi) != H.t_mix(K.metropolis_staged(ring4, 1.0, 1.0), ring4.pi)


def test_unbekannter_kern_ist_ein_fehler(ring4):
    with pytest.raises(ValueError, match="unbekannter Kern"):
        K.build_kernel(ring4, "Zufallsweg")


# ---- Drehung und Mischung ---------------------------------------------------------------------------------------------------------------------
def test_drehung_mit_p_halb_ist_reversibel_und_mit_p_eins_nicht(ring4):
    assert H.db_deviation(K.rotation_drift(ring4, 0.5), ring4.pi) < 1e-15, "Drehung und Gegendrehung gleich wahrscheinlich: symmetrischer Fluss zwischen Zuständen gleichen Gewichts"
    assert H.stationary_residual(K.rotation_drift(ring4, 0.5), ring4.pi) < 1e-15
    assert H.stationary_residual(K.rotation_drift(ring4, 1.0), ring4.pi) < 1e-15
    assert H.db_deviation(K.rotation_drift(ring4, 1.0), ring4.pi) > 1e-3


def test_mischung_grenzfaelle_q_null_und_q_eins(ring4):
    assert K.mixture(ring4, 0.0, 0.7) == pytest.approx(K.metropolis(ring4), abs=1e-16)
    assert K.mixture(ring4, 1.0, 1.0) == pytest.approx(rotation_matrix(ring4), abs=1e-16)


# ---- Kreisströme ------------------------------------------------------------------------------------------------------------------------------
def test_n3_hat_keinen_kreis_und_der_kreisstrom_meldet_es():
    ring = make_ring(3, 2.0)
    assert len(K.cycle_basis(ring)) == 0
    with pytest.raises(K.NoCycleError, match="Baum"):
        K.cycle_kernels(ring, 0.5, 1.0, 1)


def test_kreisbasis_haengt_nur_von_n_ab_nicht_von_lambda():
    assert np.array_equal(K.cycle_basis(make_ring(5, 1.0)), K.cycle_basis(make_ring(5, 4.0)))


# ---- Prüfer ----------------------------------------------------------------------------------------------------------------------------------
def test_pruefer_an_einer_hand_kette_mit_drei_zustaenden():
    """P mit π = (1/2, 1/4, 1/4) von Hand: reversibel (π₀P₀₁ = π₁P₁₀ = 1/8) und Zeilensumme 1."""
    P = np.array([[0.5, 0.25, 0.25], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5]])
    pi = np.array([0.5, 0.25, 0.25])
    assert H.row_sum_error(P) == 0 and H.min_entry(P) == 0 and H.stationary_residual(P, pi) == 0 and H.db_deviation(P, pi) == 0
    assert H.stationary_solve(P) == pytest.approx(pi, abs=1e-15) and H.n_classes(P) == 1
    assert H.tv(np.array([1.0, 0, 0]), pi) == 0.5


def test_stationaere_verteilung_ist_nicht_eindeutig_bei_zwei_klassen():
    P = np.eye(3)
    assert H.stationary_solve(P) is None and H.n_classes(P) == 3
    assert K.metropolis(make_ring(4, 2.0), 0.0).trace() == pytest.approx(7)       # f = 0: Einheitsmatrix, jede Verteilung stationär


def test_t_mix_nimmt_die_schwelle_inklusive_d_gleich_ein_viertel_genau():
    """Zwei Zustände mit a = b = 1/4: d(t) = (1/2) |1 − 2a|^t = (1/2)(1/2)^t, also d(1) = 1/4 genau (alle Werte in Gleitkomma exakt). t_mix(1/4) = 1."""
    P = np.array([[0.75, 0.25], [0.25, 0.75]])
    pi = np.array([0.5, 0.5])
    assert H.d_curve(P, pi, 3).tolist() == [0.5, 0.25, 0.125, 0.0625]
    assert H.t_mix(P, pi) == 1 and H.t_mix(P, pi, eps=0.2) == 2 and H.t_mix(P, pi, eps=0.6) == 0


def test_zeitumkehr_der_zwei_zustands_kette_mit_drift_nicht_noetig():
    P = np.array([[0.9, 0.1], [0.2, 0.8]])
    pi = np.array([2 / 3, 1 / 3])
    assert H.time_reversal(P, pi) == pytest.approx(P, abs=1e-15), "zwei Zustände: immer reversibel"

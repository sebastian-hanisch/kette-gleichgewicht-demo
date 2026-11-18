"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): kleinster und größter Ring, Gewicht nahe null und sehr groß, Leitwertfaktor null,
Mischung mit q = 0 und q = 1, Schrittzahl 1, Burn-in 0 und an der Grenze, unzulässige Leitwerte."""

import numpy as np
import pytest

import kgw_checks as H
import kgw_constants as C
import kgw_cycle5 as Y
import kgw_evaluation as E
import kgw_kernels as K
from kgw_ring import make_ring, normalisation


def test_ring_mit_weniger_als_drei_plaetzen_ist_nicht_definiert():
    for n in (0, 1, 2):
        with pytest.raises(ValueError, match="erst ab n = 3"):
            make_ring(n, 2.0)
    with pytest.raises(ValueError, match="positiv"):
        make_ring(4, 0.0)


def test_kleinster_ring_n3_vier_zustaende_ein_baum_alle_drei_regeln_haben_pi():
    ring = make_ring(3, 2.0)
    assert ring.size == 4 and ring.n_edges == 3 and ring.pi == pytest.approx(np.array([1, 2, 2, 2]) / 7, abs=1e-15)
    for name in (C.K_METRO, C.K_HEAT, C.K_EQUAL):
        P = K.build_kernel(ring, name)
        assert H.stationary_residual(P, ring.pi) < 1e-15 and H.t_mix(P, ring.pi) is not None
    assert E.verdict_freedom(E.freedom_row(3))[0] == "info"
    assert E.scan_currents(3, 2.0, 0.5, 1.0, 1).no_cycle


def test_groesster_ring_n10_hat_123_zustaende_und_alle_experimente_rechnen_durch():
    ring = make_ring(10, 5.0)
    assert ring.size == 123 == make_ring(10, 0.5).size
    b = E.analyse_build(10, 5.0, C.K_EQUAL)
    assert b.tmix == 13655 and b.residual < 1e-12
    a = E.analyse_arrival(C.A_METRO, 10, 5.0, 0.2, 1.0, C.START_FULL, 100)
    assert a.horizon == min(max(60, 4 * 173), 400) == 400 and a.burn == 100 and len(a.mu) == 401
    assert E.scan_currents(10, 5.0, 0.5, 1.0, 11001).t_rev is not None


def test_gewicht_sehr_klein_und_sehr_gross_gibt_noch_einen_gueltigen_kern():
    for lam in (0.01, 100.0, 1000.0):
        ring = make_ring(5, lam)
        assert ring.pi.sum() == pytest.approx(1.0, abs=1e-12) and ring.pi == pytest.approx(np.array([lam ** int(k) for k in ring.count]) / normalisation(5, lam), rel=1e-10)
        for name in (C.K_METRO, C.K_HEAT):
            P = K.build_kernel(ring, name)
            assert H.row_sum_error(P) < 1e-12 and H.stationary_residual(P, ring.pi) < 1e-10, (lam, name)
    assert ring.pi[ring.count == ring.count.max()].sum() > 0.999, "λ = 1000: fast die ganze Verteilung liegt auf den fünf Maximalbelegungen (zwei belegte Plätze)"


def test_leitwertfaktor_null_gibt_die_einheitsmatrix_jede_verteilung_ist_stationaer():
    ring = make_ring(4, 2.0)
    P = K.metropolis(ring, 0.0)
    assert P == pytest.approx(np.eye(7), abs=1e-16)
    assert H.stationary_solve(P) is None and H.n_classes(P) == 7 and H.t_mix(P, ring.pi) is None
    assert H.stationary_residual(P, np.full(7, 1 / 7)) == pytest.approx(0.0, abs=1e-16)


def test_unzulaessige_leitwerte_geben_eine_meldung_keine_zahl():
    ring = make_ring(4, 2.0)
    zu_gross = np.full(ring.n_edges, 0.5)
    with pytest.raises(ValueError, match="Zeilensumme über 1"):
        K.kernel_from_flows(ring, zu_gross, zu_gross)
    with pytest.raises(ValueError, match="negativ"):
        K.kernel_from_flows(ring, -zu_gross * 0.01, zu_gross * 0.01)


def test_kreis_p_null_komma_fuenf_und_eins_und_ein_schritt():
    for p in (0.5, 1.0):
        assert H.stationary_residual(Y.kernel(p), Y.PI5) < 1e-15
    net = Y.simulate_net_flow(1.0, 1, 11001)
    assert sorted(net.tolist()) == [0.0, 0.0, 0.0, 0.0, 1.0], "p = 1, ein Schritt: genau eine Kante wird einmal vorwärts überquert"
    assert Y.simulate_net_flow(0.5, 1, 3).shape == (5,)
    assert np.abs(Y.simulate_net_flow(1.0, 5, 11001) - 0.2).max() < 1e-15, "p = 1, fünf Schritte: jede Kante genau einmal vorwärts, Nettofluss 1/5"


def test_mischung_burn_in_null_und_burn_in_an_der_grenze():
    a = E.analyse_arrival(C.A_MIX, 6, 2.0, 0.2, 1.0, C.START_EMPTY, 0)
    assert a.tv_burn == pytest.approx(a.mu[0])
    assert a.mu[0] == pytest.approx(1.0 - make_ring(6, 2.0).pi[0], abs=1e-14), "TV(δ_∅, π) = 1 − π(∅)"
    b = E.analyse_arrival(C.A_METRO, 4, 2.0, 0.2, 1.0, C.START_EMPTY, C.BURN_MAX)
    assert b.horizon >= C.BURN_MAX and len(b.mu) == b.horizon + 1 and b.tv_burn == pytest.approx(b.mu[C.BURN_MAX])


def test_alle_startzustaende_sind_gueltig_und_verschieden():
    ring = make_ring(6, 2.0)
    starts = {E.start_state(ring, s) for s in C.START_OPTIONS}
    assert len(starts) == 3 and ring.label(E.start_state(ring, C.START_EMPTY)) == "∅" and ring.label(E.start_state(ring, C.START_ONE)) == "{0}"
    assert ring.count[E.start_state(ring, C.START_FULL)] == 3


def test_reine_drehung_hat_mehrere_klassen_fuer_jedes_n():
    for n in range(3, 11):
        a = E.analyse_arrival(C.A_ROT, n, 2.0, 0.2, 1.0, C.START_EMPTY, 10)
        assert a.classes >= 2 and a.tmix is None and a.d.min() > 0.5, n


def test_zeitanteile_mit_einem_schritt_und_zehntausend_schritten():
    a = E.analyse_time_fractions(4, 2.0, C.K_METRO, 1.0, 1.0, 10_000, 5)
    assert 0 < a.tv < 0.2 and a.noise > 0 and a.ratio == pytest.approx(a.tv / a.noise)

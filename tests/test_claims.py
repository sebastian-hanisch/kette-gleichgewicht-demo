"""Claims-Tests: jede Zahl der README wird hier nachgerechnet (Ganzzahlen und Brüche exakt, Mischzeiten als Ganzzahlen, Zufallsergebnisse mit dem Zufalls-Seed der README).

Die README nennt je Zeile der Tabelle „Befunde“ den Test. Zufallsergebnisse laufen mit dem exakten SplitMix64 (keine C-Bibliothek) und sind auf jeder Plattform gleich; die Tests prüfen
sie trotzdem bis auf die in der README genannte Stelle, nicht bitgenau. Jeder Test prüft auch, dass die Zahl im Wortlaut der README steht (`_steht`).
Messungen, die nicht im Test stehen, nennt die README ausdrücklich („Messung dieser README, nicht im Test“): die Rechenzeiten."""

import re
import subprocess
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import exakt
import kgw_checks as H
import kgw_constants as C
import kgw_cycle5 as Y
import kgw_evaluation as E
import kgw_kernels as K
import kgw_simulation as S
from kgw_ring import make_ring, normalisation, rotation_matrix, rotation_orbits

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _steht(*pieces):
    for p in pieces:
        assert p in README, f"steht nicht in der README: {p}"


def _tmix3(n, lam):
    ring = make_ring(n, lam)
    return tuple(H.t_mix(K.build_kernel(ring, name), ring.pi) for name in (C.K_METRO, C.K_HEAT, C.K_EQUAL))


def test_c1_zustaende_gewichte_normierung():
    r4, r5 = make_ring(4, 2.0), make_ring(5, 2.0)
    assert r4.size == 7 and sorted(round(w) for w in r4.weights()) == [1, 2, 2, 2, 2, 4, 4] and round(1 / r4.pi[0]) == 17
    assert r5.size == 11 and round(1 / r5.pi[0]) == 31
    assert [make_ring(n, 2.0).size for n in range(3, 13)] == [4, 7, 11, 18, 29, 47, 76, 123, 199, 322]
    for n in range(3, 13):
        for lam in (0.5, 1.0, 2.0, 5.0):
            assert float(np.sum(lam ** make_ring(n, lam).count.astype(float))) == pytest.approx(normalisation(n, lam), rel=1e-12)
    _steht("7 Zustände, Gewichte 1, 2, 2, 2, 2, 4, 4, Z = 17", "11 Zustände, Z = 31", "4, 7, 11, 18, 29, 47, 76, 123, 199, 322")


def test_c2_gleiches_pi_andere_zeit():
    tabelle = {(4, 2.0): (15, 22, 18), (6, 2.0): (24, 36, 50), (6, 5.0): (87, 104, 502), (8, 5.0): (127, 153, 2675), (10, 5.0): (173, 208, 13655)}
    for (n, lam), zeiten in tabelle.items():
        assert _tmix3(n, lam) == zeiten, (n, lam)
    _steht("**15 / 22 / 18**", "24 / 36 / 50", "87 / 104 / 502", "127 / 153 / 2675", "173 / 208 / **13 655**", "15, 22 und 18 Schritte", "173, 208 und 13 655 Schritte")


def test_c3_hand_n4_lambda2(ring4, raum4):
    assert K.equal_conductance_value(ring4) == pytest.approx(1 / 68, abs=1e-15)
    assert sorted(set(np.round(K.metropolis_conductance(ring4) * 68, 12))) == [1.0, 2.0]
    d1 = [exakt.d_t(P, raum4.pi, 1) for P in (raum4.metropolis(), raum4.gleiche_leitwerte()[0], raum4.waermebad_nach_regel())]
    assert d1 == [F(10, 17), F(95, 136), F(67, 102)]
    mu = H.mu_curve(K.metropolis(ring4), ring4.pi, 0, 2)
    assert mu == pytest.approx([16 / 17, 9 / 17, 15 / 68], abs=1e-14)
    _steht("c = 1/68", "Metropolis 1/68 und 2/68", "d(1) = 10/17 (Metropolis), 67/102 (Heat-bath), 95/136 (gleiche Leitwerte)", "16/17, 9/17, 15/68")


def test_c4_gemeinsamer_leitwert_gegen_mittleren_metropolis_leitwert():
    for (n, lam), erwartet in {(8, 2.0): 0.302, (10, 5.0): 0.015}.items():
        ring = make_ring(n, lam)
        assert K.equal_conductance_value(ring) / np.mean(K.metropolis_conductance(ring)) == pytest.approx(erwartet, abs=5e-4), (n, lam)
    for (n, lam), erwartet in {(8, 2.0): 0.394, (10, 5.0): 0.091}.items():
        ring = make_ring(n, lam)
        assert np.mean(K.equal_conductance_value(ring) / K.metropolis_conductance(ring)) == pytest.approx(erwartet, abs=5e-4), (n, lam, "Mittel der Quotienten je Kante")
    _steht("0,302fache", "0,015fache", "0,394 und 0,091")


def test_c5_falsch_gebaute_kerne():
    ring = make_ring(6, 2.0)
    erwartet = {C.K_INVERSE: (0.4154, 0.2769), C.K_HEAT_INV: (0.4154, 0.1846), C.K_HALF: (0.1889, 0.0923)}
    for name, (tv, res) in erwartet.items():
        P = K.build_kernel(ring, name)
        assert H.row_sum_error(P) < 1e-14 and H.min_entry(P) >= 0
        assert H.tv(H.stationary_solve(P), ring.pi) == pytest.approx(tv, abs=5e-5) and H.stationary_residual(P, ring.pi) == pytest.approx(res, abs=5e-5), name
    r4 = make_ring(4, 2.0)
    assert H.tv(H.stationary_solve(K.fixed_half(r4)), r4.pi) == pytest.approx(22 / 119, abs=1e-14)
    assert H.tv(H.stationary_solve(K.inverse_ratio(r4)), r4.pi) == pytest.approx(39 / 119, abs=1e-14)
    for n in (4, 5, 6, 8):
        for name in C.KERN_FEHLER:
            assert H.stationary_residual(K.build_kernel(make_ring(n, 1.0), name), make_ring(n, 1.0).pi) < 1e-9, (n, name)
    _steht("**0,4154 / 0,2769**", "0,4154 / 0,1846", "**0,1889 / 0,0923**", "22/119", "39/119", "bei λ = 1 sind alle drei Fehlerfälle wirkungslos")


def test_c6_freiheitsgrade():
    erwartet = {3: (3, 3, 0), 4: (10, 8, 2), 5: (20, 15, 5), 6: (43, 30, 13), 7: (84, 56, 28), 8: (162, 104, 58), 9: (303, 189, 114), 10: (558, 340, 218), 11: (1012, 605, 407),
                12: (1815, 1068, 747)}
    for n, (stat, rev, kreise) in erwartet.items():
        row = E.freedom_row(n)
        assert (row["stationaer"], row["reversibel"], row["kreise"]) == (stat, rev, kreise), n
        assert row["rang_numerisch"] == (n <= 9) and row["connected"]
    assert [(E.freedom_row(n)["voll_stationaer"], E.freedom_row(n)["voll_reversibel"]) for n in (4, 5, 6)] == [(36, 21), (100, 55), (289, 153)]
    assert all(E.freedom_row(n)["voll_numerisch"] for n in range(3, 8)) and not E.freedom_row(8)["voll_numerisch"]
    _steht("n = 3: 3 / 3 / 0 (ein Baum)", "n = 4: **10 / 8 / 2**", "n = 5: 20 / 15 / 5", "n = 6: 43 / 30 / 13", "n = 7: 84 / 56 / 28", "n = 8: 162 / 104 / 58", "n = 9: 303 / 189 / 114",
           "558, 1012 und 1815", "218, 407 und 747", "36 stationäre gegen 21 reversible", "100 gegen 55", "289 gegen 153")


def test_c7_kreis_abweichung_und_kolmogorov():
    assert [round(Y.deviation(p), 4) for p in (0.6, 0.7, 0.9, 1.0)] == [0.04, 0.08, 0.16, 0.2]
    assert Y.kolmogorov(0.7) == pytest.approx((0.16807, 0.00243), abs=1e-15)
    assert F(7, 10) ** 5 == F(16807, 100000) and F(3, 10) ** 5 == F(243, 100000)
    P = Y.kernel(0.7)
    assert H.time_reversal(P, Y.PI5)[1, 0] == pytest.approx(0.7)
    _steht("0,0400 / 0,0800 / 0,1600 / 0,2000", "vorwärts 0,16807 gegen rückwärts 0,00243", "16807/100000 und 243/100000")


@pytest.mark.parametrize("seed,erwartet", [(11001, 0.0799), (11002, 0.0802)])
def test_c8_nettofluss_gemessen(seed, erwartet):
    net = Y.simulate_net_flow(0.7, 1_000_000, seed)
    assert Y.net_flow_exact(0.7) == pytest.approx(0.08, abs=1e-15)
    assert net == pytest.approx(np.full(5, erwartet), abs=5e-5), "alle fünf Kanten auf die vierte Stelle"
    assert np.abs(net - 0.08).max() < 4 * Y.net_flow_sd(1_000_000) and round(Y.net_flow_sd(1_000_000), 4) == 0.0004
    _steht("0,0799 bei Zufalls-Seed 11001 und 0,0802 bei Zufalls-Seed 11002", "obere Schranke der Standardabweichung dient etwa 0,0004")


def test_c9_kreis_ankunft():
    assert [H.t_mix(Y.kernel(p), Y.PI5) for p in (0.5, 0.7, 0.9, 1.0)] == [5, 6, 15, None]
    assert [round(H.second_eigenvalue_abs(Y.kernel(p)), 4) for p in (0.5, 0.7, 0.9, 1.0)] == [0.809, 0.8425, 0.9357, 1.0]
    assert H.d_curve(Y.kernel(1.0), Y.PI5, 200) == pytest.approx(np.full(201, 0.8), abs=1e-12)
    lazy = 0.5 * np.eye(5) + 0.5 * Y.kernel(1.0)
    assert H.t_mix(lazy, Y.PI5) == 5
    assert np.linalg.eigvals(Y.kernel(0.7))[np.argsort(-np.abs(np.linalg.eigvals(Y.kernel(0.7))))][1].imag != 0
    _steht("t_mix = 5 / 6 / 15 / nie", "\\|λ₂\\| = 0,8090 / 0,8425 / 0,9357 / 1", "d(t) = 0,8 für t = 0 bis 200", "t_mix = 5")


def test_c10_drehung_als_zug():
    rows = []
    for n in (4, 5, 6, 7, 8):
        for lam in (2.0, 5.0):
            ring = make_ring(n, lam)
            t0 = H.t_mix(K.metropolis(ring), ring.pi)
            for q in (0.05, 0.2, 0.5):
                rows.append((n, lam, q, t0, {p: H.t_mix(K.mixture(ring, q, p), ring.pi) for p in (0.5, 0.7, 1.0)}))
    assert len(rows) == 30
    gains = [r[3] / t for r in rows for t in r[4].values()]
    assert len(gains) == 90 and round(min(gains), 2) == 1.41 and round(max(gains), 1) == 11.2
    spread = lambda r: max(r[4].values()) - min(r[4].values())
    gerade, ungerade = [spread(r) for r in rows if r[0] % 2 == 0], [spread(r) for r in rows if r[0] % 2 == 1]
    assert (len(gerade), max(gerade), gerade.count(0)) == (18, 1, 17) and (len(ungerade), max(ungerade), ungerade.count(0)) == (12, 2, 5)
    assert [r[4][1.0] for r in rows if r[0] == 6 and r[1] == 5.0] == [27, 32, 51] and next(r[3] for r in rows if r[0] == 6 and r[1] == 5.0) == 87
    _steht("**1,41- bis 11,2fach**", "87 → 27, 32, 51", "höchstens 1 Schritt (17 von 18 Zellen gleich)", "höchstens 2 Schritte (5 von 12 gleich)")


def test_c11_kreisstroeme_zufallsstroeme():
    zeilen = E.anchor_current_rows(11001)
    erwartet = [(5, 2.0, 25, 0, 3), (5, 5.0, 61, 1, 7), (6, 2.0, 48, -2, 0), (6, 5.0, 174, -1, 1), (7, 2.0, 48, 0, 1), (7, 5.0, 137, 1, 2)]
    assert [(r["n"], r["lam"], r["t_rev"], r["min"], r["max"]) for r in zeilen] == erwartet
    assert (min(r["min"] for r in zeilen), max(r["max"] for r in zeilen)) == (-2, 7)
    ueber = [E.anchor_current_rows(s) for s in (11001, 11002, 11003, 11004, 11005)]
    assert (min(r["min"] for z in ueber for r in z), max(r["max"] for z in ueber for r in z)) == (-6, 8)
    s62, s55 = E.scan_currents(6, 2.0, 0.5, 1.0, 11001), E.scan_currents(5, 5.0, 0.5, 1.0, 11001)
    assert sum(1 for d in s62.diffs if d < 0) == 15 and sum(1 for d in s55.diffs if d < 0) == 0 and len(s62.diffs) == len(s55.diffs) == 20
    # gegen reines Metropolis ist in diesen sechs Zeilen und fünf Zufalls-Seeds kein Kreisstrom schneller (README, Negativbefund)
    for z in ueber:
        for r in z:
            assert r["t_rev"] + r["min"] > H.t_mix(K.metropolis(make_ring(r["n"], r["lam"])), make_ring(r["n"], r["lam"]).pi), r
    nicht_darueber = sum(E.scan_currents(n, lam, 0.5, 1.0, 11001).lam2_not_above for n in (5, 6, 7) for lam in (2.0, 5.0))
    assert nicht_darueber == 120
    _steht("n = 5, λ = 2 (t_mix 25): 0 bis +3", "n = 5, λ = 5 (61): +1 bis +7", "n = 6, λ = 2 (48): −2 bis 0", "n = 6, λ = 5 (174): −1 bis +1", "n = 7, λ = 2 (48): 0 bis +1",
           "n = 7, λ = 5 (137): +1 bis +2", "**−2 bis +7 Schritte**", "−6 bis +8", "15 von 20 Strömen kürzer, bei n = 5, λ = 5 keiner", "in 120 von 120 Fällen")


def test_c12_kreisstrom_von_hand():
    h = E.hand_cycle_example()
    assert (h.p_empty_to_0, h.p_empty_to_2, h.p_empty_stay) == pytest.approx((0.25, 0.0, 0.5), abs=1e-15)
    assert h.deviation == pytest.approx(1 / 68, abs=1e-15) and (h.tmix_cycle, h.tmix_base) == (29, 29)
    _steht("s = 1/136", "aus ∅ gehen 1/4 nach {0}, 0 nach {2}, 1/2 bleiben", "Abweichung von Detailed Balance 1/68", "t_mix 29 mit und ohne Kreisstrom")


def test_c13_kern_gegen_ankunft():
    ring = make_ring(6, 2.0)
    t = (0, 1, 2, 5, 10, 20, 40, 60)
    metro = H.mu_curve(K.metropolis(ring), ring.pi, 0, 60)
    mix = H.mu_curve(K.mixture(ring, 0.2, 1.0), ring.pi, 0, 60)
    assert [round(float(metro[i]), 4) for i in t] == [0.9846, 0.8154, 0.3, 0.1163, 0.0513, 0.0125, 0.0008, 0.0]
    assert [round(float(mix[i]), 4) for i in t] == [0.9846, 0.8, 0.48, 0.157, 0.0713, 0.0226, 0.0025, 0.0003]
    assert H.db_deviation(K.metropolis(ring), ring.pi) < 1e-15 and round(H.db_deviation(K.mixture(ring, 0.2, 1.0), ring.pi), 4) == 0.0123
    assert np.all(np.diff(metro) < 0) and np.all(np.diff(mix) < 0), "streng fallend (bis die Rundung greift: 60 Schritte)"
    r4 = exakt.Raum(4, 2)
    mu, tvs = [F(1)] + [F(0)] * 6, []
    for _ in range(6):
        tvs.append(exakt.tv(mu, r4.pi))
        mu = exakt.vm(mu, r4.metropolis())
    assert tvs == [F(16, 17), F(9, 17), F(15, 68), F(69, 544), F(297, 4352), F(1305, 34816)]
    _steht("0,9846 / 0,8154 / 0,3000 / 0,1163 / 0,0513 / 0,0125 / 0,0008 / 0,0000", "0,9846 / 0,8000 / 0,4800 / 0,1570 / 0,0713 / 0,0226 / 0,0025 / 0,0003", "**0,0123**",
           "16/17, 9/17, 15/68, 69/544, 297/4352, 1305/34816")


def test_c14_reine_drehung():
    assert [rotation_orbits(make_ring(n, 2.0)) for n in range(3, 11)] == [2, 3, 3, 5, 5, 8, 10, 15]
    ring = make_ring(5, 2.0)
    R = rotation_matrix(ring)
    _, labels = connected_components(csr_matrix(R > 0), directed=True, connection="strong")
    assert H.n_classes(R) == 3 and sorted(np.bincount(labels).tolist()) == [1, 5, 5]
    assert H.d_curve(R, ring.pi, 40) == pytest.approx(np.full(41, 30 / 31), abs=1e-13) and round(30 / 31, 4) == 0.9677
    neben = np.zeros(11)
    neben[0] = 1 / 3
    neben[1:] = 1 / 15
    assert H.stationary_residual(R, neben) < 1e-15 and not np.allclose(neben, ring.pi)
    _steht("2, 3, 3, 5, 5, 8, 10, 15 für n = 3 bis 10", "3 Klassen (1, 5 und 5 Zustände), d(t) = 30/31 = 0,9677", "⅓ auf ∅, je 1/15")


def test_c15_zeitanteile_einer_kette():
    ring = make_ring(6, 2.0)
    metro, mix = K.metropolis(ring), K.mixture(ring, 0.2, 0.7)
    noise = S.iid_time_fraction_noise(ring.pi, 200_000, 11080, 4)
    assert round(noise, 4) == 0.0037
    tv_m = [H.tv(S.time_fractions(metro, ring.pi, 200_000, s), ring.pi) for s in range(11081, 11085)]
    tv_x = [H.tv(S.time_fractions(mix, ring.pi, 200_000, s), ring.pi) for s in range(11081, 11085)]
    assert [round(v, 4) for v in tv_m] == [0.0142, 0.0112, 0.0165, 0.0137] and [round(v, 4) for v in tv_x] == [0.008, 0.0049, 0.0088, 0.0054]
    assert (round(min(tv_m) / noise, 1), round(max(tv_m) / noise, 1)) == (3.0, 4.4) and (round(min(tv_x) / noise, 1), round(max(tv_x) / noise, 1)) == (1.3, 2.4)
    _steht("0,0142 / 0,0112 / 0,0165 / 0,0137", "0,0080 / 0,0049 / 0,0088 / 0,0054", "0,0037", "3,0- bis 4,4fache", "1,3- bis 2,4fache")


def test_c16_burn_in_aendert_den_kern_nicht():
    """Nach 60 Schritten Burn-in liegt TV(μ_t, π) der nichtreversiblen Mischung bei 0,0003, die Detailed-Balance-Abweichung ihres Kerns bei 0,0123: Burn-in macht die Verteilung
    nach t Schritten gut, den Kern nicht reversibel (die Abweichung ist mehr als das 40fache des Abstands)."""
    ring = make_ring(6, 2.0)
    P = K.mixture(ring, 0.2, 1.0)
    tv60 = H.mu_curve(P, ring.pi, 0, 60)[60]
    assert tv60 < 0.0004 and H.db_deviation(P, ring.pi) > 40 * tv60
    _steht("5000 Burn-in-Schritte stellen Detailed Balance sicher", "ist falsch")


def test_readme_nennt_die_testzahl_der_suite():
    ausgabe = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True,
                             stdin=subprocess.DEVNULL, timeout=600).stdout
    zahl = int(re.search(r"(\d+) tests? collected", ausgabe).group(1))
    treffer = re.findall(r"\*\*(\d+) Tests\*\*", README)
    assert treffer and int(treffer[0]) == zahl, (treffer, zahl)


def test_readme_nennt_jede_zahl_der_tabelle_den_live_link_als_marke_und_alle_nachbar_links():
    assert "TODO nach dem Deploy" in README and "sebastianhanisch-kette-gleichgewicht-demo.streamlit.app" in README
    for repo in ("exakt-oder-abtasten-demo", "metropolis-kleiner-raum-demo", "gibbs-glauber-demo", "mischzeit-exakt-demo", "posterior-sampling-demo", "markov-queue-demo"):
        assert f"https://github.com/sebastian-hanisch/{repo})" in README, repo
    tests_genannt = set(re.findall(r"`(test_c\d+)`", README))
    vorhanden = set(re.findall(r"^def (test_c\d+)_", Path(__file__).read_text(encoding="utf-8"), flags=re.M))
    assert tests_genannt and tests_genannt <= vorhanden, tests_genannt - vorhanden

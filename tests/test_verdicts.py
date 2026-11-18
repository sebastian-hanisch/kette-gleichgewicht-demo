"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Auswertungen, nicht Zufalls-Seeds; die Auswertungen werden hier von Hand gebaut."""

import numpy as np

import kgw_constants as C
import kgw_evaluation as E


def _build(kern=C.K_METRO, residual=0.0, db=0.0, tmix=15, tv=None, n=4, lam=2.0, size=7, refs=None):
    return E.BuildAnalysis(n=n, lam=lam, kern=kern, size=size, labels=(), P=np.eye(2), pi=np.ones(2) / 2, row_err=0.0, min_entry=0.0, residual=residual,
                           residual_vec=np.zeros(2), db=db, stationary=None, stationary_tv=tv, tmix=tmix,
                           refs=refs or {C.K_METRO: 15, C.K_HEAT: 22, C.K_EQUAL: 18})


def test_build_ja_nennt_kern_einstellung_mischzeit_und_die_drei_vergleichszeiten():
    kind, text = E.verdict_build(_build())
    assert kind == "success"
    for piece in ("**π ist stationär**", "Metropolis, n = 4, λ = 2, 7 Zustände", "**15 Schritte**", "Metropolis 15, Heat-bath 22 und gleiche Leitwerte 18 Schritte",
                  "Detailed-Balance-Abweichung unter 10⁻⁹"):
        assert piece in text, piece


def test_build_schwelle_fuer_pi_stationaer_ist_1e_minus_9_inklusive():
    assert E.verdict_build(_build(residual=1e-9))[0] == "success"
    assert E.verdict_build(_build(residual=2e-9, kern=C.K_HALF, tv=0.1))[0] == "warning"


def test_build_fehlerfall_nennt_abstand_und_die_gleichverteilung_nur_bei_fester_annahme():
    kind, text = E.verdict_build(_build(kern=C.K_HALF, residual=0.0923, tv=0.1889, n=6, size=18))
    assert kind == "warning" and "‖πP − π‖₁ = 0,0923" in text and "beträgt 0,1889" in text and "Stationär ist die Gleichverteilung." in text
    _, text = E.verdict_build(_build(kern=C.K_INVERSE, residual=0.2769, tv=0.4154, n=6, size=18))
    assert "Gleichverteilung" not in text and "0,2769" in text and "0,4154" in text


def test_build_fehlerfall_bei_lambda_eins_ist_wirkungslos_und_meldet_keinen_fehler():
    kind, text = E.verdict_build(_build(kern=C.K_HEAT_INV, residual=1e-16, lam=1.0, n=6, size=18))
    assert kind == "info" and "wirkungslos" in text and "mit λ = 1 sind alle Gewichte gleich" in text
    kind, text = E.verdict_build(_build(kern=C.K_HALF, residual=1e-16, lam=2.0))
    assert kind == "info" and "mit λ = 1" not in text


def test_build_nie_wenn_die_kette_nicht_ankommt():
    assert "kommt die Kette innerhalb von 2²² Schritten **nicht an**" in E.verdict_build(_build(tmix=None))[1]
    assert "braucht die Kette **15 Schritte**" in E.verdict_build(_build())[1]


def test_zeitanteile_verdikt_beide_zweige():
    a = E.TimeFractionAnalysis(steps=50_000, seed=11001, tv=0.0234, noise=0.0075, ratio=0.0234 / 0.0075, stationary_tv=0.0)
    kind, text = E.verdict_time_fractions(a, 0.0)
    assert kind == "info" and "50 000 Schritte" in text and "**0,0234**" in text and "0,0075" in text and "das 3,1fache" in text and "Zufalls-Seed 11001" in text
    kind, text = E.verdict_time_fractions(E.TimeFractionAnalysis(50_000, 11001, 0.1937, 0.0075, 25.8, 0.1889), 0.0923)
    assert kind == "warning" and "0,1937" in text and "0,1889" in text and "nicht zu π" in text


def _cycle(p=0.7, mean=0.0799, band_ok=True, tmix=6):
    return E.Cycle5Analysis(p=p, steps=1_000_000, seed=11001, deviation=abs(2 * p - 1) / 5, kol_fwd=p ** 5, kol_bwd=(1 - p) ** 5, net_exact=(2 * p - 1) / 5,
                            net_measured=np.full(5, mean), sd=0.00045, max_dev=0.0001 if band_ok else 0.01, band_ok=band_ok, tmix=tmix, lam2=0.8425,
                            P=np.eye(5), P_star=np.eye(5), stationary_residual=0.0)


def test_kreis_p_0_7_nennt_abweichung_nettofluss_exakt_und_gemessen_und_die_produkte():
    kind, text = E.verdict_cycle5(_cycle())
    assert kind == "success"
    for piece in ("p = 0,7", "Abweichung 0,0800", "exakt 0,0800", "gemessen 0,0799", "Zufalls-Seed 11001", "1 000 000 Schritte", "0,16807 gegen rückwärts 0,00243",
                  "hinreichend, nicht notwendig"):
        assert piece in text, piece


def test_kreis_messung_ausserhalb_des_bandes_ist_eine_warnung():
    kind, text = E.verdict_cycle5(_cycle(band_ok=False))
    assert kind == "warning" and "außerhalb von ±4 Standardabweichungen" in text and "0,0100" in text


def test_kreis_p_halb_ist_reversibel_und_p_eins_eine_reine_drehung():
    kind, text = E.verdict_cycle5(_cycle(p=0.5, mean=0.0001))
    assert kind == "info" and "reversibel" in text and "0,03125" in text
    kind, text = E.verdict_cycle5(_cycle(p=1.0, mean=0.2, tmix=None))
    assert kind == "success" and "reine Drehung" in text and "exakt 0,2000" in text and "Produkt vorwärts 1,00000 gegen rückwärts 0,00000" in text


def _row(n, V, E_, kreise, stat, rev, numeric=True, vs=36, vr=21):
    return {"n": n, "V": V, "E": E_, "kreise": kreise, "stationaer": stat, "reversibel": rev, "rang_numerisch": numeric, "voll_stationaer": vs, "voll_reversibel": vr}


def test_freiheitsgrade_n4_zehn_gegen_acht_zwei_kreise_und_36_gegen_21():
    kind, text = E.verdict_freedom(_row(4, 7, 8, 2, 10, 8))
    assert kind == "success"
    for piece in ("n = 4: 7 Zustände, 8 Kanten", "10 stationäre Freiheitsgrade", "Rang der Erhaltungsgleichungen gerechnet", "8 reversible", "2 Kreise", "36 gegen 21"):
        assert piece in text, piece


def test_freiheitsgrade_n3_ist_ein_baum_und_grosse_n_nennen_die_formel():
    kind, text = E.verdict_freedom(_row(3, 4, 3, 0, 3, 3, vs=9, vr=6))
    assert kind == "info" and "Baum" in text and "jede stationäre Kette auf diesem Graphen ist reversibel" in text and "9 gegen 6" in text
    _, text = E.verdict_freedom(_row(10, 123, 340, 218, 558, 340, numeric=False, vs=14884, vr=7503))
    assert "nach der Formel 2E − (V − 1), Rang nicht gerechnet" in text and "558 stationäre" in text


def _scan(diffs, l2_rev=0.9, l2_cur=None, no_cycle=False):
    l2_cur = l2_cur or tuple([0.89] * len(diffs))
    return E.CurrentScan(n=5, lam=2.0, factor=0.5, share=1.0, seed=11001, no_cycle=no_cycle, t_rev=25, t_cur=tuple(25 + d for d in diffs), diffs=tuple(diffs), lam2_rev=l2_rev,
                         lam2_cur=l2_cur, lam2_not_above=sum(1 for v in l2_cur if round(v, 4) <= round(l2_rev, 4)), smax=0.1, db_first=0.01, P_first=None, P_rev=np.eye(2))


def test_kreisstroeme_gemischt_nennt_spanne_mit_vorzeichen_und_zaehlt_die_faelle():
    kind, text = E.verdict_currents(_scan([-3, -1, 0, 2, 6] * 4))
    assert kind == "info"
    for piece in ("**−3 bis +6 Schritte**", "kürzer in 8, länger in 8, gleich in 4 Fällen", "20 zufällige Kreisströme", "|λ₂| ist in 20 von 20", "t_mix 25 Schritte",
                  "nicht von selbst schneller"):
        assert piece in text, piece


def test_kreisstroeme_nur_langsamer_nur_schneller_und_baum():
    assert "Kein Kreisstrom ist hier schneller" in E.verdict_currents(_scan([0, 1, 2]))[1]
    _, text = E.verdict_currents(_scan([-1, -2, -3]))
    assert "Jeder dieser Kreisströme ist hier schneller" in text and "nicht von selbst schneller" not in text
    _, text = E.verdict_currents(_scan([], no_cycle=True))
    assert "Baum" in text and "keine nichtreversible Kette" in text


def _arrival(chain=C.A_METRO, db=0.0, tmix=24, classes=1, p=None, q=None, mu=None, d=None, tmix_metro=None, tmix_half=None, horizon=60):
    mu = np.linspace(1, 0, horizon + 1) if mu is None else mu
    d = np.linspace(1, 0, horizon + 1) if d is None else d
    return E.ArrivalAnalysis(chain=chain, n=6, lam=2.0, start_label="leerer Zustand: ∅", size=18, horizon=horizon, d=d, mu=mu, db=db, residual=0.0, tmix=tmix, classes=classes,
                             burn=10, tv_burn=float(mu[10]), p=p, q=q, tmix_metro=tmix_metro, tmix_p_half=tmix_half)


def test_ankunft_metropolis_nennt_kernabweichung_null_verteilung_nach_t_und_burn_in():
    mu = np.zeros(61)
    mu[[0, 2, 10, 40]] = [0.9846, 0.3, 0.0513, 0.0008]
    kind, text = E.verdict_arrival(_arrival(mu=mu))
    assert kind == "success"
    for piece in ("Metropolis (n = 6, λ = 2, Start leerer Zustand: ∅)", "Detailed-Balance-Abweichung des Kerns **0** bei jedem t", "0,9846 / 0,3000 / 0,0513 / 0,0008 bei t = 0 / 2 / 10 / 40",
                  "nach 10 Schritten Burn-in 0,0513", "t_mix = 24 Schritte", "nicht den Kern"):
        assert piece in text, piece


def test_ankunft_mischung_nennt_gewinn_und_kontrolle_und_die_drift_als_nicht_die_ursache():
    kind, text = E.verdict_arrival(_arrival(chain=C.A_MIX, db=0.0123, tmix=11, p=1.0, q=0.2, tmix_metro=24, tmix_half=11))
    assert kind == "success" and "**0,0123** bei jedem t" in text and "t_mix 24 gegen 11" in text and "p = 0,5 (reversibel, gleiche Drehung): t_mix 11" in text
    assert "nicht von der Drift" in text


def test_ankunft_reine_drehung_nennt_klassen_und_die_untere_schranke_von_d():
    d = np.full(61, 30 / 31)
    kind, text = E.verdict_arrival(_arrival(chain=C.A_ROT, tmix=None, classes=3, d=d, mu=d.copy()))
    assert kind == "warning" and "**3 Klassen**" in text and "d(t) ≥ 0,9677 für alle t = 0 bis 60" in text and "leerer Zustand: ∅ 0,9677" in text


def test_ankunft_kreis_p_eins_kommt_nie_an_und_p_0_7_nach_6_schritten():
    d = np.full(61, 0.8)
    kind, text = E.verdict_arrival(_arrival(chain=C.A_CYCLE5, db=0.2, tmix=None, p=1.0, d=d, mu=d.copy()))
    assert kind == "warning" and "d(t) = 0,8000 für alle t = 0 bis 60" in text and "kommt nie an" in text and "1 Klasse" in text
    mu = np.zeros(61)
    mu[[0, 2, 10, 40]] = [0.8, 0.51, 0.111, 0.0007]
    kind, text = E.verdict_arrival(_arrival(chain=C.A_CYCLE5, db=0.08, tmix=6, p=0.7, mu=mu))
    assert kind == "success" and "t_mix = **6 Schritte**" in text and "0,0800 bei jedem t" in text and "0,8000 / 0,5100 / 0,1110 / 0,0007" in text


def test_kernfragen_vier_antworten_mit_den_zahlen_der_auswertungen():
    build = _build()
    fragen = E.kernfragen(build, _cycle(), {"n": 4, "stationaer": 10, "reversibel": 8, "kreise": 2}, _arrival())
    assert [q for q, _ in fragen] == ["Hat diese Kette π als stationäre Verteilung?", "Erfüllt sie Detailed Balance, und muss sie?", "Wie viele Freiheitsgrade haben die Ketten, die zu π passen?",
                                      "Kommt sie an, und wie schnell?"]
    answers = [a for _, a in fragen]
    assert answers[0].startswith("Ja:") and "Metropolis, n = 4, λ = 2" in answers[0]
    assert "Ja, Detailed Balance gilt" in answers[1] and "p = 0,7" in answers[1] and "Abweichung 0,0800" in answers[1]
    assert "n = 4 Plätzen: 10 stationäre Freiheitsgrade, davon 8 reversible" in answers[2] and "Unterschied 2" in answers[2]
    assert "t_mix = 24 Schritte" in answers[3] and "Klassen" not in answers[3]
    nein = E.kernfragen(_build(kern=C.K_HALF, residual=0.0923, db=0.005, tv=0.1889), _cycle(), {"n": 4, "stationaer": 10, "reversibel": 8, "kreise": 2},
                        _arrival(classes=3))[0][1]
    assert nein.startswith("Nein:") and "0,0923" in nein and "0,1889" in nein

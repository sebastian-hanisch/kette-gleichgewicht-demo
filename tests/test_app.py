"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Button, fährt jeden Slider an seine Grenzen und prüft,
dass kein Python-Fehler auftritt (insbesondere `StreamlitDuplicateElementId` bei Diagrammen ohne eindeutiges key= und `StreamlitAPIException` bei Reglern mit Grenzen min == max).
Dazu: je Preset die erwartete Verdikt-Zeile (ein Preset ohne Effekt darf keinen Effekt melden), Fehlerfall bei λ = 1, Permalink, Fehlerseite bei ungültigen Werten."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import kgw_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "▶"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app(**query) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    for k, v in query.items():
        at.query_params[k] = v
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texte(at) -> list:
    return [e.value for kind in (at.success, at.info, at.warning, at.error) for e in kind]


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


# ein ausgegrauter Regler lässt sich (wie im Browser) erst bedienen, wenn seine Wahl ihn freigibt
FREIGABE = {"f0_slider": ("kern_select", C.K_STAGED), "f1_slider": ("kern_select", C.K_STAGED), "q_slider": ("arrival_select", C.A_MIX), "p4_slider": ("arrival_select", C.A_MIX)}


def test_every_slider_at_its_min_and_max_does_not_raise():
    keys = [s.key for s in _fresh_app().slider]
    assert keys, "keine Slider gefunden"
    for key in keys:
        for edge in ("min", "max"):
            at = _fresh_app()
            if key in FREIGABE:
                at.selectbox(key=FREIGABE[key][0]).select(FREIGABE[key][1]).run(timeout=TIMEOUT)
            slider = at.slider(key=key)
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (key, edge, [str(e) for e in at.exception])


def test_every_choice_of_a_selectbox_and_select_slider_does_not_raise():
    probe = _fresh_app()
    jobs = [("selectbox", w.key, list(w.options)) for w in probe.selectbox] + [("select_slider", w.key, list(w.options)) for w in probe.select_slider]
    assert {key for _, key, _ in jobs} == {"kern_select", "arrival_select", "start_select", "time_steps_select", "steps_select"}
    for art, key, options in jobs:
        for option in options:
            at = _fresh_app()
            if art == "selectbox":
                at.selectbox(key=key).select(option).run(timeout=TIMEOUT)
            else:
                at.select_slider(key=key).set_value(int(option.replace(" ", ""))).run(timeout=TIMEOUT)   # Optionen kommen formatiert („1 000 000“), der Wert ist die Zahl
            assert not at.exception, (key, option, [str(e) for e in at.exception])


# je Preset die erwartete Verdikt-Zeile (Anker der README)
PRESET_ERWARTUNG = {
    "Metropolis (n = 4, λ = 2)": ("success", ["π ist stationär", "Metropolis, n = 4, λ = 2, 7 Zustände", "15 Schritte", "Heat-bath 22 und gleiche Leitwerte 18 Schritte"]),
    "Standardfall (n = 8, λ = 2)": ("success", ["Metropolis, n = 8, λ = 2, 47 Zustände", "35 Schritte"]),
    "Kalt (n = 8, λ = 5)": ("success", ["Metropolis, n = 8, λ = 5, 47 Zustände", "127 Schritte", "Heat-bath 153 und gleiche Leitwerte 2675 Schritte"]),
    "Gleiche Leitwerte (n = 10, λ = 5)": ("success", ["Gleiche Leitwerte, n = 10, λ = 5, 123 Zustände", "13 655 Schritte", "Metropolis 173"]),
    "Heat-bath (n = 6, λ = 2)": ("success", ["Heat-bath, n = 6, λ = 2, 18 Zustände", "36 Schritte"]),
    "Falsch gebaut: feste Annahme ½ (n = 6, λ = 2)": ("warning", ["Fällt durch die Prüfung", "‖πP − π‖₁ = 0,0923", "Stationär ist die Gleichverteilung", "0,1889"]),
    "5-Kreis mit Drift 0,7": ("success", ["p = 0,7: Gleichverteilung stationär", "Abweichung 0,0800", "exakt 0,0800, gemessen 0,0799", "1 000 000 Schritte"]),
    "Reine Drehung (n = 5, λ = 2)": ("warning", ["Reine Drehung (n = 5, λ = 2)", "3 Klassen", "d(t) ≥ 0,9677"]),
    "Kreisstrom (n = 5, λ = 2)": ("info", ["n = 5, λ = 2, reversibler Kern", "t_mix 25 Schritte", "20 zufällige Kreisströme"]),
    "Kern gegen Burn-in (n = 6, λ = 2)": ("success", ["Metropolis mit Drehung und Drift (n = 6, λ = 2", "**0,0123** bei jedem t", "0,9846 / 0,4800 / 0,0713 / 0,0025"]),
}


def test_jedes_preset_hat_eine_erwartete_zeile():
    assert set(PRESET_ERWARTUNG) == set(C.PRESETS)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_zeigt_die_erwartete_verdikt_zeile(name):
    at = _fresh_app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    kind, pieces = PRESET_ERWARTUNG[name]
    texts = [e.value for e in getattr(at, kind)]
    assert any(all(p in t for p in pieces) for t in texts), (name, pieces, [t[:80] for t in texts])


def test_fehlerfall_bei_lambda_eins_meldet_keinen_effekt():
    at = _fresh_app(kern=C.K_HEAT_INV, lam="1.0", n="6")
    assert any("wirkungslos" in e.value for e in at.info)
    assert not any("Fällt durch die Prüfung" in e.value for e in at.warning)


def test_kernfragen_nennen_vier_antworten():
    at = _fresh_app()
    for frage in ("Hat diese Kette π als stationäre Verteilung?", "Erfüllt sie Detailed Balance, und muss sie?", "Wie viele Freiheitsgrade haben die Ketten, die zu π passen?", "Kommt sie an, und wie schnell?"):
        assert any(frage in m.value for m in at.markdown), frage


def test_permalink_uebernimmt_werte_begrenzt_und_ignoriert_ungueltiges():
    at = _fresh_app(n="7", lam="5", kern=C.K_HEAT, p5="0.9", ank=C.A_ROT)
    assert at.slider(key="n_slider").value == 7 and at.slider(key="lam_slider").value == 5.0 and at.selectbox(key="kern_select").value == C.K_HEAT
    assert at.slider(key="p5_slider").value == pytest.approx(0.9) and at.selectbox(key="arrival_select").value == C.A_ROT
    at = _fresh_app(n="99", kern="Zufallsweg", lam="abc", t="12345")
    assert at.slider(key="n_slider").value == C.N_MAX and at.selectbox(key="kern_select").value == C.K_METRO
    assert at.slider(key="lam_slider").value == C.DEFAULT_LAM and at.select_slider(key="steps_select").value == C.DEFAULT_STEPS


def test_zufalls_seed_aendert_die_zufallsergebnisse_und_gleicher_seed_nicht():
    a = _fresh_app(seed="11001", t="10000")
    b = _fresh_app(seed="11001", t="10000")
    c = _fresh_app(seed="11002", t="10000")
    nettofluss = lambda at: next(m.value for m in at.metric if m.label == "Nettofluss (gemessen, Mittel)")
    assert nettofluss(a) == nettofluss(b) != nettofluss(c)

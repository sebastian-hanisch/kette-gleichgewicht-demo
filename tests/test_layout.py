"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben, Kernfragen vor dem Expander, Regler nur dort ausgegraut, wo sie wirkungslos sind.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset). Der Test unten stellt sicher, dass wirklich JEDER Regler
(Seitenleiste und Hauptbereich) dort steht: sonst bliebe das Preset trotz Änderung markiert."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import kgw_constants as C
import kgw_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
FREIGABE = {"f0_slider": ("kern_select", C.K_STAGED), "f1_slider": ("kern_select", C.K_STAGED), "q_slider": ("arrival_select", C.A_MIX), "p4_slider": ("arrival_select", C.A_MIX)}
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input")


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    """Button-Typ je Preset-Name (die Preset-Buttons der Seitenleiste)."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    for container in (at.sidebar, at.main):
        for art in REGLER_ARTEN:
            try:
                return getattr(container, art)(key=key)
            except KeyError:
                continue
    raise AssertionError(f"Regler {key} steht weder in der Seitenleiste noch im Hauptbereich")


def _reihenfolge(at) -> list:
    out = []
    for el in at.main.children.values():
        if type(el).__name__ in ("Markdown", "Subheader", "Expander"):
            out.append(getattr(el, "value", None) or getattr(el, "label", ""))
    return out


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)  # die ersten Buttons der Seitenleiste
    assert not [b for b in at.main.button if b.label in C.PRESETS]


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))  # das Stylesheet setzt an der Klasse des Containers an
    assert at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    typen = _typen(_app())
    assert [name for name, typ in typen.items() if typ == "primary"] == [next(iter(C.PRESETS))]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]
    at.run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]


def test_jeder_regler_steht_in_setting_specs_und_jeder_eintrag_hat_einen_regler():
    at = _app()
    gezeichnet = {w.key for container in (at.sidebar, at.main) for art in REGLER_ARTEN for w in getattr(container, art) if w.key}
    assert gezeichnet == set(P.SETTING_SPECS), f"fehlt in SETTING_SPECS: {sorted(gezeichnet - set(P.SETTING_SPECS))}; ohne Regler: {sorted(set(P.SETTING_SPECS) - gezeichnet)}"


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_auf(state_key):
    at = _app()
    spec = P.SETTING_SPECS[state_key]
    if state_key in FREIGABE:                      # ein ausgegrauter Regler lässt sich (wie im Browser) erst bedienen, wenn seine Wahl ihn freigibt
        _regler(at, FREIGABE[state_key][0]).set_value(FREIGABE[state_key][1]).run()
        assert not _regler(at, state_key).proto.disabled
    regler = _regler(at, state_key)
    if spec.options is not None:
        neu = next(o for o in spec.options if o != regler.value)
    else:
        neu = spec.hi if regler.value != spec.hi else spec.lo
    regler.set_value(neu).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "primary" not in _typen(at).values()
    # erneutes Anklicken markiert wieder
    name = next(iter(C.PRESETS))
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert _typen(at)[name] == "primary"


def test_kernfragen_stehen_vor_den_experimenten_und_dem_formel_expander():
    reihenfolge = _reihenfolge(_app())
    erg = next(i for i, t in enumerate(reihenfolge) if t.startswith("## 🎯 Ergebnis"))
    kern = next(i for i, t in enumerate(reihenfolge) if t.startswith("## ❓ Kernfragen"))
    exp = [i for i, t in enumerate(reihenfolge) if t.startswith("🔬 Experiment")]
    grenzen = next(i for i, t in enumerate(reihenfolge) if t.startswith("🚧 Wo die Annahmen enden"))
    formel = next(i for i, t in enumerate(reihenfolge) if t == "📐 Mathematische Formulierung")
    assert erg < kern < min(exp) and len(exp) == 4 and max(exp) < grenzen < formel


def test_regler_sind_nur_dort_ausgegraut_wo_sie_wirkungslos_sind():
    at = _app()
    aus = lambda key: _regler(at, key).proto.disabled
    # Standard: Kern Metropolis, Kette Metropolis
    assert aus("f0_slider") and aus("f1_slider") and aus("q_slider") and aus("p4_slider") and not aus("start_select")
    at = _app()
    _regler(at, "kern_select").set_value(C.K_STAGED).run()
    assert not _regler(at, "f0_slider").proto.disabled and not _regler(at, "f1_slider").proto.disabled
    _regler(at, "n_slider").set_value(3).run()
    assert not _regler(at, "f0_slider").proto.disabled and _regler(at, "f1_slider").proto.disabled, "n = 3 hat keine Stufe über k = 1: f₁ ohne Wirkung"
    _regler(at, "arrival_select").set_value(C.A_MIX).run()
    assert not _regler(at, "q_slider").proto.disabled and not _regler(at, "p4_slider").proto.disabled
    _regler(at, "arrival_select").set_value(C.A_CYCLE5).run()
    assert _regler(at, "q_slider").proto.disabled and not _regler(at, "p4_slider").proto.disabled and _regler(at, "start_select").proto.disabled

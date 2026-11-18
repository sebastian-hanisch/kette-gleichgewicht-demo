"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Alle Regler stehen in SETTING_SPECS (auch die im Hauptbereich beim Experiment): `aktives_preset` vergleicht jeden mit dem Preset, und die Adresszeile
spiegelt die ganze Konfiguration. Ein Preset nennt nur Regler, die vom Standard abweichen; alle übrigen gelten dort mit ihrem Standardwert."""

import math
import random
from dataclasses import dataclass

import streamlit as st

import kgw_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None


SETTING_SPECS = {
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "lam_slider": SettingSpec("lam", float, C.DEFAULT_LAM, C.LAM_MIN, C.LAM_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "kern_select": SettingSpec("kern", str, C.K_METRO, options=C.KERN_OPTIONS),
    "f0_slider": SettingSpec("f0", float, C.DEFAULT_F, C.F_MIN, C.F_MAX),
    "f1_slider": SettingSpec("f1", float, C.DEFAULT_F, C.F_MIN, C.F_MAX),
    "time_steps_select": SettingSpec("zt", int, C.DEFAULT_TIME_STEPS, options=C.TIME_STEPS_OPTIONS),
    "p5_slider": SettingSpec("p5", float, C.DEFAULT_P5, C.P5_MIN, C.P5_MAX),
    "steps_select": SettingSpec("t", int, C.DEFAULT_STEPS, options=C.STEPS_OPTIONS),
    "share_slider": SettingSpec("s", float, C.DEFAULT_S, C.S_MIN, C.S_MAX),
    "fbase_slider": SettingSpec("fb", float, C.DEFAULT_FB, C.FB_MIN, C.FB_MAX),
    "arrival_select": SettingSpec("ank", str, C.A_METRO, options=C.ARRIVAL_OPTIONS),
    "q_slider": SettingSpec("q", float, C.DEFAULT_Q, C.Q_MIN, C.Q_MAX),
    "p4_slider": SettingSpec("p4", float, C.DEFAULT_P4, C.P4_MIN, C.P4_MAX),
    "start_select": SettingSpec("st", str, C.START_EMPTY, options=C.START_OPTIONS),
    "burn_slider": SettingSpec("burn", int, C.DEFAULT_BURN, C.BURN_MIN, C.BURN_MAX),
}
# Preset-Schlüssel -> Regler
PRESET_KEYS = {
    "n": "n_slider", "lam": "lam_slider", "seed": "seed_input", "kern": "kern_select", "f0": "f0_slider", "f1": "f1_slider", "time_steps": "time_steps_select",
    "p5": "p5_slider", "steps": "steps_select", "share": "share_slider", "fbase": "fbase_slider", "arrival": "arrival_select", "q": "q_slider", "p4": "p4_slider",
    "start": "start_select", "burn": "burn_slider",
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _clean(spec: SettingSpec, raw):
    """Wert aus der Adresszeile in den Typ des Reglers, begrenzt; ungültig → None."""
    try:
        value = spec.caster(raw)
    except (ValueError, TypeError):
        return None
    if spec.options is not None:
        return value if value in spec.options else None
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            value = _clean(spec, qp[spec.url_param])
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def preset_value(name: str, state_key: str):
    """Wert, den das Preset für den Regler setzt (Standardwert, wenn es ihn nicht nennt)."""
    key_of = {v: k for k, v in PRESET_KEYS.items()}
    return C.PRESETS[name].get(key_of[state_key], SETTING_SPECS[state_key].default)


def apply_preset(name):
    for state_key in SETTING_SPECS:
        st.session_state[state_key] = preset_value(name, state_key)


def _same(a, b) -> bool:
    """Gleichheit mit Rundungstoleranz für Zahlen mit Nachkommastellen (ein Regler mit Schrittweite liefert 0,7000000000000001 statt 0,7)."""
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return a == b


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State."""
    for name in C.PRESETS:
        if all(_same(st.session_state.get(state_key, spec.default), preset_value(name, state_key)) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden (setzt alle Regler):")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)

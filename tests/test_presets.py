"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink begrenzt und lehnt ungültige Werte ab; Standardwerte stimmen mit den Settings überein."""

import dataclasses

import kgw_constants as C
import kgw_evaluation as ev
import kgw_presets as P


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_bounds_or_options():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[P.PRESET_KEYS[key]]
            if spec.options is not None:
                assert value in spec.options, (name, key)
            else:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_the_first_preset_is_the_default_and_no_other_preset_equals_it():
    names = list(C.PRESETS)
    assert C.PRESETS[names[0]] == {}
    values = {name: tuple(P.preset_value(name, k) for k in P.SETTING_SPECS) for name in names}
    assert len(set(values.values())) == len(names), "jedes Preset setzt eine andere Kombination: sonst wäre die Hervorhebung mehrdeutig"


def test_setting_specs_cover_every_preset_key_and_every_default_matches_the_settings():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    settings = dataclasses.asdict(ev.Settings())
    for key, state_key in P.PRESET_KEYS.items():
        assert settings[key] == P.SETTING_SPECS[state_key].default, key


def test_defaults_lie_inside_their_own_bounds():
    for state_key, spec in P.SETTING_SPECS.items():
        if spec.options is not None:
            assert spec.default in spec.options, state_key
        else:
            assert spec.lo <= spec.default <= spec.hi, state_key


def test_permalink_values_are_clipped_and_invalid_values_ignored():
    n = P.SETTING_SPECS["n_slider"]
    assert P._clean(n, "99") == C.N_MAX and P._clean(n, "-4") == C.N_MIN and P._clean(n, "abc") is None
    kern = P.SETTING_SPECS["kern_select"]
    assert P._clean(kern, C.K_HEAT) == C.K_HEAT and P._clean(kern, "Zufallsweg") is None
    steps = P.SETTING_SPECS["steps_select"]
    assert P._clean(steps, "1000000") == 1_000_000 and P._clean(steps, "12345") is None
    assert P._clean(P.SETTING_SPECS["lam_slider"], "7") == C.LAM_MAX


def test_floats_equal_up_to_rounding_count_as_the_same_setting():
    assert P._same(0.7000000000000001, 0.7) and not P._same(0.75, 0.7) and P._same(4, 4) and not P._same("a", "b")


def test_the_default_preset_names_the_example_by_hand():
    first = next(iter(C.PRESETS))
    assert "n = 4" in first and "λ = 2" in first

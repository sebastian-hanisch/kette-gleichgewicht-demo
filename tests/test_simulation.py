"""Zufall: der kopierte SplitMix64 (bekannte Werte), die vektorisierte Fassung gegen den schrittweisen Generator, Zeitanteile und Nettofluss gegen eine einfache Schleife."""

import numpy as np
import pytest

import kgw_checks as H
import kgw_cycle5 as Y
import kgw_kernels as K
import kgw_simulation as S
from kgw_ring import make_ring
from kgw_rng import SplitMix64


def test_splitmix64_bekannte_werte_fuer_seed_null():
    """Referenzwerte des SplitMix64 (Vigna) für den Zustand 0: die ersten drei Ausgaben."""
    rng = SplitMix64(0)
    assert [rng.next_u64() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]


@pytest.mark.parametrize("seed", [0, 1, 11001, 2 ** 63 + 5, 2 ** 64 - 1])
def test_vektorisierter_strom_ist_der_schrittweise_generator(seed):
    rng = SplitMix64(seed)
    ref = [rng.next_u64() for _ in range(500)]
    assert [int(x) for x in S.u64_stream(seed, 500)] == ref
    rng = SplitMix64(seed)
    assert S.uniforms(seed, 500).tolist() == [rng.random() for _ in range(500)]


def test_nettofluss_gegen_eine_einfache_schleife():
    for p, seed, steps in ((0.7, 11001, 5000), (0.5, 3, 4000), (1.0, 9, 100), (0.9, 2 ** 64 - 1, 3000)):
        rng = SplitMix64(seed)
        x = rng.below(5)
        up, down = [0] * 5, [0] * 5
        for _ in range(steps):
            if rng.random() < p:
                up[x] += 1
                x = (x + 1) % 5
            else:
                x = (x - 1) % 5
                down[x] += 1
        assert Y.simulate_net_flow(p, steps, seed) == pytest.approx([(up[i] - down[i]) / steps for i in range(5)], abs=1e-15), (p, seed)


def test_zeitanteile_gegen_eine_einfache_schleife_mit_dem_generator():
    ring = make_ring(4, 2.0)
    P = K.metropolis(ring)
    rng = SplitMix64(77)
    x = rng.choice(list(ring.pi))
    cum = [list(np.cumsum(P[i])) for i in range(ring.size)]
    for c in cum:
        c[-1] = 2.0
    counts = np.zeros(ring.size)
    for _ in range(3000):
        u = rng.random()
        x = next(j for j, v in enumerate(cum[x]) if u < v)
        counts[x] += 1
    assert S.time_fractions(P, ring.pi, 3000, 77) == pytest.approx(counts / 3000, abs=1e-15)


def test_gleicher_zufalls_seed_gleiche_zeitanteile_anderer_seed_andere():
    ring = make_ring(5, 2.0)
    P = K.heat_bath(ring)
    a, b, c = (S.time_fractions(P, ring.pi, 4000, s) for s in (5, 5, 6))
    assert np.array_equal(a, b) and not np.allclose(a, c)
    assert a.sum() == pytest.approx(1.0, abs=1e-12)


def test_rauschgrenze_sinkt_mit_der_laenge_der_stichprobe():
    ring = make_ring(6, 2.0)
    kurz, lang = S.iid_time_fraction_noise(ring.pi, 2_000, 1, 4), S.iid_time_fraction_noise(ring.pi, 200_000, 1, 4)
    assert lang < kurz / 3, "TV wächst wie 1/√T: bei 100facher Länge etwa ein Zehntel"


def test_simulation_braucht_mindestens_einen_schritt():
    ring = make_ring(4, 2.0)
    with pytest.raises(ValueError):
        S.time_fractions(K.metropolis(ring), ring.pi, 0, 1)
    with pytest.raises(ValueError):
        Y.simulate_net_flow(0.7, 0, 1)
    with pytest.raises(ValueError):
        S.iid_time_fraction_noise(ring.pi, 0, 1, 2)
    assert H.tv(S.time_fractions(K.metropolis(ring), ring.pi, 1, 1), ring.pi) > 0, "ein Schritt: die Zeitanteile sind ein Einheitsvektor"

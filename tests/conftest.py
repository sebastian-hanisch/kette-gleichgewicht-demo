"""Gemeinsame Fixtures: das Beispiel von Hand (Ring mit 4 Plätzen, λ = 2, 7 Zustände) und der zweite Raum (n = 5, λ = 2, 11 Zustände), jeweils mit dem exakten Aufbau in Brüchen."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import exakt  # noqa: E402
from kgw_ring import make_ring  # noqa: E402


@pytest.fixture(scope="session")
def ring4():
    return make_ring(4, 2.0)


@pytest.fixture(scope="session")
def raum4():
    return exakt.Raum(4, 2)


@pytest.fixture(scope="session")
def perm4(ring4, raum4):
    return exakt.app_nach_exakt(ring4, raum4)


@pytest.fixture(scope="session")
def ring5():
    return make_ring(5, 2.0)


@pytest.fixture(scope="session")
def raum5():
    return exakt.Raum(5, 2)


@pytest.fixture(scope="session")
def perm5(ring5, raum5):
    return exakt.app_nach_exakt(ring5, raum5)

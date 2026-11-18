"""Kerne (Übergangsmatrizen) mit gegebenem Gleichgewicht π auf dem Hard-Core-Ring, die zugehörigen Fehlerfälle und nichtreversible Kerne.

Leitwert-Sicht (Levin/Peres/Wilmer, Kapitel 1): ein Fluss f(x, y) = π(x) P(x, y) auf den Kanten des Zustandsgraphen bestimmt den Kern, die Diagonale
ergänzt die Zeilensumme auf 1. Ist der Fluss symmetrisch (f(x, y) = f(y, x)), gilt Detailed Balance und damit πP = π; die Zeilensumme begrenzt den Fluss.
Hier stehen: Metropolis, Heat-bath und gleiche Leitwerte (alle reversibel zu π), drei Fehlerfälle (Kerne mit Zeilensumme 1, die π verfehlen), die Drehung
des Rings, eine Mischung aus Metropolis und Drehung mit Drift und Kreisströme (nichtreversible Kerne mit demselben π). Zufall nur über kgw_rng (SplitMix64).
"""

from functools import lru_cache

import numpy as np

import kgw_constants as C
from kgw_ring import Ring, make_ring, rotation_matrix
from kgw_rng import SplitMix64


class NoCycleError(ValueError):
    """Der Zustandsgraph hat keinen Kreis (ein Baum): jede stationäre Kette auf ihm ist reversibel."""


def _complete_diagonal(P: np.ndarray) -> np.ndarray:
    """Setzt die Diagonale so, dass jede Zeile die Summe 1 hat; eine Zeilensumme über 1 ist ein Fehler der Eingabe."""
    np.fill_diagonal(P, 0.0)
    diag = 1.0 - P.sum(axis=1)
    if diag.min() < -1e-12:
        raise ValueError(f"Zeilensumme über 1 (um {-diag.min():.3g}): die Leitwerte sind zu groß für diese Verteilung.")
    P[np.diag_indices_from(P)] = np.maximum(diag, 0.0)
    return P


def kernel_from_flows(ring: Ring, flow_ab: np.ndarray, flow_ba: np.ndarray) -> np.ndarray:
    """Kern aus den Flüssen je Kante: P(a, b) = f(a, b) / π(a) und P(b, a) = f(b, a) / π(b); die Diagonale ergänzt die Zeilensumme."""
    a, b = ring.edges[:, 0], ring.edges[:, 1]
    flow_ab, flow_ba = np.asarray(flow_ab, float), np.asarray(flow_ba, float)
    scale = max(float(np.max(flow_ab)), float(np.max(flow_ba)), 1e-300)
    if min(float(np.min(flow_ab)), float(np.min(flow_ba))) < -1e-12 * scale:
        raise ValueError("Ein Fluss ist negativ: ein Fluss ist ein Leitwert und darf nicht negativ sein.")
    P = np.zeros((ring.size, ring.size))
    P[a, b] = np.maximum(flow_ab, 0.0) / ring.pi[a]    # Rundungsreste knapp unter null (Kreisstrom an der Grenze) zählen als null
    P[b, a] = np.maximum(flow_ba, 0.0) / ring.pi[b]
    return _complete_diagonal(P)


def metropolis_conductance(ring: Ring) -> np.ndarray:
    """Metropolis-Leitwert je Kante: min(π(x), π(y)) / n, der größte zulässige Wert bei Platzwahl 1/n (LPW, Seite 39)."""
    a, b = ring.edges[:, 0], ring.edges[:, 1]
    return np.minimum(ring.pi[a], ring.pi[b]) / ring.n


def metropolis(ring: Ring, factor: float = 1.0) -> np.ndarray:
    """Metropolis mit Leitwertfaktor 0 < f ≤ 1 (f = 1 ist Metropolis selbst, kleineres f eine trägere Fassung desselben Flussnetzes)."""
    c = factor * metropolis_conductance(ring)
    return kernel_from_flows(ring, c, c)


def staged_factors(ring: Ring, f0: float, f1: float) -> np.ndarray:
    """Faktor je Kante: f0 für Kanten zwischen k = 0 und k = 1 belegten Plätzen, f1 für alle höheren Stufen k → k + 1."""
    stage = np.minimum(ring.count[ring.edges[:, 0]], ring.count[ring.edges[:, 1]])
    return np.where(stage == 0, f0, f1)


def metropolis_staged(ring: Ring, f0: float, f1: float) -> np.ndarray:
    """Eigene Leitwerte: Metropolis-Leitwert mal Faktor je Stufe (nur um zulässige Kerne zu bauen)."""
    c = staged_factors(ring, f0, f1) * metropolis_conductance(ring)
    return kernel_from_flows(ring, c, c)


def equal_conductance_value(ring: Ring) -> float:
    """Größter gemeinsamer Leitwert c, der auf allen Kanten zulässig ist: c = min über x von π(x) / Grad(x)."""
    return float(np.min(ring.pi / np.maximum(ring.deg, 1)))


def equal_conductance(ring: Ring) -> np.ndarray:
    c = np.full(ring.n_edges, equal_conductance_value(ring))
    return kernel_from_flows(ring, c, c)


def heat_bath_rule(ring: Ring, p_occupy: float) -> np.ndarray:
    """Platzwahl gleichverteilt (1/n); haben beide Nachbarn keinen Teilchen: belegen mit Wahrscheinlichkeit p_occupy, sonst leeren; ist ein Nachbar belegt: leeren."""
    n, P = ring.n, np.zeros((ring.size, ring.size))
    for i, m in enumerate(ring.masks):
        for v in range(n):
            blocked = ((m >> ((v - 1) % n)) & 1) or ((m >> ((v + 1) % n)) & 1)
            emptied = ring.index[m & ~(1 << v)]
            if blocked:
                P[i, emptied] += 1.0 / n
            else:
                P[i, ring.index[m | (1 << v)]] += p_occupy / n
                P[i, emptied] += (1.0 - p_occupy) / n
    return P


def heat_bath(ring: Ring) -> np.ndarray:
    """Heat-bath (Gibbs-Sampler, Glauber-Dynamik): belegen mit λ/(1+λ), die bedingte Wahrscheinlichkeit von π."""
    return heat_bath_rule(ring, ring.lam / (1.0 + ring.lam))


# ---- Fehlerfälle: Zeilensumme 1, aber π ist nicht stationär -----------------------------------------------------------------------------------
def fixed_half(ring: Ring) -> np.ndarray:
    """Fehler: jeder zulässige Vorschlag (Platz mit Wahrscheinlichkeit 1/n) wird mit Wahrscheinlichkeit ½ angenommen. Reversibel zur Gleichverteilung."""
    a, b = ring.edges[:, 0], ring.edges[:, 1]
    P = np.zeros((ring.size, ring.size))
    P[a, b] = 0.5 / ring.n
    P[b, a] = 0.5 / ring.n
    return _complete_diagonal(P)


def inverse_ratio(ring: Ring) -> np.ndarray:
    """Fehler: Metropolis mit umgekehrtem Verhältnis, Annahme min(1, π(x)/π(y)) statt min(1, π(y)/π(x)). Reversibel zu einer Verteilung ∝ 1/Gewicht."""
    a, b = ring.edges[:, 0], ring.edges[:, 1]
    P = np.zeros((ring.size, ring.size))
    P[a, b] = np.minimum(1.0, ring.pi[a] / ring.pi[b]) / ring.n
    P[b, a] = np.minimum(1.0, ring.pi[b] / ring.pi[a]) / ring.n
    return _complete_diagonal(P)


def heat_bath_inverse(ring: Ring) -> np.ndarray:
    """Fehler: Heat-bath, der mit 1/(1+λ) statt λ/(1+λ) belegt (Gewicht 1/λ je Platz statt λ)."""
    return heat_bath_rule(ring, 1.0 / (1.0 + ring.lam))


KERNEL_BUILDERS = {
    C.K_METRO: lambda ring, f0=1.0, f1=1.0: metropolis(ring),
    C.K_HEAT: lambda ring, f0=1.0, f1=1.0: heat_bath(ring),
    C.K_EQUAL: lambda ring, f0=1.0, f1=1.0: equal_conductance(ring),
    C.K_STAGED: lambda ring, f0=1.0, f1=1.0: metropolis_staged(ring, f0, f1),
    C.K_HALF: lambda ring, f0=1.0, f1=1.0: fixed_half(ring),
    C.K_INVERSE: lambda ring, f0=1.0, f1=1.0: inverse_ratio(ring),
    C.K_HEAT_INV: lambda ring, f0=1.0, f1=1.0: heat_bath_inverse(ring),
}


def build_kernel(ring: Ring, name: str, f0: float = 1.0, f1: float = 1.0) -> np.ndarray:
    if name not in KERNEL_BUILDERS:
        raise ValueError(f"unbekannter Kern: {name}")
    return KERNEL_BUILDERS[name](ring, f0, f1)


# ---- Drehung und Mischung -----------------------------------------------------------------------------------------------------------------
def rotation_drift(ring: Ring, p: float) -> np.ndarray:
    """Drehung um einen Platz mit Wahrscheinlichkeit p, Gegendrehung mit 1 − p: erhält die Zahl der belegten Plätze und damit π (jede π-erhaltende Permutation tut es)."""
    R = rotation_matrix(ring)
    return p * R + (1.0 - p) * R.T


def mixture(ring: Ring, q: float, p: float) -> np.ndarray:
    """(1 − q) · Metropolis + q · [p · Drehung + (1 − p) · Gegendrehung]: nichtreversibel für p ≠ ½ und q > 0, π bleibt stationär."""
    return (1.0 - q) * metropolis(ring) + q * rotation_drift(ring, p)


# ---- Kreisströme -------------------------------------------------------------------------------------------------------------------------
def cycle_basis(ring: Ring) -> np.ndarray:
    """Eine Kreisbasis des Zustandsgraphen (hängt nur von n ab, nicht von λ): je Nicht-Baum-Kante ein Vektor über die Kanten (Vorzeichen = Richtung a → b);
    Zeilenzahl E − V + 1."""
    return _cycle_basis_of(make_ring(ring.n, 1.0))


@lru_cache(maxsize=16)
def _cycle_basis_of(ring: Ring) -> np.ndarray:
    V, E = ring.size, ring.n_edges
    adj = [[] for _ in range(V)]
    for k, (a, b) in enumerate(ring.edges):
        adj[a].append((int(b), k, 1))
        adj[b].append((int(a), k, -1))
    parent = {0: None}
    order, tree = [0], set()
    for u in order:
        for v, k, sign in adj[u]:
            if v not in parent:
                parent[v] = (u, k, sign)
                tree.add(k)
                order.append(v)

    def to_root(v):
        out = []
        while parent[v] is not None:
            u, k, sign = parent[v]
            out.append((k, -sign))
            v = u
        return out

    rows = []
    for k, (a, b) in enumerate(ring.edges):
        if k in tree:
            continue
        vec = np.zeros(E)
        vec[k] = 1.0
        for kk, s in to_root(int(b)):
            vec[kk] += s
        for kk, s in to_root(int(a)):
            vec[kk] -= s
        rows.append(vec)
    return np.array(rows).reshape(-1, E)


def random_cycle_current(ring: Ring, seed: int) -> np.ndarray:
    """Zufälliger Kreisstrom ψ (antisymmetrisch, in jedem Zustand erhaltend): Summe der Basiskreise mit normalverteilten Gewichten (SplitMix64)."""
    basis = cycle_basis(ring)
    if len(basis) == 0:
        raise NoCycleError("Der Zustandsgraph dieses Rings ist ein Baum: es gibt keinen Kreis und damit keinen nichtreversiblen Kern.")
    rng = SplitMix64(seed)
    coef = np.array([rng.normal() for _ in range(len(basis))])
    return coef @ basis


def cycle_kernels(ring: Ring, factor: float, share: float, seed: int):
    """(Kern mit Kreisstrom, reversibler Basiskern, größte zulässige Stärke): Fluss f · c_M ± s · s_max · ψ auf jeder Kante.

    Der Basiskern ist reversibel (Leitwert f · c_M); der Kreisstrom ψ wird auf die größte Stärke s_max skaliert, bei der alle Flüsse nichtnegativ bleiben,
    und mit dem Anteil `share` (0 bis 1) davon addiert. Jede Zustandsbilanz bleibt erhalten, also bleibt π stationär."""
    psi = random_cycle_current(ring, seed)
    base = factor * metropolis_conductance(ring)
    nonzero = np.abs(psi) > 1e-15
    s_max = float(np.min(base[nonzero] / np.abs(psi[nonzero]))) if nonzero.any() else 0.0
    strength = share * s_max
    P = kernel_from_flows(ring, base + strength * psi, base - strength * psi)
    return P, kernel_from_flows(ring, base, base), s_max

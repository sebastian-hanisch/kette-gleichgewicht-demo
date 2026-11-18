"""Der Hard-Core-Ring: Zustände, Zielverteilung π, Zustandsgraph der Einzelplatz-Züge und die Drehung.

Ausgedachtes Modell, keine reale Datenquelle: n Stellplätze im Ring, kein Paar benachbarter Plätze ist zugleich belegt. Ein Zustand x ist die Menge der
belegten Plätze, sein Gewicht ist λ^|x| (|x| = Zahl der belegten Plätze), π(x) = λ^|x| / Z mit der Normierung Z = Summe aller Gewichte.
Die Zustandszahlen sind die Lucas-Zahlen 4, 7, 11, 18, 29, 47, 76, 123, 199, 322 für n = 3 bis 12. Die Zustände heißen in der Reihenfolge
„erst nach Zahl der belegten Plätze, dann nach Bitmaske“; Zustand 0 ist immer der leere Ring.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True, eq=False)
class Ring:
    n: int
    lam: float
    masks: tuple          # Bitmaske der belegten Plätze je Zustand
    count: np.ndarray     # Zahl der belegten Plätze je Zustand
    pi: np.ndarray        # Zielverteilung
    edges: np.ndarray     # (E, 2): Zustandsgraph, Kanten a < b, die sich in genau einem Platz unterscheiden
    deg: np.ndarray       # Grad je Zustand im Zustandsgraphen
    rot: np.ndarray       # rot[i] = Zustand, der aus Zustand i durch Drehung um einen Platz entsteht
    index: dict           # Bitmaske -> Zustand

    @property
    def size(self) -> int:
        return len(self.masks)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def label(self, i: int) -> str:
        m = self.masks[i]
        occupied = [v for v in range(self.n) if (m >> v) & 1]
        return "∅" if not occupied else "{" + ",".join(str(v) for v in occupied) + "}"

    def labels(self) -> list:
        return [self.label(i) for i in range(self.size)]

    @property
    def empty(self) -> int:
        return 0

    @property
    def full(self) -> int:
        """Erster Zustand mit der größten Zahl belegter Plätze („Maximalbelegung“)."""
        return int(np.argmax(self.count))

    def single(self) -> int:
        """Der Zustand, in dem nur Platz 0 belegt ist."""
        return self.index[1]

    def weights(self) -> np.ndarray:
        return float(self.lam) ** self.count.astype(float)


def independent_masks(n: int) -> np.ndarray:
    """Alle Bitmasken ohne zwei benachbarte belegte Plätze im Ring der Länge n (n ≥ 3), aufsteigend."""
    full = (1 << n) - 1
    m = np.arange(1 << n, dtype=np.int64)
    shifted = ((m << 1) | (m >> (n - 1))) & full
    return m[(m & shifted) == 0]


def _popcount(m: np.ndarray) -> np.ndarray:
    out = np.zeros(len(m), dtype=int)
    for v in range(int(m.max()).bit_length() if len(m) else 0):
        out += (m >> v) & 1
    return out


@lru_cache(maxsize=64)
def make_ring(n: int, lam: float) -> Ring:
    """Ring mit n Plätzen und Gewicht λ je belegtem Platz (n ≥ 3, λ > 0)."""
    if n < 3:
        raise ValueError("Der Ring ist erst ab n = 3 Plätzen definiert (bei n = 2 wären die beiden Plätze zugleich links und rechts Nachbarn).")
    if lam <= 0:
        raise ValueError("λ muss positiv sein.")
    masks = independent_masks(n)
    counts = _popcount(masks)
    order = np.lexsort((masks, counts))
    masks, counts = masks[order], counts[order]
    index = {int(m): i for i, m in enumerate(masks)}
    w = float(lam) ** counts.astype(float)
    pi = w / w.sum()
    edges = []
    for i, m in enumerate(masks):
        for v in range(n):
            j = index.get(int(m) ^ (1 << v))
            if j is not None and j > i:
                edges.append((i, j))
    edges = np.array(edges, dtype=int).reshape(-1, 2)
    deg = np.zeros(len(masks), dtype=int)
    np.add.at(deg, edges[:, 0], 1)
    np.add.at(deg, edges[:, 1], 1)
    full = (1 << n) - 1
    rot = np.array([index[((int(m) << 1) | (int(m) >> (n - 1))) & full] for m in masks], dtype=int)
    return Ring(n, float(lam), tuple(int(m) for m in masks), counts, pi, edges, deg, rot, index)


def normalisation(n: int, lam: float) -> float:
    """Normierung Z = μ₊ⁿ + μ₋ⁿ mit μ± = (1 ± √(1 + 4λ)) / 2 (Transfermatrix des Rings): ein zweiter Weg zu Z, ohne die Zustände aufzuzählen."""
    root = (1.0 + 4.0 * lam) ** 0.5
    return ((1 + root) / 2) ** n + ((1 - root) / 2) ** n


def rotation_matrix(ring: Ring) -> np.ndarray:
    """Permutationsmatrix der Drehung um einen Platz: Zeile i hat ihre 1 in Spalte rot[i]."""
    R = np.zeros((ring.size, ring.size))
    R[np.arange(ring.size), ring.rot] = 1.0
    return R


def rotation_orbits(ring: Ring) -> int:
    """Zahl der Bahnen der Drehung auf den Zuständen (= Zahl der Klassen der reinen Drehung)."""
    seen = np.zeros(ring.size, dtype=bool)
    orbits = 0
    for i in range(ring.size):
        if seen[i]:
            continue
        orbits += 1
        j = i
        while not seen[j]:
            seen[j] = True
            j = ring.rot[j]
    return orbits

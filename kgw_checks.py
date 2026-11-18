"""Prüfungen eines Kerns gegen π: Zeilensummen, πP = π, Detailed Balance, stationäre Verteilung aus einem linearen System, Klassen, Zeitumkehr,
Totalvariationsabstand d(t), Mischzeit t_mix(1/4), zweitgrößter Eigenwertbetrag und die Freiheitsgrade der Flüsse.

Zähleinheiten: Schritte (ein Schritt der Kette), Freiheitsgrade (Dimension des linearen Raums der Flüsse), Klassen (kommunizierende Klassen).
d(t) = max über alle Starts x von TV(P^t(x, ·), π) und t_mix = kleinstes t mit d(t) ≤ 1/4 (Levin/Peres/Wilmer, Kapitel 4). Die zweite Zähleinheit
der Mischzeit, d̄(t) zwischen zwei Starts mit Schwelle 1/e, kommt hier nicht vor.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import kgw_constants as C
from kgw_ring import Ring


def row_sum_error(P: np.ndarray) -> float:
    return float(np.abs(P.sum(axis=1) - 1.0).max())


def min_entry(P: np.ndarray) -> float:
    return float(P.min())


def stationary_residual(P: np.ndarray, pi: np.ndarray) -> float:
    """‖πP − π‖₁: null genau dann, wenn π stationär ist."""
    return float(np.abs(pi @ P - pi).sum())


def residual_vector(P: np.ndarray, pi: np.ndarray) -> np.ndarray:
    return pi @ P - pi


def db_deviation(P: np.ndarray, pi: np.ndarray) -> float:
    """Größte Abweichung von Detailed Balance: max über x, y von |π(x) P(x, y) − π(y) P(y, x)|."""
    flow = pi[:, None] * P
    return float(np.abs(flow - flow.T).max())


def stationary_solve(P: np.ndarray):
    """Stationäre Verteilung aus dem linearen System π(P − I) = 0, Summe 1 (eine Gleichung wird durch die Normierung ersetzt); None, wenn nicht eindeutig lösbar."""
    S = P.shape[0]
    A = P.T - np.eye(S)
    A[0, :] = 1.0
    b = np.zeros(S)
    b[0] = 1.0
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return x if np.all(np.isfinite(x)) else None


def tv(u: np.ndarray, v: np.ndarray) -> float:
    return float(0.5 * np.abs(u - v).sum())


def n_classes(P: np.ndarray) -> int:
    """Zahl der kommunizierenden Klassen (starke Zusammenhangskomponenten des Graphen der positiven Übergänge)."""
    return int(connected_components(csr_matrix(P > 1e-15), directed=True, connection="strong")[0])


def time_reversal(P: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Zeitumkehrte Kette P*(x, y) = π(y) P(y, x) / π(x): hat dasselbe π; P* = P genau dann, wenn Detailed Balance gilt."""
    return (P.T * pi[None, :]) / pi[:, None]


def eigenvalues_abs(P: np.ndarray) -> np.ndarray:
    """Beträge der Eigenwerte, absteigend; der erste ist 1."""
    return np.sort(np.abs(np.linalg.eigvals(P)))[::-1]


def second_eigenvalue_abs(P: np.ndarray) -> float:
    return float(eigenvalues_abs(P)[1])


def d_of(M: np.ndarray, pi: np.ndarray) -> float:
    """d für eine Matrix M = P^t: größter Totalvariationsabstand ihrer Zeilen zu π."""
    return float(0.5 * np.abs(M - pi[None, :]).sum(axis=1).max())


def d_curve(P: np.ndarray, pi: np.ndarray, t_max: int) -> np.ndarray:
    """d(t) für t = 0 bis t_max (Schritt für Schritt mit Matrixprodukten)."""
    M = np.eye(P.shape[0])
    out = [d_of(M, pi)]
    for _ in range(t_max):
        M = M @ P
        out.append(d_of(M, pi))
    return np.array(out)


def mu_curve(P: np.ndarray, pi: np.ndarray, start: int, t_max: int) -> np.ndarray:
    """TV(μ_t, π) für t = 0 bis t_max, μ_t = e_start P^t: die Verteilung nach t Schritten vom Start `start`."""
    v = np.zeros(P.shape[0])
    v[start] = 1.0
    out = [tv(v, pi)]
    for _ in range(t_max):
        v = v @ P
        out.append(tv(v, pi))
    return np.array(out)


def t_mix(P: np.ndarray, pi: np.ndarray, eps: float = C.EPS_MIX):
    """Kleinstes t mit d(t) ≤ eps (d ist nichtwachsend), exakt aus Matrixpotenzen: P^(2^j) durch Quadrieren (Zeilen nachnormiert), bis d(2^j) ≤ eps, dann binäres Heben
    zum größten t mit d(t) > eps; das Ergebnis ist dieses t plus 1. None, wenn d(2^T_MAX_POW) noch über eps liegt (die Kette kommt in dieser Zahl von Schritten nicht an)."""
    S = P.shape[0]
    tol = 1e-12
    if d_of(np.eye(S), pi) <= eps + tol:
        return 0
    powers = [P]
    while d_of(powers[-1], pi) > eps + tol:
        if len(powers) > C.T_MAX_POW:
            return None
        Q = powers[-1] @ powers[-1]
        powers.append(Q / Q.sum(axis=1, keepdims=True))
    M, t = np.eye(S), 0
    for j in range(len(powers) - 2, -1, -1):          # d(2^(len−1)) ≤ eps: gesucht ist das größte t < 2^(len−1) mit d(t) > eps
        cand = M @ powers[j]
        if d_of(cand, pi) > eps + tol:
            M, t = cand, t + (1 << j)
    return t + 1


# ---- Freiheitsgrade der Flüsse ---------------------------------------------------------------------------------------------------------------
def conservation_matrix(size: int, pairs) -> np.ndarray:
    """Zeile y, Spalte (x → y): +1 für den Zufluss in y, −1 für den Abfluss aus x; Nullraum = Flüsse mit Zufluss = Abfluss in jedem Zustand."""
    A = np.zeros((size, len(pairs)))
    for k, (x, y) in enumerate(pairs):
        A[y, k] += 1.0
        A[x, k] -= 1.0
    return A


def graph_connected(ring: Ring) -> bool:
    n_comp = connected_components(csr_matrix((np.ones(ring.n_edges), (ring.edges[:, 0], ring.edges[:, 1])), shape=(ring.size, ring.size)), directed=False)[0]
    return int(n_comp) == 1


def flow_dimensions(ring: Ring, numeric: bool) -> dict:
    """V Zustände, E Kanten. Stationäre Flüsse auf dem Graphen: 2E − Rang(Erhaltung), Formel 2E − (V − 1); reversible: E; Unterschied E − V + 1 (Kreise).
    Mit numeric = True wird der Rang der Erhaltungsgleichungen gerechnet (sonst gilt die Formel, gekennzeichnet in 'rang_numerisch')."""
    V, E = ring.size, ring.n_edges
    out = {"V": V, "E": E, "kreise": E - V + 1, "reversibel": E, "stationaer_formel": 2 * E - (V - 1), "rang_numerisch": bool(numeric)}
    if numeric:
        pairs = [(int(a), int(b)) for a, b in ring.edges] + [(int(b), int(a)) for a, b in ring.edges]
        rank = int(np.linalg.matrix_rank(conservation_matrix(V, pairs)))
        out["rang"] = rank
        out["stationaer"] = len(pairs) - rank
    else:
        out["rang"] = V - 1
        out["stationaer"] = out["stationaer_formel"]
    return out


def flow_dimensions_all_moves(size: int, numeric: bool) -> dict:
    """Ohne Graphbeschränkung (jeder Zug erlaubt): size·(size−1) Flüsse, Rang size − 1, also (size−1)² stationäre und size·(size−1)/2 reversible."""
    out = {"stationaer": (size - 1) ** 2, "reversibel": size * (size - 1) // 2, "rang_numerisch": bool(numeric)}
    if numeric:
        pairs = [(x, y) for x in range(size) for y in range(size) if x != y]
        rank = int(np.linalg.matrix_rank(conservation_matrix(size, pairs)))
        out["stationaer"] = len(pairs) - rank
    return out

"""Auswertungen der vier Experimente und ihre Verdikte (reine Funktionen: (Art, Text), die App ruft st.<Art>(Text)).

1. Eine Kette bauen und prüfen: Kern aus Leitwerten, Prüfung πP = π, Detailed Balance, stationäre Verteilung aus einem linearen System, Mischzeit.
2. Detailed Balance ist hinreichend, nicht notwendig: der 5-Kreis mit Drift.
3. Wie viele Freiheitsgrade haben die Ketten, die zu π passen: Dimension des Flussraums und zufällige Kreisströme.
4. Stationär, aber angekommen? Kern gegen Verteilung nach t Schritten; Drehung als Zug, reine Drehung.
Zähleinheiten: Schritte, Freiheitsgrade, Klassen; Zufallsergebnisse (Zeitanteile, Nettofluss) sind Bänder, keine Zahlen.
"""

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import numpy as np

import kgw_checks as H
import kgw_constants as C
import kgw_cycle5 as Y
import kgw_kernels as K
import kgw_simulation as S
from kgw_format import bruch, dez, ganz, mit_einheit, schritte, spanne, vorzeichen, zahl
from kgw_ring import Ring, make_ring, rotation_matrix, rotation_orbits


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    lam: float = C.DEFAULT_LAM
    seed: int = C.DEFAULT_SEED
    kern: str = C.K_METRO
    f0: float = C.DEFAULT_F
    f1: float = C.DEFAULT_F
    time_steps: int = C.DEFAULT_TIME_STEPS
    p5: float = C.DEFAULT_P5
    steps: int = C.DEFAULT_STEPS
    share: float = C.DEFAULT_S
    fbase: float = C.DEFAULT_FB
    arrival: str = C.A_METRO
    q: float = C.DEFAULT_Q
    p4: float = C.DEFAULT_P4
    start: str = C.START_EMPTY
    burn: int = C.DEFAULT_BURN


# =====================================================================================================================================================
# Experiment 1: eine Kette bauen und prüfen
# =====================================================================================================================================================
@dataclass(frozen=True, eq=False)
class BuildAnalysis:
    n: int
    lam: float
    kern: str
    size: int
    labels: tuple
    P: np.ndarray
    pi: np.ndarray
    row_err: float
    min_entry: float
    residual: float
    residual_vec: np.ndarray
    db: float
    stationary: object          # Feld oder None
    stationary_tv: object       # float oder None
    tmix: object                # int oder None
    refs: dict                  # Metropolis, Heat-bath, Gleiche Leitwerte -> t_mix


@lru_cache(maxsize=64)
def reference_times(n: int, lam: float) -> dict:
    """t_mix(1/4) der drei Regeln mit demselben π (Schritte)."""
    ring = make_ring(n, lam)
    return {name: H.t_mix(K.build_kernel(ring, name), ring.pi) for name in (C.K_METRO, C.K_HEAT, C.K_EQUAL)}


def analyse_build(n: int, lam: float, kern: str, f0: float = 1.0, f1: float = 1.0) -> BuildAnalysis:
    ring = make_ring(n, lam)
    P = K.build_kernel(ring, kern, f0, f1)
    stat = H.stationary_solve(P)
    return BuildAnalysis(
        n=n, lam=lam, kern=kern, size=ring.size, labels=tuple(ring.labels()), P=P, pi=ring.pi,
        row_err=H.row_sum_error(P), min_entry=H.min_entry(P), residual=H.stationary_residual(P, ring.pi),
        residual_vec=H.residual_vector(P, ring.pi), db=H.db_deviation(P, ring.pi), stationary=stat,
        stationary_tv=None if stat is None else H.tv(stat, ring.pi), tmix=H.t_mix(P, ring.pi), refs=reference_times(n, lam),
    )


def _arrival_text(tmix) -> str:
    if tmix is None:
        return "kommt die Kette innerhalb von 2²² Schritten **nicht an**"
    return f"braucht die Kette **{ganz(tmix)} Schritte**"


def verdict_build(a: BuildAnalysis) -> tuple:
    """Verdikt zur Frage „Hat diese Kette π als stationäre Verteilung?“."""
    setting = f"n = {a.n}, λ = {zahl(a.lam)}, {a.size} Zustände"
    if a.residual <= C.TOL:
        if a.kern in C.KERN_FEHLER:
            reason = " (mit λ = 1 sind alle Gewichte gleich)" if a.lam == 1.0 else ""
            return "info", (f"Hier fängt die Prüfung nichts: ‖πP − π‖₁ < 10⁻⁹, der Fehlerfall ist bei diesen Einstellungen wirkungslos{reason}. "
                            f"Zur Probe λ ≠ 1 wählen ({setting}).")
        db = "unter 10⁻⁹" if a.db <= C.TOL else dez(a.db, 4)
        r = a.refs
        return "success", (f"**π ist stationär** (‖πP − π‖₁ < 10⁻⁹, Detailed-Balance-Abweichung {db}): {a.kern}, {setting}. "
                           f"Bis d(t) ≤ 1/4 {_arrival_text(a.tmix)}; bei gleichem π brauchen Metropolis {schritte(r[C.K_METRO])}, "
                           f"Heat-bath {schritte(r[C.K_HEAT])} und gleiche Leitwerte {schritte(r[C.K_EQUAL])} Schritte.")
    own = " Stationär ist die Gleichverteilung." if a.kern == C.K_HALF else ""
    tv = "" if a.stationary_tv is None else f" Der Totalvariationsabstand ihrer stationären Verteilung zu π beträgt {dez(a.stationary_tv, 4)}."
    return "warning", (f"**Fällt durch die Prüfung:** ‖πP − π‖₁ = {dez(a.residual, 4)}. Zeilensummen stimmen (größter Fehler unter 10⁻⁹), alle Einträge sind ≥ 0, "
                       f"die Kette läuft, hat aber π nicht als stationäre Verteilung ({a.kern}, {setting}).{own}{tv}")


# ---- Zeitanteile (Simulation) ------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class TimeFractionAnalysis:
    steps: int
    seed: int
    tv: float
    noise: float
    ratio: object
    stationary_tv: object


def analyse_time_fractions(n: int, lam: float, kern: str, f0: float, f1: float, steps: int, seed: int) -> TimeFractionAnalysis:
    ring = make_ring(n, lam)
    P = K.build_kernel(ring, kern, f0, f1)
    tv = H.tv(S.time_fractions(P, ring.pi, steps, seed), ring.pi)
    noise = S.iid_time_fraction_noise(ring.pi, steps, seed + 1_000_003, C.NOISE_REPS)
    stat = H.stationary_solve(P)
    return TimeFractionAnalysis(steps, seed, tv, noise, tv / noise if noise > 0 else None, None if stat is None else H.tv(stat, ring.pi))


def verdict_time_fractions(a: TimeFractionAnalysis, residual: float) -> tuple:
    """Verdikt der Simulation: Zeitanteile einer einzigen Kette gegen π, neben der Rauschgrenze unabhängiger Ziehungen."""
    head = (f"Zeitanteile einer Kette über {ganz(a.steps)} Schritte (Zufalls-Seed {a.seed}): Totalvariationsabstand zu π **{dez(a.tv, 4)}**, "
            f"Rauschgrenze unabhängiger Ziehungen (größter Wert aus {C.NOISE_REPS} Stichproben gleicher Größe) {dez(a.noise, 4)}")
    if residual > C.TOL:
        return "warning", head + f". Der Wert liegt beim Abstand {dez(a.stationary_tv, 4)} der stationären Verteilung zu π: auch ein langer Lauf führt nicht zu π."
    ratio = "" if a.ratio is None else f", das {dez(a.ratio, 1)}fache"
    return "info", head + f"{ratio}: eine Kette liefert keine unabhängigen Ziehungen. Ein Lauf ist ein Band, keine Zahl."


# =====================================================================================================================================================
# Experiment 2: der 5-Kreis mit Drift
# =====================================================================================================================================================
@dataclass(frozen=True, eq=False)
class Cycle5Analysis:
    p: float
    steps: int
    seed: int
    deviation: float
    kol_fwd: float
    kol_bwd: float
    net_exact: float
    net_measured: np.ndarray
    sd: float
    max_dev: float
    band_ok: bool
    tmix: object
    lam2: float
    P: np.ndarray
    P_star: np.ndarray
    stationary_residual: float


def analyse_cycle5(p: float, steps: int, seed: int) -> Cycle5Analysis:
    P = Y.kernel(p)
    net = Y.simulate_net_flow(p, steps, seed)
    exact = Y.net_flow_exact(p)
    sd = Y.net_flow_sd(steps)
    max_dev = float(np.abs(net - exact).max())
    kf, kb = Y.kolmogorov(p)
    return Cycle5Analysis(
        p=p, steps=steps, seed=seed, deviation=Y.deviation(p), kol_fwd=kf, kol_bwd=kb, net_exact=exact, net_measured=net, sd=sd, max_dev=max_dev,
        band_ok=max_dev < 4 * sd, tmix=H.t_mix(P, Y.PI5), lam2=H.second_eigenvalue_abs(P), P=P, P_star=H.time_reversal(P, Y.PI5),
        stationary_residual=H.stationary_residual(P, Y.PI5),
    )


def verdict_cycle5(a: Cycle5Analysis) -> tuple:
    """Verdikt zur Frage „Erfüllt sie Detailed Balance, und muss sie?“ am 5-Kreis."""
    mean = float(np.mean(a.net_measured))
    measured = (f"gemessen {dez(mean, 4)} (Zufalls-Seed {a.seed}, {ganz(a.steps)} Schritte, grobe obere Schranke der Standardabweichung je Kante etwa {dez(a.sd, 4)})")
    if a.p == 0.5:
        return "info", (f"p = 0,5: reversibel. Abweichung 0, Produkte vorwärts und rückwärts gleich ({dez(a.kol_fwd, 5)}), Nettofluss je Kante und Schritt exakt 0, "
                        f"{measured}.")
    kind = "success" if a.band_ok else "warning"
    outside = "" if a.band_ok else f" **Die Messung liegt außerhalb von ±4 Standardabweichungen um den exakten Wert** (größte Abweichung {dez(a.max_dev, 4)})."
    tail = ("Die zeitumkehrte Kette hat die Drift rückwärts: Detailed Balance ist hinreichend, nicht notwendig."
            if a.p < 1.0 else "Bei p = 1 ist das eine reine Drehung: π stationär, Detailed Balance maximal verletzt.")
    return kind, (f"p = {zahl(a.p)}: Gleichverteilung stationär, Detailed Balance verletzt (Abweichung {dez(a.deviation, 4)}); Nettofluss je Kante und Schritt "
                  f"exakt {dez(a.net_exact, 4)}, {measured}. Produkt vorwärts {dez(a.kol_fwd, 5)} gegen rückwärts {dez(a.kol_bwd, 5)}.{outside} {tail}")


# =====================================================================================================================================================
# Experiment 3: wie viele Ketten passen zu π?
# =====================================================================================================================================================
@lru_cache(maxsize=1)
def freedom_rows() -> tuple:
    """Zeilen für n = 3 bis 12: Zustände, Kanten, Kreise, Freiheitsgrade stationär (Rang numerisch bis n = 9, sonst Formel) und reversibel, ohne Graphbeschränkung."""
    rows = []
    for n in C.N_TABLE:
        ring = make_ring(n, 1.0)
        dims = H.flow_dimensions(ring, numeric=n <= C.N_RANK_MAX)
        full = H.flow_dimensions_all_moves(ring.size, numeric=n <= C.N_RANK_FULL_MAX)
        rows.append({"n": n, "V": dims["V"], "E": dims["E"], "kreise": dims["kreise"], "stationaer": dims["stationaer"], "reversibel": dims["reversibel"],
                     "formel": dims["stationaer_formel"], "rang_numerisch": dims["rang_numerisch"], "connected": H.graph_connected(ring),
                     "voll_stationaer": full["stationaer"], "voll_reversibel": full["reversibel"], "voll_numerisch": full["rang_numerisch"]})
    return tuple(rows)


def freedom_row(n: int) -> dict:
    return next(r for r in freedom_rows() if r["n"] == n)


def verdict_freedom(row: dict) -> tuple:
    """Verdikt zur Frage „Wie viele Freiheitsgrade haben die Ketten, die zu π passen?“ für den Ring mit n Plätzen."""
    how = "Rang der Erhaltungsgleichungen gerechnet" if row["rang_numerisch"] else "nach der Formel 2E − (V − 1), Rang nicht gerechnet"
    base = (f"n = {row['n']}: {row['V']} Zustände, {row['E']} Kanten: {row['stationaer']} stationäre Freiheitsgrade ({how}), {row['reversibel']} reversible, "
            f"{row['kreise']} Kreise")
    full = f"; ohne Graphbeschränkung (jeder Zug erlaubt) {row['voll_stationaer']} gegen {row['voll_reversibel']}"
    if row["kreise"] == 0:
        return "info", base + ". **Der Zustandsgraph ist ein Baum:** jede stationäre Kette auf diesem Graphen ist reversibel, es gibt keinen Kreisstrom" + full + "."
    return "success", base + f" (Kreisströme: Flüsse, die auf einem Kreis in einer Richtung zusätzlich laufen und jede Zustandsbilanz erhalten)" + full + "."


@dataclass(frozen=True, eq=False)
class CurrentScan:
    n: int
    lam: float
    factor: float
    share: float
    seed: int
    no_cycle: bool
    t_rev: object
    t_cur: tuple
    diffs: tuple
    lam2_rev: float
    lam2_cur: tuple
    lam2_not_above: int         # in so vielen Fällen ist |λ₂| höchstens so groß wie beim reversiblen Kern (auf vier Stellen)
    smax: float
    db_first: float
    P_first: object
    P_rev: np.ndarray


def current_seed(base_seed: int, j: int) -> int:
    return base_seed * C.CURRENT_SEED_STRIDE + j


def current_count(n: int) -> int:
    return C.N_CURRENTS if n <= C.N_CURRENTS_SMALL_MAX_N else C.N_CURRENTS_LARGE


def scan_currents(n: int, lam: float, factor: float, share: float, seed: int, count=None) -> CurrentScan:
    """`count` zufällige Kreisströme (Standard 20, ab n = 9 zehn; Zufalls-Seeds seed · 1000 + j) auf dem Zustandsgraphen: t_mix und |λ₂| gegen den reversiblen Kern mit Leitwert f · c_M."""
    count = current_count(n) if count is None else count
    ring = make_ring(n, lam)
    P_rev = K.metropolis(ring, factor)
    t_rev = H.t_mix(P_rev, ring.pi)
    l2_rev = H.second_eigenvalue_abs(P_rev)
    if len(K.cycle_basis(ring)) == 0:
        return CurrentScan(n, lam, factor, share, seed, True, t_rev, (), (), l2_rev, (), 0, 0.0, 0.0, None, P_rev)
    t_cur, l2_cur, first, smax0, db0 = [], [], None, 0.0, 0.0
    for j in range(count):
        P, _, smax = K.cycle_kernels(ring, factor, share, current_seed(seed, j))
        if j == 0:
            first, smax0, db0 = P, smax, H.db_deviation(P, ring.pi)
        t_cur.append(H.t_mix(P, ring.pi))
        l2_cur.append(H.second_eigenvalue_abs(P))
    diffs = tuple(t - t_rev for t in t_cur)
    return CurrentScan(n, lam, factor, share, seed, False, t_rev, tuple(t_cur), diffs, l2_rev, tuple(l2_cur),
                       sum(1 for v in l2_cur if round(v, 4) <= round(l2_rev, 4)), smax0, db0, first, P_rev)


def verdict_currents(a: CurrentScan) -> tuple:
    """Verdikt der Abtastung zufälliger Kreisströme: Änderung von t_mix gegen den reversiblen Kern (Zähleinheit Schritte)."""
    if a.no_cycle:
        return "info", (f"n = {a.n}: Der Zustandsgraph ist ein Baum, es gibt keinen Kreisstrom und damit keine nichtreversible Kette auf diesem Graphen "
                        f"(reversibler Kern: t_mix {mit_einheit(a.t_rev)}).")
    lo, hi = min(a.diffs), max(a.diffs)
    better = sum(1 for d in a.diffs if d < 0)
    worse = sum(1 for d in a.diffs if d > 0)
    if better == 0:
        tail = "Kein Kreisstrom ist hier schneller als der reversible Kern."
    elif better == len(a.diffs):
        tail = "Jeder dieser Kreisströme ist hier schneller als der reversible Kern."
    else:
        tail = "Mal kürzer, mal länger: nichtreversibel ist hier nicht von selbst schneller."
    return "info", (f"n = {a.n}, λ = {zahl(a.lam)}, reversibler Kern mit Faktor f = {dez(a.factor, 2)}: t_mix {mit_einheit(a.t_rev)}. "
                    f"{len(a.diffs)} zufällige Kreisströme (Anteil {dez(a.share, 2)} der größten zulässigen Stärke) ändern t_mix um **{vorzeichen(lo)} bis {vorzeichen(hi)} Schritte** "
                    f"(kürzer in {better}, länger in {worse}, gleich in {len(a.diffs) - better - worse} Fällen); |λ₂| ist in {a.lam2_not_above} von {len(a.diffs)} "
                    f"Fällen höchstens so groß wie beim reversiblen Kern. {tail}")


@lru_cache(maxsize=8)
def anchor_current_rows(seed: int) -> tuple:
    """Die sechs Zeilen n = 5, 6, 7 und λ = 2, 5 (Basisfaktor ½, größte zulässige Stärke, 20 zufällige Kreisströme)."""
    rows = []
    for n in (5, 6, 7):
        for lam in (2.0, 5.0):
            a = scan_currents(n, lam, 0.5, 1.0, seed)
            rows.append({"n": n, "lam": lam, "t_rev": a.t_rev, "min": min(a.diffs), "median": float(np.median(a.diffs)), "max": max(a.diffs)})
    return tuple(rows)


@dataclass(frozen=True, eq=False)
class HandCycle:
    s: float
    deviation: float
    p_empty_to_0: float
    p_empty_to_2: float
    p_empty_stay: float
    tmix_base: object
    tmix_cycle: object
    text: str


def hand_cycle_example() -> HandCycle:
    """Das Beispiel von Hand (n = 4, λ = 2): halber Metropolis-Leitwert als Basis, dazu der Kreisstrom s = 1/136 auf ∅ → {0} → {0,2} → {2} → ∅.

    Nur die vier Flüsse auf diesem Kreis ändern sich. Der Zustand {0} hat die Maske 1, {2} die Maske 4 und {0,2} die Maske 5."""
    ring = make_ring(4, 2.0)
    s = 1.0 / 136.0
    base = 0.5 * K.metropolis_conductance(ring)
    f_ab, f_ba = base.copy(), base.copy()
    cycle = [0, ring.index[1], ring.index[5], ring.index[4], 0]
    where = {(int(a), int(b)): k for k, (a, b) in enumerate(ring.edges)}
    for x, y in zip(cycle[:-1], cycle[1:]):
        if (x, y) in where:
            f_ab[where[(x, y)]] += s
            f_ba[where[(x, y)]] -= s
        else:
            f_ba[where[(y, x)]] += s
            f_ab[where[(y, x)]] -= s
    P = K.kernel_from_flows(ring, f_ab, f_ba)
    base_kernel = K.kernel_from_flows(ring, base, base)
    t_base, t_cycle = H.t_mix(base_kernel, ring.pi), H.t_mix(P, ring.pi)
    dev = H.db_deviation(P, ring.pi)
    text = (f"Von Hand (n = 4, λ = 2): halber Metropolis-Leitwert als reversibler Anteil, dazu der Kreisstrom s = 1/136 auf ∅ → {{0}} → {{0,2}} → {{2}} → ∅. "
            f"Nur vier Flüsse ändern sich: aus ∅ gehen {bruch(Fraction(P[0, ring.index[1]]).limit_denominator(1000))} nach {{0}} und "
            f"{bruch(Fraction(P[0, ring.index[4]]).limit_denominator(1000))} nach {{2}}, {bruch(Fraction(P[0, 0]).limit_denominator(1000))} bleiben; "
            f"Abweichung von Detailed Balance {bruch(Fraction(dev).limit_denominator(1000))}; t_mix {mit_einheit(t_cycle)} mit und {mit_einheit(t_base)} ohne Kreisstrom.")
    return HandCycle(s, dev, float(P[0, ring.index[1]]), float(P[0, ring.index[4]]), float(P[0, 0]), t_base, t_cycle, text)


# =====================================================================================================================================================
# Experiment 4: stationär, aber angekommen?
# =====================================================================================================================================================
@dataclass(frozen=True, eq=False)
class ArrivalAnalysis:
    chain: str
    n: int
    lam: float
    start_label: str
    size: int
    horizon: int
    d: np.ndarray               # d(t), t = 0 … horizon
    mu: np.ndarray              # TV(μ_t, π) vom Start
    db: float
    residual: float
    tmix: object
    classes: int
    burn: int
    tv_burn: float
    p: object                   # p4 (nur Drehung mit Drift und 5-Kreis)
    q: object
    tmix_metro: object          # Metropolis zum Vergleich (nur Mischung)
    tmix_p_half: object         # Kontrolle p = ½ (nur Mischung)


def horizon_for(tmix) -> int:
    return C.HORIZON_MIN if tmix is None else int(min(max(C.HORIZON_MIN, 4 * tmix), C.HORIZON_MAX))


def start_state(ring: Ring, start: str) -> int:
    return {C.START_EMPTY: ring.empty, C.START_ONE: ring.single(), C.START_FULL: ring.full}[start]


def analyse_arrival(chain: str, n: int, lam: float, q: float, p4: float, start: str, burn: int) -> ArrivalAnalysis:
    if chain == C.A_CYCLE5:
        P, pi, size, start_i, start_label = Y.kernel(p4), Y.PI5, Y.N5, 0, "beliebig (alle Zustände gleich)"
        tmix_metro = tmix_half = None
    else:
        ring = make_ring(n, lam)
        pi, size, start_i = ring.pi, ring.size, start_state(ring, start)
        start_label = f"{start}: {ring.label(start_i)}"
        P = {C.A_METRO: lambda: K.metropolis(ring), C.A_EQUAL: lambda: K.equal_conductance(ring), C.A_HEAT: lambda: K.heat_bath(ring),
             C.A_MIX: lambda: K.mixture(ring, q, p4), C.A_ROT: lambda: rotation_matrix(ring)}[chain]()
        tmix_metro = tmix_half = None
        if chain == C.A_MIX:
            tmix_metro = H.t_mix(K.metropolis(ring), pi)
            tmix_half = H.t_mix(K.mixture(ring, q, 0.5), pi)
    tmix = H.t_mix(P, pi)
    horizon = max(horizon_for(tmix), burn)
    mu = H.mu_curve(P, pi, start_i, horizon)
    uses_p = chain in (C.A_MIX, C.A_CYCLE5)
    return ArrivalAnalysis(
        chain=chain, n=n, lam=lam, start_label=start_label, size=size, horizon=horizon, d=H.d_curve(P, pi, horizon), mu=mu, db=H.db_deviation(P, pi),
        residual=H.stationary_residual(P, pi), tmix=tmix, classes=H.n_classes(P), burn=burn, tv_burn=float(mu[burn]), p=p4 if uses_p else None,
        q=q if chain == C.A_MIX else None, tmix_metro=tmix_metro, tmix_p_half=tmix_half,
    )


def _db_text(db: float) -> str:
    return "0" if db <= C.TOL else dez(db, 4)


def verdict_arrival(a: ArrivalAnalysis) -> tuple:
    """Verdikt zur Frage „Kommt sie an, und wie schnell?“: der Kern (Detailed Balance) ändert sich mit t nicht, die Verteilung nach t Schritten schon."""
    marks = [t for t in (0, 2, 10, 40) if t <= a.horizon]
    path = " / ".join(dez(a.mu[t], 4) for t in marks)
    at = " / ".join(str(t) for t in marks)
    if a.chain == C.A_CYCLE5:
        if a.tmix is None:
            return "warning", (f"5-Kreis, p = {zahl(a.p)}: π ist stationär und eindeutig ({a.classes} Klasse), aber d(t) = {dez(float(a.d.min()), 4)} für alle t = 0 bis "
                               f"{a.horizon}: **die Kette kommt nie an** (Periode 5). Die Detailed-Balance-Abweichung des Kerns ist {_db_text(a.db)}.")
        return "success", (f"5-Kreis, p = {zahl(a.p)}: π stationär, Detailed-Balance-Abweichung des Kerns {_db_text(a.db)} bei jedem t, t_mix = **{mit_einheit(a.tmix)}**; "
                           f"TV(μ_t, π) = {path} bei t = {at}.")
    if a.chain == C.A_ROT:
        return "warning", (f"Reine Drehung (n = {a.n}, λ = {zahl(a.lam)}): π ist stationär, aber die Kette hat **{a.classes} Klassen** und bleibt in der Klasse ihres Starts; "
                           f"d(t) ≥ {dez(float(a.d.min()), 4)} für alle t = 0 bis {a.horizon}, TV(μ_t, π) vom Start {a.start_label} {spanne(float(a.mu.min()), float(a.mu.max()))}. "
                           "Es gibt weitere stationäre Verteilungen: „π ist die stationäre Verteilung“ gilt nur bei einer einzigen Klasse.")
    base = (f"{a.chain} (n = {a.n}, λ = {zahl(a.lam)}, Start {a.start_label}): Detailed-Balance-Abweichung des Kerns **{_db_text(a.db)}** bei jedem t; "
            f"TV(μ_t, π) = {path} bei t = {at}; nach {a.burn} Schritten Burn-in {dez(a.tv_burn, 4)}; t_mix = {mit_einheit(a.tmix)}.")
    if a.chain == C.A_MIX:
        base += (f" Gewinn gegen reines Metropolis: t_mix {schritte(a.tmix_metro)} gegen {schritte(a.tmix)}; Kontrolle mit p = 0,5 (reversibel, gleiche Drehung): "
                 f"t_mix {schritte(a.tmix_p_half)}. Der Gewinn kommt von der Drehung als Zug, nicht von der Drift.")
    return "success", base + " Burn-in ändert die Verteilung nach t Schritten, nicht den Kern."


# =====================================================================================================================================================
# Kernfragen
# =====================================================================================================================================================
def kernfragen(build: BuildAnalysis, cyc: Cycle5Analysis, freedom: dict, arr: ArrivalAnalysis) -> list:
    """Die vier Kernfragen mit ihrer kurzen, live gerechneten Antwort: Liste von (Frage, Antwort)."""
    if build.residual <= C.TOL:
        a1 = f"Ja: ‖πP − π‖₁ < 10⁻⁹ ({build.kern}, n = {build.n}, λ = {zahl(build.lam)})."
    else:
        a1 = f"Nein: ‖πP − π‖₁ = {dez(build.residual, 4)}; die stationäre Verteilung liegt {dez(build.stationary_tv, 4)} von π entfernt."
    db = "Ja, Detailed Balance gilt" if build.db <= C.TOL else f"Nein, Abweichung {dez(build.db, 4)}"
    a2 = (f"{db} für diese Kette. Nötig ist es nicht: beim 5-Kreis mit p = {zahl(cyc.p)} ist die Gleichverteilung stationär bei Abweichung {dez(cyc.deviation, 4)}.")
    a3 = (f"Auf dem Ring mit n = {freedom['n']} Plätzen: {freedom['stationaer']} stationäre Freiheitsgrade, davon {freedom['reversibel']} reversible; "
          f"der Unterschied {freedom['kreise']} sind Kreisströme.")
    a4 = (f"{arr.chain}: t_mix = {mit_einheit(arr.tmix)}" + (f", {arr.classes} Klassen" if arr.classes > 1 else "")
          + f"; Detailed-Balance-Abweichung des Kerns {_db_text(arr.db)}, unabhängig von t.")
    return [("Hat diese Kette π als stationäre Verteilung?", a1), ("Erfüllt sie Detailed Balance, und muss sie?", a2),
            ("Wie viele Freiheitsgrade haben die Ketten, die zu π passen?", a3), ("Kommt sie an, und wie schnell?", a4)]

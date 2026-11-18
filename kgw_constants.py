"""Konstanten: Regler-Grenzen, feste Annahmen, Farben, Presets und Namen der Kerne."""

# ---- Ring und Gewicht ---------------------------------------------------------------------------------------------------------------------
N_MIN, N_MAX, DEFAULT_N = 3, 10, 4          # Plätze im Ring; n = 10 hat 123 Zustände (die größte Matrix, die live gerechnet wird)
LAM_MIN, LAM_MAX, LAM_STEP, DEFAULT_LAM = 0.5, 5.0, 0.5, 2.0
SEED_MAX, DEFAULT_SEED = 999999, 11001      # Zufalls-Seed der Simulationen und der zufälligen Kreisströme
N_TABLE = tuple(range(3, 13))               # Tabelle der Freiheitsgrade: n = 3 bis 12 (Zählung des Zustandsgraphen)
N_RANK_MAX = 9                              # bis hierhin wird der Rang der Erhaltungsgleichungen numerisch gerechnet, darüber gilt die Formel
N_RANK_FULL_MAX = 7                         # „jeder Zug erlaubt“: Rang numerisch bis n = 7 (29 Zustände, 812 Flüsse)
TOL = 1e-9                                  # ab dieser Abweichung gilt „πP = π“ bzw. Detailed Balance als verletzt
EPS_MIX = 0.25                              # Schwelle der Mischzeit t_mix = t_mix(1/4)
T_MAX_POW = 22                              # t_mix wird bis 2^22 Schritte gesucht (darüber: „kommt nicht an“)

# ---- die Kerne der Auswahl ----------------------------------------------------------------------------------------------------------------
K_METRO, K_HEAT, K_EQUAL, K_STAGED = "Metropolis", "Heat-bath", "Gleiche Leitwerte", "Eigene Leitwerte (Faktoren f₀, f₁)"
K_HALF, K_INVERSE, K_HEAT_INV = "Fehler: feste Annahme ½", "Fehler: umgekehrtes Verhältnis", "Fehler: Heat-bath mit 1/(1+λ)"
KERN_OPTIONS = (K_METRO, K_HEAT, K_EQUAL, K_STAGED, K_HALF, K_INVERSE, K_HEAT_INV)
KERN_FEHLER = (K_HALF, K_INVERSE, K_HEAT_INV)
F_MIN, F_MAX, F_STEP, DEFAULT_F = 0.05, 1.0, 0.05, 1.0   # Leitwertfaktoren f₀ (leer ↔ ein Platz) und f₁ (alle höheren Stufen)

# ---- Experiment 2: 5-Kreis mit Drift ------------------------------------------------------------------------------------------------------
P5_MIN, P5_MAX, P5_STEP, DEFAULT_P5 = 0.5, 1.0, 0.05, 0.7
STEPS_OPTIONS = (10_000, 100_000, 1_000_000)
DEFAULT_STEPS = 100_000                     # die Voreinstellung rechnet schnell; 10⁶ Schritte sind die Messung der README

# ---- Experiment 3: Kreisströme ------------------------------------------------------------------------------------------------------------
S_MIN, S_MAX, S_STEP, DEFAULT_S = 0.0, 1.0, 0.05, 1.0     # Anteil des größten zulässigen Kreisstroms
FB_MIN, FB_MAX, FB_STEP, DEFAULT_FB = 0.1, 1.0, 0.05, 0.5  # Basisfaktor f: der reversible Anteil ist f · (Metropolis-Leitwert)
N_CURRENTS = 20                             # zufällige Kreisströme je Zeile der Abtastung (n ≤ 8)
N_CURRENTS_LARGE = 10                       # ab n = 9 nur so viele (die Matrizen haben 76 und 123 Zustände: sonst über 2 Sekunden je Eingabe)
N_CURRENTS_SMALL_MAX_N = 8
CURRENT_SEED_STRIDE = 1000

# ---- Experiment 4: Ankunft ----------------------------------------------------------------------------------------------------------------
A_METRO, A_EQUAL, A_HEAT = "Metropolis", "Gleiche Leitwerte", "Heat-bath"
A_MIX, A_ROT, A_CYCLE5 = "Metropolis mit Drehung und Drift", "Reine Drehung", "5-Kreis mit Drift"
ARRIVAL_OPTIONS = (A_METRO, A_EQUAL, A_HEAT, A_MIX, A_ROT, A_CYCLE5)
Q_MIN, Q_MAX, Q_STEP, DEFAULT_Q = 0.05, 0.5, 0.05, 0.2
P4_MIN, P4_MAX, P4_STEP, DEFAULT_P4 = 0.5, 1.0, 0.05, 1.0
START_EMPTY, START_ONE, START_FULL = "leerer Zustand", "ein Platz belegt: {0}", "Maximalbelegung"
START_OPTIONS = (START_EMPTY, START_ONE, START_FULL)
BURN_MIN, BURN_MAX, DEFAULT_BURN = 0, 100, 10
HORIZON_MIN, HORIZON_MAX = 60, 400          # Länge der gezeigten Verläufe in Schritten

# ---- Simulation der Zeitanteile (Experiment 1) ---------------------------------------------------------------------------------------------
TIME_STEPS_OPTIONS = (10_000, 50_000, 200_000)
DEFAULT_TIME_STEPS = 50_000
NOISE_REPS = 4

# ---- Farben -------------------------------------------------------------------------------------------------------------------------------
COLOR_A = "#2F6B65"       # Zielverteilung / Hauptkern
COLOR_B = "#d97706"       # Vergleich
COLOR_C = "#4c78a8"
COLOR_LINE = "#9aa0a6"

# ---- Presets ------------------------------------------------------------------------------------------------------------------------------
PRESETS = {
    "Metropolis (n = 4, λ = 2)": {},
    "Standardfall (n = 8, λ = 2)": {"n": 8, "lam": 2.0},
    "Kalt (n = 8, λ = 5)": {"n": 8, "lam": 5.0},
    "Gleiche Leitwerte (n = 10, λ = 5)": {"n": 10, "lam": 5.0, "kern": K_EQUAL},
    "Heat-bath (n = 6, λ = 2)": {"n": 6, "lam": 2.0, "kern": K_HEAT},
    "Falsch gebaut: feste Annahme ½ (n = 6, λ = 2)": {"n": 6, "lam": 2.0, "kern": K_HALF},
    "5-Kreis mit Drift 0,7": {"p5": 0.7, "steps": 1_000_000},
    "Reine Drehung (n = 5, λ = 2)": {"n": 5, "lam": 2.0, "arrival": A_ROT},
    "Kreisstrom (n = 5, λ = 2)": {"n": 5, "lam": 2.0},
    "Kern gegen Burn-in (n = 6, λ = 2)": {"n": 6, "lam": 2.0, "arrival": A_MIX},
}
PRESET_HELP = {
    "Metropolis (n = 4, λ = 2)": "Das Beispiel von Hand: 7 Zustände, Gewichte 1, 2, 2, 2, 2, 4, 4, Normierung 17. Kern: Metropolis.",
    "Standardfall (n = 8, λ = 2)": "Der gemeinsame Standardfall der Nachbar-Demos: 8 Plätze im Ring, 47 Zustände, λ = 2.",
    "Kalt (n = 8, λ = 5)": "Ein großes λ legt fast die ganze Verteilung auf die Zustände mit vielen belegten Plätzen: die Ketten brauchen länger.",
    "Gleiche Leitwerte (n = 10, λ = 5)": "Derselbe Fluss auf jeder Kante des Zustandsgraphen: π bleibt stationär, aber die Kette braucht tausende Schritte.",
    "Heat-bath (n = 6, λ = 2)": "Heat-bath (Gibbs-Sampler): ein Platz wird gleichverteilt gewählt und aus seiner bedingten Verteilung neu gesetzt.",
    "Falsch gebaut: feste Annahme ½ (n = 6, λ = 2)": "Jeder zulässige Vorschlag wird mit Wahrscheinlichkeit ½ angenommen: läuft, aber die Prüfung πP = π schlägt an.",
    "5-Kreis mit Drift 0,7": "Eine Irrfahrt auf dem Kreis mit fünf Zuständen, vorwärts mit Wahrscheinlichkeit 0,7: Gleichverteilung stationär ohne Detailed Balance (Zufalls-Seed 11001, 10⁶ Schritte).",
    "Reine Drehung (n = 5, λ = 2)": "Der Ring wird nur gedreht: π ist stationär, aber die Kette bleibt in ihrer Klasse und kommt nicht an.",
    "Kreisstrom (n = 5, λ = 2)": "Zufällige Kreisströme auf dem Zustandsgraphen des Rings mit 5 Plätzen: nichtreversible Ketten mit demselben π.",
    "Kern gegen Burn-in (n = 6, λ = 2)": "Metropolis mit Drehung und Drift: Der Kern verletzt Detailed Balance bei jedem t, die Verteilung nach t Schritten nähert sich π trotzdem.",
}

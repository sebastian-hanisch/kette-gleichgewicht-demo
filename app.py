"""Eine Kette mit gewünschtem Gleichgewicht: bauen, prüfen, und was Detailed Balance nicht sagt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein ausgedachtes Modell (Hard-Core-Ring: Belegung von Stellplätzen, keine zwei benachbarten belegt) mit exakter Verteilung π. Gezeigt werden Kerne zu diesem π,
die Prüfung πP = π als Zahl, die Menge aller passenden Ketten (Freiheitsgrade), nichtreversible Ketten und der Unterschied zwischen stationär, Detailed Balance und angekommen.

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import kgw_constants as C
import kgw_evaluation as E
import kgw_texts as T
import kgw_visualization as V
from kgw_format import dez, ganz, schritte, zahl
from kgw_layout import dunkles_design, kopf_css
from kgw_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)

st.set_page_config(page_title="Kette und Gleichgewicht – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _build(n, lam, kern, f0, f1):
    return E.analyse_build(n, lam, kern, f0, f1)


@st.cache_data(show_spinner=False)
def _time_fractions(n, lam, kern, f0, f1, steps, seed):
    return E.analyse_time_fractions(n, lam, kern, f0, f1, steps, seed)


@st.cache_data(show_spinner=False)
def _cycle5(p, steps, seed):
    return E.analyse_cycle5(p, steps, seed)


@st.cache_data(show_spinner=False)
def _scan(n, lam, fbase, share, seed):
    return E.scan_currents(n, lam, fbase, share, seed)


@st.cache_data(show_spinner=False)
def _anchor(seed):
    return E.anchor_current_rows(seed)


@st.cache_data(show_spinner=False)
def _arrival(chain, n, lam, q, p4, start, burn):
    return E.analyse_arrival(chain, n, lam, q, p4, start, burn)


@st.cache_data(show_spinner=False)
def _hand_example():
    return E.hand_cycle_example()


def show(verdict):
    kind, text = verdict
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kgw_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title(T.TITLE)
st.markdown(T.INTRO)

with st.expander(T.HOW_TITLE, expanded=True):
    st.markdown(T.HOW)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Ring**")
    n = st.slider("Plätze im Ring n", *bounds("n_slider"), key="n_slider", help="Zahl der Stellplätze im Ring; n = 10 hat 123 Zustände. Gilt für alle Experimente außer dem 5-Kreis.")
    lam = st.slider("Gewicht λ je belegtem Platz", *bounds("lam_slider"), step=C.LAM_STEP, key="lam_slider",
                    help="π(x) ist proportional zu λ hoch Zahl der belegten Plätze. Bei λ = 1 sind alle Zustände gleich wahrscheinlich.")
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed (Simulation, Kreisströme)", *bounds("seed_input"), key="seed_input", step=1,
                           help="Gleicher Zufalls-Seed, gleiche Folge, auf jeder Plattform (ganzzahliger SplitMix64).")
    st.button("🎲 Neuer Zufalls-Seed", width="stretch", on_click=randomize_seed)

n, lam, seed = int(n), float(lam), int(seed)
st.caption(T.MODEL_NOTE)

# --- Ergebnis: die gebaute Kette ------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis: die gebaute Kette")
kern = st.selectbox("Kern", C.KERN_OPTIONS, key="kern_select",
                    help="Drei Regeln mit π als stationärer Verteilung, eigene Leitwerte, und drei Fehlerfälle (Zeilensumme 1, aber π wird verfehlt).")
c_f0, c_f1 = st.columns(2)
staged = kern == C.K_STAGED
f0 = c_f0.slider("Faktor f₀ (leer ↔ ein Platz)", *bounds("f0_slider"), step=C.F_STEP, key="f0_slider", disabled=not staged,
                 help="Nur für „Eigene Leitwerte“: Metropolis-Leitwert mal f₀ auf den Kanten zwischen dem leeren Ring und einem belegten Platz.")
f1 = c_f1.slider("Faktor f₁ (alle höheren Stufen)", *bounds("f1_slider"), step=C.F_STEP, key="f1_slider", disabled=not staged or n < 4,
                 help="Nur für „Eigene Leitwerte“ und n ≥ 4: Metropolis-Leitwert mal f₁ auf allen Kanten von k belegten zu k + 1 belegten Plätzen, k ≥ 1.")
build = _build(n, lam, kern, float(f0), float(f1))

m1, m2, m3, m4 = st.columns(4)
m1.metric("π stationär?", "ja" if build.residual <= C.TOL else "nein")
m2.metric("‖πP − π‖₁", "< 10⁻⁹" if build.residual <= C.TOL else dez(build.residual, 4))
m3.metric("Detailed-Balance-Abweichung", "< 10⁻⁹" if build.db <= C.TOL else dez(build.db, 4))
m4.metric("Mischzeit t_mix (Schritte)", schritte(build.tmix))
st.plotly_chart(V.build_matrix(build.P, build.labels), width="stretch", key="matrix_chart")
st.caption(f"Übergangsmatrix des Kerns: Zeile = von x, Spalte = nach y; {build.size} Zustände, Zustand ∅ ist der leere Ring, {{0,2}} die Belegung der Plätze 0 und 2.")
show(E.verdict_build(build))

# --- Kernfragen (der Platz bleibt hier reserviert, die Antworten kommen aus den Experimenten weiter unten) -------------------------------------

st.markdown("## ❓ Kernfragen")
kernfragen_box = st.container()

st.markdown("---")

# --- Experiment 1 -------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Eine Kette bauen und prüfen")
st.caption(T.CAPTION_BUILD)
col_a, col_b = st.columns(2)
if build.stationary is not None:
    col_a.plotly_chart(V.build_distributions(build.pi, build.stationary, build.labels, "stationäre Verteilung der Kette (lineares System)"), width="stretch", key="dist_chart")
col_b.plotly_chart(V.build_residual(build.residual_vec, build.labels), width="stretch", key="residual_chart")
row_text = "größter Fehler unter 10⁻⁹" if build.row_err < C.TOL else f"größter Fehler {build.row_err:.1e}".replace(".", ",")
entry_text = "alle Einträge ≥ 0" if build.min_entry >= -1e-12 else f"kleinster Eintrag {build.min_entry:.1e}".replace(".", ",")
st.caption(f"Links: π neben der stationären Verteilung der gebauten Kette, die ein lineares System liefert (Abstand zu π {dez(build.stationary_tv, 4) if build.stationary_tv is not None else '–'}). "
           f"Rechts: (πP − π)(x) je Zustand. Zeilensummen: {row_text}, {entry_text}.")
r1, r2, r3 = st.columns(3)
for col, name in zip((r1, r2, r3), (C.K_METRO, C.K_HEAT, C.K_EQUAL)):
    col.metric(f"{name}: t_mix", f"{schritte(build.refs[name])} Schritte")
st.caption("Drei Kerne mit demselben π, drei Mischzeiten (n und λ aus der Seitenleiste, Zähleinheit Schritte bis d(t) ≤ 1/4).")
tf_steps = st.select_slider("Länge der Simulation (Schritte)", options=C.TIME_STEPS_OPTIONS, key="time_steps_select", format_func=ganz,
                            help="Eine einzige Kette, Start aus π, gezählt wird der Anteil der Zeit in jedem Zustand.")
with st.spinner(f"Simuliere {ganz(tf_steps)} Schritte …"):
    tf = _time_fractions(n, lam, kern, float(f0), float(f1), int(tf_steps), seed)
show(E.verdict_time_fractions(tf, build.residual))

st.markdown("---")

# --- Experiment 2 -------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Detailed Balance ist hinreichend, nicht notwendig")
st.caption(T.CAPTION_CYCLE5)
c_p, c_t = st.columns(2)
p5 = c_p.slider("Vorwärtswahrscheinlichkeit p", *bounds("p5_slider"), step=C.P5_STEP, key="p5_slider", help="Wahrscheinlichkeit, auf dem Kreis mit fünf Zuständen einen Schritt vorwärts zu gehen.")
steps = c_t.select_slider("Länge der Simulation (Schritte)", options=C.STEPS_OPTIONS, key="steps_select", format_func=ganz,
                          help="Eine einzige Kette; gemessen wird der Nettofluss je Kante. 10⁶ Schritte sind die Messung der README (Zufalls-Seed 11001).")
with st.spinner(f"Simuliere {ganz(steps)} Schritte …"):
    cyc5 = _cycle5(float(p5), int(steps), seed)
x1, x2, x3, x4 = st.columns(4)
x1.metric("Abweichung von Detailed Balance", dez(cyc5.deviation, 4))
x2.metric("Nettofluss je Kante (exakt)", dez(cyc5.net_exact, 4))
x3.metric("Nettofluss (gemessen, Mittel)", dez(float(cyc5.net_measured.mean()), 4))
x4.metric("t_mix (Schritte), |λ₂|", f"{schritte(cyc5.tmix)}, {dez(cyc5.lam2, 4)}")
st.plotly_chart(V.build_cycle5(cyc5.p, cyc5.net_exact, float(cyc5.net_measured.mean())), width="stretch", key="cycle5_chart")
show(E.verdict_cycle5(cyc5))
cp1, cp2 = st.columns(2)
cp1.markdown("**Kern P** (Zeile = von, Spalte = nach)")
cp1.dataframe(pd.DataFrame(cyc5.P, columns=[str(i) for i in range(5)]).round(4), width="stretch")
cp2.markdown("**Zeitumkehrte Kette P\\*** (dasselbe π, Drift rückwärts)")
cp2.dataframe(pd.DataFrame(cyc5.P_star, columns=[str(i) for i in range(5)]).round(4), width="stretch")

st.markdown("---")

# --- Experiment 3 -------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Wie viele Freiheitsgrade haben die Ketten, die zu π passen?")
st.caption(T.CAPTION_FREEDOM)
rows = E.freedom_rows()
frame = pd.DataFrame([{
    "n": r["n"], "Zustände V": r["V"], "Kanten E": r["E"], "Kreise E − V + 1": r["kreise"], "stationär": r["stationaer"],
    "Rang": "gerechnet" if r["rang_numerisch"] else "Formel", "reversibel": r["reversibel"], "jeder Zug: stationär": r["voll_stationaer"],
    "jeder Zug: reversibel": r["voll_reversibel"]} for r in rows])
st.dataframe(frame, hide_index=True, width="stretch")
st.plotly_chart(V.build_freedom(rows), width="stretch", key="freedom_chart")
show(E.verdict_freedom(E.freedom_row(n)))

st.markdown("**Nichtreversible Ketten zum selben π:** ein zufälliger Kreisstrom auf dem Zustandsgraphen, auf die größte zulässige Stärke skaliert.")
c_s, c_fb = st.columns(2)
share = c_s.slider("Anteil des größten zulässigen Kreisstroms", *bounds("share_slider"), step=C.S_STEP, key="share_slider",
                   help="0 ist der reversible Basiskern, 1 die größte Stärke, bei der alle Flüsse nichtnegativ bleiben.")
fbase = c_fb.slider("Basisfaktor f (reversibler Anteil)", *bounds("fbase_slider"), step=C.FB_STEP, key="fbase_slider",
                    help="Der reversible Anteil ist f mal der Metropolis-Leitwert; die Hälfte lässt Platz für den Kreisstrom.")
with st.spinner(f"Rechne {E.current_count(n)} zufällige Kreisströme …"):
    scan = _scan(n, lam, float(fbase), float(share), seed)
show(E.verdict_currents(scan))
if not scan.no_cycle:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Abweichung von Detailed Balance (Strom Nr. 0)", dez(scan.db_first, 4))
    k2.metric("t_mix mit Kreisstrom (Strom Nr. 0)", schritte(scan.t_cur[0]), delta=f"{scan.t_cur[0] - scan.t_rev:+d} gegen reversibel".replace("-", "−") if scan.t_rev is not None and scan.t_cur[0] is not None else None,
              delta_color="inverse")
    k3.metric("t_mix reversibel (Basiskern)", schritte(scan.t_rev))
    k4.metric("|λ₂| (Strom Nr. 0 / reversibel)", f"{dez(scan.lam2_cur[0], 4)} / {dez(scan.lam2_rev, 4)}")
with st.spinner("Rechne die sechs Zeilen n = 5 bis 7 …"):
    anchor = _anchor(seed)
st.plotly_chart(V.build_currents(anchor), width="stretch", key="currents_chart")
st.caption("Änderung von t_mix durch 20 zufällige Kreisströme gegen den reversiblen Kern (Basisfaktor ½, größte zulässige Stärke), Spanne und Median je Zeile; Zufalls-Seed aus der Seitenleiste.")
st.caption(_hand_example().text)

st.markdown("---")

# --- Experiment 4 -------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Stationär, aber angekommen?")
st.caption(T.CAPTION_ARRIVAL)
chain = st.selectbox("Kette", C.ARRIVAL_OPTIONS, key="arrival_select",
                     help="Drei Kerne mit Detailed Balance, die Mischung mit Drehung und Drift (nichtreversibel), die reine Drehung des Rings und der 5-Kreis mit Drift.")
d1, d2, d3, d4 = st.columns(4)
q = d1.slider("Anteil q der Drehung", *bounds("q_slider"), step=C.Q_STEP, key="q_slider", disabled=chain != C.A_MIX,
              help="Nur für die Mischung: Anteil der Schritte, die den Ring drehen.")
p4 = d2.slider("Wahrscheinlichkeit p der Drehung", *bounds("p4_slider"), step=C.P4_STEP, key="p4_slider", disabled=chain not in (C.A_MIX, C.A_CYCLE5),
               help="Für die Mischung und den 5-Kreis: Wahrscheinlichkeit, vorwärts zu drehen (1 − p: rückwärts). p = 0,5 ist reversibel.")
start = d3.selectbox("Start", C.START_OPTIONS, key="start_select", disabled=chain == C.A_CYCLE5, help="Zustand, aus dem die Verteilung nach t Schritten gerechnet wird. Der 5-Kreis ist symmetrisch.")
burn = d4.slider("Burn-in t", *bounds("burn_slider"), key="burn_slider", help="Zahl der Schritte, nach denen die Verteilung mit π verglichen wird.")
if chain == C.A_CYCLE5:
    st.caption("Für den 5-Kreis gelten n, λ und der Start nicht.")
arrival = _arrival(chain, n, lam, float(q), float(p4), start, int(burn))
a1, a2, a3, a4 = st.columns(4)
a1.metric("Detailed-Balance-Abweichung des Kerns", "0" if arrival.db <= C.TOL else dez(arrival.db, 4))
a2.metric("t_mix (Schritte)", schritte(arrival.tmix))
a3.metric("Klassen", str(arrival.classes))
a4.metric(f"TV(μ_t, π) bei t = {arrival.burn}", dez(arrival.tv_burn, 4))
st.plotly_chart(V.build_arrival(arrival), width="stretch", key="arrival_chart")
show(E.verdict_arrival(arrival))

sync_query_params({
    "n_slider": n, "lam_slider": lam, "seed_input": seed, "kern_select": kern, "f0_slider": float(f0), "f1_slider": float(f1), "time_steps_select": int(tf_steps),
    "p5_slider": float(p5), "steps_select": int(steps), "share_slider": float(share), "fbase_slider": float(fbase), "arrival_select": chain, "q_slider": float(q),
    "p4_slider": float(p4), "start_select": start, "burn_slider": int(burn),
})

with kernfragen_box:
    for question, answer in E.kernfragen(build, cyc5, E.freedom_row(n), arrival):
        st.markdown(f"**{question}** {answer}")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(T.LIMITS_MARKDOWN)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(T.MATH_LESEHILFE)
    st.markdown(T.MATH_FORMULAS)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [MCMC: von der Kette zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-mcmc.html)."
)

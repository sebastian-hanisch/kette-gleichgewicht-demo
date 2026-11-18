"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import numpy as np
import plotly.graph_objects as go

import kgw_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.25) if legend else dict(visible=False))
    return lock_axes(fig)


def build_matrix(P: np.ndarray, labels) -> go.Figure:
    """Übergangsmatrix als Heatmap (Zeile = von, Spalte = nach); bis 11 Zustände mit den Zahlen in den Feldern."""
    size = len(labels)
    names = [str(l) for l in labels]
    text = [[f"{v:.3f}" if v > 1e-12 else "" for v in row] for row in P] if size <= 11 else None
    fig = go.Figure(go.Heatmap(z=P, x=names, y=names, zmin=0.0, zmax=float(P.max()), colorscale="Blues", text=text, texttemplate="%{text}" if text else None,
                               hovertemplate="von %{y} nach %{x}: %{z:.4f}<extra></extra>", colorbar=dict(title="P(x, y)", thickness=12)))
    fig.update_yaxes(autorange="reversed", title_text="von x", type="category", showticklabels=size <= 18)
    fig.update_xaxes(title_text="nach y", type="category", side="top", showticklabels=size <= 18)
    return _base(fig, height=420 if size <= 11 else 520, legend=False)


def build_distributions(pi: np.ndarray, other: np.ndarray, labels, other_name: str) -> go.Figure:
    """π neben der stationären Verteilung, die aus dem linearen System folgt (gruppierte Balken je Zustand)."""
    names = [str(l) for l in labels]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=pi, name="Zielverteilung π", marker_color=C.COLOR_A, hovertemplate="%{x}: %{y:.5f}<extra></extra>"))
    fig.add_trace(go.Bar(x=names, y=other, name=other_name, marker_color=C.COLOR_B, hovertemplate="%{x}: %{y:.5f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Wahrscheinlichkeit")
    fig.update_xaxes(type="category", showticklabels=len(names) <= 18)
    return _base(fig)


def build_residual(vec: np.ndarray, labels) -> go.Figure:
    """πP − π je Zustand (null in jedem Zustand genau dann, wenn π stationär ist)."""
    names = [str(l) for l in labels]
    fig = go.Figure(go.Bar(x=names, y=vec, marker_color=C.COLOR_C, hovertemplate="%{x}: %{y:.5f}<extra></extra>"))
    fig.update_yaxes(title_text="(πP − π)(x)", zeroline=True)
    fig.update_xaxes(type="category", showticklabels=len(names) <= 18)
    return _base(fig, legend=False)


def build_cycle5(p: float, net_exact: float, net_measured: float) -> go.Figure:
    """Die fünf Zustände auf dem Kreis; außen Vorwärtsschritte (p), innen Rückwärtsschritte (1 − p), in der Mitte der Nettofluss je Kante."""
    ang = [math.pi / 2 - 2 * math.pi * i / 5 for i in range(5)]
    xs, ys = [math.cos(a) for a in ang], [math.sin(a) for a in ang]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="markers+text", text=[str(i) for i in range(5)], textposition="middle center", textfont=dict(color="white", size=14),
                               marker=dict(size=34, color=C.COLOR_A), hoverinfo="skip"))
    for i in range(5):
        j = (i + 1) % 5
        mx, my = (xs[i] + xs[j]) / 2, (ys[i] + ys[j]) / 2
        nx, ny = mx / math.hypot(mx, my), my / math.hypot(mx, my)
        for sign, prob, color in ((1, p, C.COLOR_B), (-1, 1 - p, C.COLOR_LINE)):
            off = 0.17 * sign
            fx, fy = xs[i] + (nx * off), ys[i] + (ny * off)
            tx, ty = xs[j] + (nx * off), ys[j] + (ny * off)
            if prob <= 1e-12:
                continue
            a0, b0, a1, b1 = (fx, fy, tx, ty) if sign == 1 else (tx, ty, fx, fy)
            shrink = 0.20
            dx, dy = a1 - a0, b1 - b0
            L = math.hypot(dx, dy)
            fig.add_annotation(x=a1 - dx / L * shrink, y=b1 - dy / L * shrink, ax=a0 + dx / L * shrink, ay=b0 + dy / L * shrink, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowwidth=1 + 3 * prob, arrowcolor=color, text="")
            fig.add_annotation(x=mx + nx * (0.27 * sign), y=my + ny * (0.27 * sign), text=f"{prob:.2f}".replace(".", ","), showarrow=False, font=dict(size=11, color=color))
    fig.add_annotation(x=0, y=0, text=f"Nettofluss je Kante<br>exakt {net_exact:.4f}<br>gemessen {net_measured:.4f}".replace(".", ","), showarrow=False, font=dict(size=11))
    # gleicher Maßstab: scaleanchor nur mit autorange; den Rand erzwingen zwei unsichtbare Marker an den Ecken (DEMO-PLAYBOOK §3)
    fig.add_trace(go.Scatter(x=[-1.9, 1.9], y=[-1.45, 1.45], mode="markers", marker=dict(opacity=0), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(visible=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False)
    return _base(fig, height=380, legend=False)


def build_freedom(rows) -> go.Figure:
    """Freiheitsgrade der Flüsse je n: reversibel (E) plus Kreisströme (E − V + 1) ergibt die Zahl der stationären."""
    ns = [str(r["n"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ns, y=[r["reversibel"] for r in rows], name="reversibel (Detailed Balance)", marker_color=C.COLOR_A,
                         hovertemplate="n = %{x}: %{y} reversible<extra></extra>"))
    fig.add_trace(go.Bar(x=ns, y=[r["kreise"] for r in rows], name="zusätzlich Kreisströme", marker_color=C.COLOR_B,
                         hovertemplate="n = %{x}: %{y} Kreise<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text="Plätze im Ring n", type="category")
    fig.update_yaxes(title_text="Freiheitsgrade (stationär = Summe)")
    return _base(fig)


def build_currents(rows) -> go.Figure:
    """Änderung von t_mix durch zufällige Kreisströme gegen den reversiblen Kern: Spanne (kleinster bis größter Wert) und Median je Zeile."""
    labels = [f"n = {r['n']}, λ = {r['lam']:g}".replace(".", ",") for r in rows]
    med = [r["median"] for r in rows]
    fig = go.Figure(go.Scatter(x=labels, y=med, mode="markers", marker=dict(size=10, color=C.COLOR_B),
                               error_y=dict(type="data", symmetric=False, array=[r["max"] - r["median"] for r in rows], arrayminus=[r["median"] - r["min"] for r in rows],
                                            color=C.COLOR_B, thickness=2, width=8),
                               hovertemplate="%{x}: Median %{y:.1f}<extra></extra>"))
    fig.add_hline(y=0, line_color=C.COLOR_LINE, line_dash="dot")
    fig.update_yaxes(title_text="Änderung von t_mix (Schritte)")
    fig.update_xaxes(type="category")
    return _base(fig, legend=False)


def build_arrival(a) -> go.Figure:
    """d(t) und TV(μ_t, π) vom Start (logarithmisch), die Detailed-Balance-Abweichung des Kerns als waagerechte Gerade, der Burn-in als senkrechte."""
    floor = 1e-6
    t = np.arange(len(a.d))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=np.maximum(a.d, floor), mode="lines", name="d(t): schlechtester Start", line=dict(color=C.COLOR_A, width=3),
                             hovertemplate="t = %{x}: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=t, y=np.maximum(a.mu, floor), mode="lines", name="TV(μ_t, π) vom gewählten Start", line=dict(color=C.COLOR_B, width=2, dash="dash"),
                             hovertemplate="t = %{x}: %{y:.4f}<extra></extra>"))
    fig.add_hline(y=0.25, line_color=C.COLOR_LINE, line_dash="dot", annotation_text="Schwelle 1/4", annotation_position="top right")
    if a.db > C.TOL:
        fig.add_hline(y=a.db, line_color=C.COLOR_C, line_width=3, annotation_text=f"Detailed-Balance-Abweichung des Kerns {a.db:.4f}".replace(".", ","),
                      annotation_position="bottom right")
    else:
        fig.add_annotation(x=0.99, y=0.03, xref="paper", yref="paper", xanchor="right", showarrow=False, text="Detailed-Balance-Abweichung des Kerns: 0 (nicht darstellbar)")
    fig.add_vline(x=a.burn, line_color=C.COLOR_LINE, line_dash="dash", annotation_text=f"Burn-in {a.burn}", annotation_position="top")
    fig.update_xaxes(title_text="Schritte t", range=[0, len(a.d) - 1])
    fig.update_yaxes(title_text="Totalvariationsabstand", type="log", range=[-6, 0.05])
    return _base(fig, height=380)

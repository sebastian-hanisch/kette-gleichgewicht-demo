"""Feste Texte der App: Einleitung, Erklärung, Grenzen, Formeln. Zahlen stehen hier nur, wenn die README sie nennt und tests/test_claims.py sie prüft."""

TITLE = "🔗 Eine Kette mit gewünschtem Gleichgewicht: bauen, prüfen, und was Detailed Balance nicht sagt"

INTRO = """
Zu einer vorgegebenen Verteilung π gibt es nicht eine Kette, sondern viele. Diese Demo **baut Ketten** (Metropolis, Heat-bath, gleiche Leitwerte) für ein ausgedachtes Modell, die
**Belegung von Stellplätzen im Ring** (kein Paar benachbarter Plätze belegt, Gewicht λ^k bei k belegten Plätzen), und **prüft sie als Zahl**: stationäre Verteilung (πP = π),
Detailed Balance, Zahl der passenden Ketten (Freiheitsgrade) und Ankunft d(t). Sie trennt drei Dinge, die man leicht verwechselt: **stationär**, **Detailed Balance** und **angekommen**.
Alles ist Lehrbuchstoff auf kleinen, exakt lösbaren Räumen; neu ist nur die Vorführbarkeit mit Zähleinheit und exakten Orakeln.

Weiter unten: die **❓ Kernfragen**, vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""

HOW_TITLE = "So funktioniert Kette und Gleichgewicht"
HOW = """
1. **Ziel vorgeben:** ein Zustand ist die Menge der belegten Plätze im Ring, sein Gewicht ist λ^k bei k belegten Plätzen, π(x) ist das Gewicht geteilt durch die Summe aller Gewichte.
2. **Fluss festlegen:** auf jeder Kante des Zustandsgraphen (zwei Zustände, die sich in genau einem Platz unterscheiden) steht ein Leitwert c(x, y) = π(x)·P(x, y).
   Die Summe der Leitwerte aus einem Zustand heraus darf π(x) nicht überschreiten, den Rest behält die Kette.
3. **Symmetrisch wählen:** c(x, y) = c(y, x) heißt Detailed Balance und lässt π stationär sein. Metropolis, Heat-bath und gleiche Leitwerte sind drei solche Wahlen.
4. **Prüfen:** ‖πP − π‖₁ ist null genau dann, wenn π stationär ist; die stationäre Verteilung aus einem linearen System liegt daneben; die Ankunft misst d(t).
5. **Was daraus nicht folgt:** Detailed Balance ist hinreichend, nicht notwendig, und weder πP = π noch Detailed Balance sagen, dass die Kette ankommt.
"""

MODEL_NOTE = (
    "Das Modell ist **ausgedacht**: n Stellplätze im Ring, keine zwei benachbarten belegt. In den gelesenen Quellen gibt es keine reale Datenquelle für diese Aufgabe; "
    "gezählt werden Zustände, Schritte, Freiheitsgrade und Klassen auf kleinen, exakt lösbaren Räumen. Was die Zahlen für ein echtes Abtastproblem heißen, wird hier nicht behauptet."
)

CAPTION_BUILD = (
    "Der Kern entsteht aus Leitwerten (Fluss je Kante des Zustandsgraphen). Drei Regeln haben π als stationäre Verteilung; drei Fehlerfälle haben Zeilensumme 1 und laufen, "
    "verfehlen aber π. **Heat-bath** (Gibbs-Sampler, Glauber-Dynamik) ist eine Kette, **Gibbs-Maß** heißt in den Koordinationsspielen die Verteilung, nicht die Kette."
)
CAPTION_CYCLE5 = (
    "Die Irrfahrt auf dem Kreis mit fünf Zuständen (Levin/Peres/Wilmer, Beispiel 1.22): vorwärts mit Wahrscheinlichkeit p, rückwärts mit 1 − p. Im Ring mit 5 Plätzen ist das die Bahn "
    "der fünf Einzelnen unter der Drehung um einen Platz. Gemessen wird mit dem ganzzahligen SplitMix64; das Ergebnis ist ein Band, keine Zahl."
)
CAPTION_FREEDOM = (
    "Ein Fluss f(x, y) = π(x)·P(x, y) auf dem Zustandsgraphen ist stationär, wenn in jedem Zustand Zufluss gleich Abfluss gilt, und reversibel, wenn f(x, y) = f(y, x). "
    "Gezählt wird die Dimension des linearen Raums der Flüsse (die zulässigen Kerne bilden darin ein Polytop: Einträge ≥ 0, Zeilensumme ≤ 1)."
)
CAPTION_ARRIVAL = (
    "Detailed Balance ist eine Eigenschaft des **Kerns**, sie hängt weder von t noch vom Start ab. Was mit t fällt, ist der Abstand der **Verteilung nach t Schritten** zu π. "
    "Der Abstand d(t) ist der größte über alle Starts, die Mischzeit t_mix das erste t mit d(t) ≤ 1/4 (eine zweite Zähleinheit, die Mischzeit d̄ zwischen zwei Starts, "
    "zeigt `mischzeit-exakt-demo`)."
)

LIMITS_MARKDOWN = """
| Aussage der Demo | gilt hier | wo sie endet |
|---|---|---|
| Metropolis, Heat-bath und gleiche Leitwerte haben π stationär | Hard-Core-Ring, n = 4 bis 10, λ = 0,5 bis 5, Einzelplatz-Züge | andere Räume und größere Ringe sind nicht gerechnet |
| Falsch gebaute Kerne fallen durch die Prüfung πP = π | drei gezeigte Fehlerfälle (n = 6, λ = 2); bei λ = 1 sind sie wirkungslos | die Prüfung belegt π für den gebauten Kern, die Ursache eines Fehlers zeigt sie nicht |
| 2E − (V − 1) stationäre und E reversible Freiheitsgrade | Rang numerisch bestätigt für n = 3 bis 9; für n = 10 bis 12 gilt nur die Formel | Dimension des linearen Raums der Flüsse, kein Beweis in der Demo; jeder Zug erlaubt: nur für n ≤ 7 gerechnet |
| Zufällige Kreisströme machen die Kette nicht von selbst schneller | zufällige Kreisströme mit größter zulässiger Stärke, sechs Zeilen (n = 5 bis 7, λ = 2 und 5), gegen den reversiblen Kern mit halbem Metropolis-Leitwert; gegen reines Metropolis ist in diesen Zeilen kein Kreisstrom schneller | gehobene Ketten auf Pfad und Ring sind in der Literatur schneller; optimierte Kreisströme sind nicht gemessen |
| Der Gewinn der Drehung hängt nicht an der Drift | Mischung aus Metropolis und Drehung, n = 4 bis 8, λ = 2 und 5 | andere Konstruktionen nicht gemessen |
| Eine Kette kommt nicht immer an | 5-Kreis mit p = 1 (periodisch, kommt nie an) und reine Drehung des Rings (reduzibel, mehrere Klassen) | für die Standard-Kerne der Demo (Metropolis, Heat-bath, gleiche Leitwerte, Mischung) ist t_mix endlich; sie sind über den leeren Ring zusammenhängend |
| Zeitanteile einer Kette weichen von π ab | eine einzige Kette, die gewählte Länge, ein Zufalls-Seed | ein Lauf ist ein Band, keine Zahl |
| Schranken aus der Spektrallücke | nicht gezeigt | die Voraussetzung „reversibel“ ist hier nicht im Wortlaut belegt, für nichtreversible Kerne zeigt die Demo keine Schranke |
"""

MATH_LESEHILFE = """
**Lesehilfe.** Gegeben ist eine Verteilung π auf einer endlichen Zustandsmenge; gesucht sind Übergangsmatrizen P (Kerne), die π als stationäre Verteilung haben, und eine Prüfung dafür.
Der Block beschreibt das Modell der App vollständig (Zustandsraum, Verteilung, die gebauten Kerne, die Prüfungen und ihre Zähleinheiten); Beweise stehen nicht hier, Detailed Balance ⇒ stationär
und das Beispiel mit Drift sind Lehrbuch.

| Symbol | Bedeutung |
|---|---|
| n | Zahl der Plätze im Ring (n ≥ 3), Plätze 0 bis n − 1, Nachbarn modulo n |
| x, y | Zustände: Mengen belegter Plätze ohne zwei benachbarte; \\|x\\| ist die Zahl belegter Plätze |
| λ | Gewicht je belegtem Platz |
| π(x), Z | Zielverteilung λ^\\|x\\| / Z mit der Summe aller Gewichte Z |
| P(x, y) | Übergangswahrscheinlichkeit von x nach y (der Kern), Zeilensumme 1 |
| c(x, y) | Leitwert (Fluss) π(x)·P(x, y) |
| V, E | Zahl der Zustände und der Kanten des Zustandsgraphen (Kante: zwei Zustände, die sich in einem Platz unterscheiden) |
| P* | zeitumkehrte Kette |
| R, p, q | Drehung des Rings um einen Platz; Wahrscheinlichkeit der Drehung gegenüber der Gegendrehung; Anteil der Drehung an der Mischung |
| d(t), t_mix | Abstand nach t Schritten (größter über alle Starts) und die Mischzeit (erstes t mit d(t) ≤ 1/4), Zähleinheit Schritte |
"""

MATH_FORMULAS = r"""
**Zustandsraum und Ziel.** Zustände sind die Mengen belegter Plätze ohne zwei benachbarte; ihre Zahl ist 4, 7, 11, 18, 29, 47, 76, 123, 199, 322 für n = 3 bis 12 (Lucas-Zahlen).

$$\pi(x)=\frac{\lambda^{|x|}}{Z},\qquad Z=\sum_{y}\lambda^{|y|}=\mu_+^{\,n}+\mu_-^{\,n},\qquad \mu_\pm=\frac{1\pm\sqrt{1+4\lambda}}{2}$$

**Stationär und Detailed Balance.** Stationär heißt

$$\pi P=\pi ,$$

Detailed Balance heißt $\pi(x)P(x,y)=\pi(y)P(y,x)$ für alle $x,y$ und ist hinreichend, nicht notwendig, für Stationarität (Levin/Peres/Wilmer, Satz 1.20). Gemessen wird

$$\lVert \pi P-\pi\rVert_1=\sum_x\bigl|(\pi P)(x)-\pi(x)\bigr|,\qquad \max_{x,y}\bigl|\pi(x)P(x,y)-\pi(y)P(y,x)\bigr| .$$

**Kerne aus Leitwerten.** Mit $c(x,y)=c(y,x)\ge 0$ und $\sum_{y\ne x}c(x,y)\le\pi(x)$ ist

$$P(x,y)=\frac{c(x,y)}{\pi(x)}\ (y\ne x),\qquad P(x,x)=1-\sum_{y\ne x}P(x,y),$$

und Detailed Balance gilt. Die drei Regeln der Demo:

$$c_M(x,y)=\frac{\min\{\pi(x),\pi(y)\}}{n}\quad\text{(Metropolis: Platz gleichverteilt, Annahme }\min\{1,\pi(y)/\pi(x)\}\text{)},$$

$$c_G=\min_x\frac{\pi(x)}{\deg(x)}\quad\text{(gleiche Leitwerte auf jeder Kante)},$$

$$P_H(x,\,x\cup\{v\})=\frac{1}{n}\,\frac{\lambda}{1+\lambda}\quad\text{(Heat-bath: Platz }v\text{ gleichverteilt, beide Nachbarn frei, sonst wird }v\text{ geleert)}.$$

Die Fehlerfälle sind Kerne mit Zeilensumme 1, die π verfehlen: Annahme $1/2$ für jeden zulässigen Vorschlag, Annahme $\min\{1,\pi(x)/\pi(y)\}$ statt $\min\{1,\pi(y)/\pi(x)\}$, und der Heat-bath mit Belegungswahrscheinlichkeit $1/(1+\lambda)$.

**Freiheitsgrade.** Ein Fluss $f(x,y)$ auf den $2E$ gerichteten Kanten ist stationär, wenn Zufluss gleich Abfluss gilt; der Rang dieser $V$ Gleichungen ist $V-1$ (zusammenhängender Graph).

$$\dim_{\text{stat}}=2E-(V-1),\qquad \dim_{\text{rev}}=E,\qquad \dim_{\text{stat}}-\dim_{\text{rev}}=E-V+1 .$$

Der Unterschied $E-V+1$ ist die Zahl der unabhängigen Kreise des Graphen (Kreisströme). Erlaubt man jeden Zug, gibt es $(V-1)^2$ stationäre und $V(V-1)/2$ reversible Freiheitsgrade.

**Zeitumkehr, Drehung, Mischung.**

$$P^{*}(x,y)=\frac{\pi(y)\,P(y,x)}{\pi(x)},\qquad P_{\text{Mix}}=(1-q)\,P_M+q\,\bigl[p\,R+(1-p)\,R^{\top}\bigr].$$

Die Drehung $R$ erhält die Zahl der belegten Plätze und damit $\pi$. Für den Kreis mit fünf Zuständen ($P(i,i+1)=p$, $P(i,i-1)=1-p$, $\pi=1/5$) ist die Abweichung von Detailed Balance

$$\frac{|2p-1|}{5}.$$

**Ankunft.**

$$d(t)=\max_x\ \tfrac12\sum_y\bigl|P^{t}(x,y)-\pi(y)\bigr|,\qquad t_{\text{mix}}=\min\{t:\ d(t)\le \tfrac14\}.$$

**Quellen** (nur so zitiert, wie sie gelesen sind): Levin/Peres/Wilmer, *Markov Chains and Mixing Times*, 2. Aufl., AMS 2017: Detailed Balance (1.29) und Satz 1.20 (Seite 13), Beispiel 1.22 (Seite 14),
Metropolis-Matrix und größtmögliche Annahme (Seite 39 bis 41), Definition von d(t) und t_mix (Kapitel 4). Vucelja, arXiv 1412.8762 (Abstract, Abschnitt zum Ring): gehobene nichtreversible Ketten brauchen auf dem Ring
mit Gleichverteilung O(N) statt ∝ N² Schritte; die Arbeit von Diaconis/Holmes/Neal ist nur über deren Abstract bekannt. Suwa/Todo, arXiv 1007.2262 (Abstract): globales Gleichgewicht ohne Detailed Balance, im Potts-Modell
eine mehr als 6-mal kürzere Autokorrelationszeit (Messung dieser Arbeit, kein allgemeines Verhältnis). Andrieu/Livingstone, arXiv 1906.06197 (Abstract): Vergleichssätze für nichtreversible Ketten
werden für eine bestimmte Klasse bewiesen. Die Aussage, 5000 Burn-in-Schritte stellten „Detailed Balance“ sicher, ist falsch (Burn-in ändert die Verteilung nach t Schritten, nicht den Kern); eine Quelle dafür wird hier nicht geführt.
OEIS A000032 für die Lucas-Zahlen.
"""

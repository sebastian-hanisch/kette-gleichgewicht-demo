# Eine Kette mit gewünschtem Gleichgewicht: bauen, prüfen, und was Detailed Balance nicht sagt – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-kette-gleichgewicht-demo.streamlit.app/)** -->

Eine von mehreren Demos zu **MCMC: Verteilungen abtasten** (Bereich Simulation) für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie dreht die Richtung der Nachbar-Demo
[`exakt-oder-abtasten-demo`](https://github.com/sebastian-hanisch/exakt-oder-abtasten-demo) um: dort ist die Kette gegeben und ihr Gleichgewicht wird gerechnet, hier ist **das Gleichgewicht die Vorgabe und die Kette zu bauen**,
so dass man π nur über Leitwerte und eine Prüfung braucht. Gezeigt werden Ketten zu einem vorgegebenen π (Metropolis, Heat-bath, gleiche Leitwerte), drei Fehlerfälle, die Zahl der Ketten, die zu π passen,
nichtreversible Ketten und der Unterschied zwischen **stationär**, **Detailed Balance** und **angekommen**.
Nachbarn: [`metropolis-kleiner-raum-demo`](https://github.com/sebastian-hanisch/metropolis-kleiner-raum-demo) (das Rezept Metropolis, die Ursachen falsch gebauter Ketten),
[`gibbs-glauber-demo`](https://github.com/sebastian-hanisch/gibbs-glauber-demo) (Heat-bath gegen Metropolis, der Takt, die Reihenfolge),
[`mischzeit-exakt-demo`](https://github.com/sebastian-hanisch/mischzeit-exakt-demo) (die Mischzeit systematisch, beide Zähleinheiten),
[`kopplung-ketten-demo`](https://github.com/sebastian-hanisch/kopplung-ketten-demo), [`kopplung-vergangenheit-demo`](https://github.com/sebastian-hanisch/kopplung-vergangenheit-demo),
[`diagnose-taeuscht-demo`](https://github.com/sebastian-hanisch/diagnose-taeuscht-demo) (ESS und R̂), [`gipfel-tempering-demo`](https://github.com/sebastian-hanisch/gipfel-tempering-demo),
[`dimension-hmc-demo`](https://github.com/sebastian-hanisch/dimension-hmc-demo), [`sampler-verfahrenswahl-demo`](https://github.com/sebastian-hanisch/sampler-verfahrenswahl-demo).

**Die Räume sind ausgedacht**, ein **Modell, keine reale Datenquelle**: n Stellplätze im Ring, kein Paar benachbarter Plätze belegt, Gewicht λ^k bei k belegten Plätzen. In den gelesenen Quellen gibt es keine reale Datenquelle für diese Aufgabe; gezählt
werden Zustände, Schritte, Freiheitsgrade und Klassen auf kleinen, exakt lösbaren Räumen. Was die Zahlen für ein echtes Abtastproblem heißen, wird hier nicht behauptet. Nichts davon ist inhaltlich neu (Levin/Peres/Wilmer, Kapitel 1 und 3);
neu ist nur die Vorführbarkeit mit Zähleinheit und exakten Orakeln.

## Warum dieses Problem

Wer eine Kette zu einer vorgegebenen Verteilung π bauen will, fragt sich: **Gibt es nur eine? Wie prüfe ich sie? Muss Detailed Balance gelten? Und kommt die Kette an?**
Antwort in diesem Modell, Zähleinheit Schritte bzw. Freiheitsgrade: Im Ring mit n = 4 Plätzen und λ = 2 (7 Zustände) haben die Flüsse, die π stationär lassen, 10 Freiheitsgrade (Dimension des Flussraums, keine Zahl einzelner Ketten); 8 davon entsprechen Detailed Balance, die übrigen 2 sind Kreisströme.
Drei Kerne mit demselben π (Metropolis, Heat-bath, gleiche Leitwerte) brauchen 15, 22 und 18 Schritte bis d(t) ≤ 1/4; bei n = 10 und λ = 5 sind es 173, 208 und 13 655 Schritte. Detailed Balance ist hinreichend, nicht notwendig: im Kreis mit fünf Zuständen
und Drift p = 0,7 ist die Gleichverteilung stationär, die Abweichung von Detailed Balance ist 0,0800, der Nettofluss je Kante exakt 0,0800 und gemessen 0,0799. Und weder πP = π noch Detailed Balance sagen, dass die Kette ankommt: die reine Drehung des
5-Kreises hat π stationär und eindeutig, aber d(t) = 0,8 für alle t. Die Falle ist, drei Dinge zu verwechseln, die Kern, Verteilung und Zeit betreffen (stationär, Detailed Balance, Ankunft). Negativbefund: nichtreversibel ist hier nicht von selbst schneller
(zufällige Kreisströme ändern t_mix gegenüber dem reversiblen Kern mit halbem Metropolis-Leitwert mal um −2, mal um +7 Schritte), und der Gewinn der Drehung hängt nicht an der Drift. Alles gilt in diesem Modell (Hard-Core-Ring n ≤ 10, λ = 0,5 bis 5, Einzelplatz-Züge und Drehung; 5-Kreis), bei diesen Parametern.

## Modell

* **Ring und Verteilung:** Zustand x = Menge der belegten Plätze 0 bis n − 1 (Nachbarn modulo n, n ≥ 3), kein Paar benachbarter Plätze belegt; π(x) = λ^|x| / Z mit der Summe aller Gewichte Z = μ₊ⁿ + μ₋ⁿ, μ± = (1 ± √(1 + 4λ)) / 2. Die Zahl der Zustände
  ist 4, 7, 11, 18, 29, 47, 76, 123, 199, 322 für n = 3 bis 12 (Lucas-Zahlen). Das Beispiel von Hand: n = 4, λ = 2: leer, vier Einzelne, zwei gegenüberliegende Paare, Gewichte 1, 2, 2, 2, 2, 4, 4, Z = 17; zweiter Raum n = 5, λ = 2: 11 Zustände, Z = 31.
* **Begriffe:** *Kern* P (Übergangsmatrix); *stationär* (in der App „stationäre Verteilung“, weil „Gleichgewicht“ im Portfolio auch Nash-Gleichgewicht heißt): πP = π; *Detailed Balance*: π(x)P(x, y) = π(y)P(y, x); *Leitwert* c(x, y) = π(x)P(x, y);
  *Kreisstrom*: ein Fluss, der auf einem Kreis des Zustandsgraphen in einer Richtung zusätzlich läuft und jede Zustandsbilanz erhält; *zeitumkehrte Kette* P*(x, y) = π(y)P(y, x)/π(x). **Heat-bath** (Gibbs-Sampler, Glauber-Dynamik) ist eine Kette,
  **Gibbs-Maß** (Koordinationsspiele) die Verteilung.
* **Ankunft:** d(t) = max über alle Starts x von TV(P^t(x, ·), π) und t_mix = erstes t mit d(t) ≤ 1/4 (Levin/Peres/Wilmer, Schwelle 1/4); Zähleinheit Schritte. Die zweite Zähleinheit der Mischzeit, d̄ zwischen zwei Starts mit Schwelle 1/e, wird hier nur genannt;
  Definition und Zahlen stehen in `mischzeit-exakt-demo`.
* **Kerne aus Leitwerten:** jede symmetrische Wahl c(x, y) = c(y, x) ≥ 0 mit Zeilensumme ≤ π(x) gibt P(x, y) = c(x, y)/π(x), P(x, x) = Rest, mit Detailed Balance und damit π stationär (Levin/Peres/Wilmer, Satz 1.20). Drei Punkte dieser Menge auf dem Zustandsgraphen
  der Einzelplatz-Züge: **Metropolis** c = min(π(x), π(y))/n (der größte zulässige Leitwert bei Platzwahl 1/n), **Heat-bath** (Platz gleichverteilt, belegen mit λ/(1 + λ), wenn beide Nachbarn frei) und **gleiche Leitwerte** c = min über x von π(x)/Grad(x).
  Dazu **eigene Leitwerte** (Metropolis mal Faktor f₀ auf den Kanten ab dem leeren Ring und f₁ auf allen höheren Stufen).
* **Drei Fehlerfälle** (Zeilensumme 1, Einträge ≥ 0, aber π verfehlt): jeder zulässige Vorschlag mit Wahrscheinlichkeit ½ angenommen; Metropolis mit umgekehrtem Verhältnis min(1, π(x)/π(y)); Heat-bath mit Belegungswahrscheinlichkeit 1/(1 + λ) statt λ/(1 + λ).
* **Nichtreversible Kerne zum selben π:** die **Drehung** des Rings um einen Platz erhält die Zahl der belegten Plätze und damit π; die Mischung (1 − q) · Metropolis + q · [p · Drehung + (1 − p) · Gegendrehung]; **Kreisströme** auf dem Zustandsgraphen
  (Fluss f · c_M ± Kreisstrom auf jeder Kante, auf die größte Stärke skaliert, bei der alle Flüsse ≥ 0 bleiben). Der **5-Kreis mit Drift** (Levin/Peres/Wilmer, Beispiel 1.22) steht im Ring mit 5 Plätzen als Bahn der fünf Einzelnen unter der Drehung.
* Der 📐-Expander der App enthält die Formeln als vollständiges Modell (ohne Beweise).

## Methodik

* **Kerne und Prüfungen** (`kgw_kernels.py`, `kgw_checks.py`): Metropolis und gleiche Leitwerte aus Leitwerten, Heat-bath aus seiner Regel; Zeilensummen, ‖πP − π‖₁, Detailed-Balance-Abweichung max |π(x)P(x, y) − π(y)P(y, x)|, die stationäre Verteilung aus einem linearen
  System (eine Gleichung durch die Normierung ersetzt), Klassen über starke Zusammenhangskomponenten. **t_mix** exakt aus Matrixpotenzen (Quadrieren und binäres Heben, Zeilen nachnormiert; Schwelle inklusive), keine Schritt-für-Schritt-Schleife bis 13 655.
  Rang der Erhaltungsgleichungen numerisch für n ≤ 9, darüber die Formel 2E − (V − 1) (in der Tabelle gekennzeichnet); jeder Zug erlaubt: numerisch bis n = 7.
* **Zufall** nur über den ganzzahligen SplitMix64 (`kgw_rng.py`, geprüfte Vorlage, unverändert kopiert; Normalzahlen ohne Logarithmus und Kosinus der C-Bibliothek): Zufallsergebnisse sind auf jeder Plattform gleich. `kgw_simulation.py` rechnet viele Ausgaben auf einmal
  (der Generator ist zählerbasiert; `tests/test_simulation.py` vergleicht mit dem schrittweisen Generator). Zufalls-Seed in der Seitenleiste. Zufallsergebnisse sind Bänder, keine Zahlen.
* **Simulationen:** Nettofluss des 5-Kreises (eine einzige Kette, 10 000 bis 10⁶ Schritte, Band ±4 Standardabweichungen nach √(0,2/T)); Zeitanteile der gebauten Kette (Start aus π, 10 000 bis 200 000 Schritte) neben der Rauschgrenze unabhängiger Ziehungen
  (größter Abstand aus 4 Stichproben gleicher Größe); 20 zufällige Kreisströme je Auswertung (ab n = 9 nur 10; Zufalls-Seeds Seed · 1000 + j).
* **Rechenzeit** (Messung dieser README, nicht im Test; eine CPU, Windows, Python 3.14, gemessen unter Last mehrerer gleichzeitiger Prozesse): im größten Fall (n = 10, λ = 5, 123 Zustände) Kern bauen und prüfen samt den drei Vergleichszeiten etwa 2,2 s, die Verteilung nach t Schritten (400 Schritte) etwa 1,0 s, Zeitanteile über 200 000 Schritte etwa 0,9 s, 10 Kreisströme (ab n = 9) etwa 1,4 s; bei n = 8 sind 20 Kreisströme in etwa 0,2 s gerechnet, die sechs Zeilen der Kreisströme in etwa 0,5 s, der 5-Kreis mit 10⁶ Schritten in etwa 0,1 s. Der erste Aufruf nach dem Start dauert länger, jedes Ergebnis liegt danach im Zwischenspeicher. Nichts ist vorgerechnet oder eingefroren; die Streamlit Cloud ist langsamer und nicht gemessen.

## Befunde (gemessen, keine Behauptungen)

Zähleinheit: Schritte (bis d(t) ≤ 1/4), Freiheitsgrade, Klassen. Die Spalte „Test“ nennt den Test in `tests/test_claims.py` (soweit nicht anders vermerkt), der genau diese Zahl prüft: Ganzzahlen und Brüche exakt, Mischzeiten als Ganzzahlen,
Zufallsergebnisse mit festem Zufalls-Seed als Band (die Zahl der README bis auf die genannte Stelle).

| Frage | Befund | Test |
|---|---|---|
| Wie viele Zustände, welche Normierung? | n = 4, λ = 2: 7 Zustände, Gewichte 1, 2, 2, 2, 2, 4, 4, Z = 17; n = 5, λ = 2: 11 Zustände, Z = 31; Zustandszahl 4, 7, 11, 18, 29, 47, 76, 123, 199, 322 für n = 3 bis 12 und Z = μ₊ⁿ + μ₋ⁿ für vier Werte λ | `test_c1` |
| Gleiches π, andere Zeit (exakt, alle Starts) | t_mix Metropolis / Heat-bath / gleiche Leitwerte: n = 4, λ = 2: **15 / 22 / 18**; n = 6, λ = 2: 24 / 36 / 50; n = 6, λ = 5: 87 / 104 / 502; n = 8, λ = 5: 127 / 153 / 2675; n = 10, λ = 5: 173 / 208 / **13 655** | `test_c2` |
| Hand: n = 4, λ = 2 | gleicher Leitwert c = 1/68 (leer ↔ Einzelner und Einzelner ↔ Paar), Metropolis 1/68 und 2/68; d(1) = 10/17 (Metropolis), 67/102 (Heat-bath), 95/136 (gleiche Leitwerte); von ∅ aus TV(μ_t, π) = 16/17, 9/17, 15/68 für t = 0, 1, 2 | `test_c3` (und `tests/test_hand_beispiele.py`) |
| Warum gleiche Leitwerte langsam sind | der gemeinsame Leitwert wird vom kleinsten Gewicht bestimmt: er ist das 0,302fache des mittleren Metropolis-Leitwerts bei n = 8, λ = 2 und das 0,015fache bei n = 10, λ = 5 (Mittel aller Kanten) | `test_c4` |
| Falsch gebaute Kerne (n = 6, λ = 2) | TV der stationären Verteilung zu π / ‖πP − π‖₁: umgekehrtes Verhältnis **0,4154 / 0,2769**; Heat-bath mit 1/(1 + λ) 0,4154 / 0,1846; feste Annahme ½ **0,1889 / 0,0923**; Zeilensummen 1 in allen; n = 4, λ = 2: feste Annahme ½: TV 22/119, umgekehrtes Verhältnis 39/119; bei λ = 1 sind alle drei Fehlerfälle wirkungslos (‖πP − π‖₁ < 10⁻⁹) | `test_c5` |
| Wie viele Freiheitsgrade haben die Ketten, die zu π passen (Dimension des Flussraums, Graph der Einzelplatz-Züge) | stationär / reversibel / Kreise: n = 3: 3 / 3 / 0 (ein Baum); n = 4: **10 / 8 / 2**; n = 5: 20 / 15 / 5; n = 6: 43 / 30 / 13; n = 7: 84 / 56 / 28 (Rang gerechnet bis n = 9: n = 8: 162 / 104 / 58; n = 9: 303 / 189 / 114); n = 10 bis 12 nach der Formel: 558, 1012 und 1815 stationäre, 218, 407 und 747 Kreise | `test_c6` |
| Ohne Graphbeschränkung (jeder Zug erlaubt) | 7 Zustände: 36 stationäre gegen 21 reversible; 11 Zustände: 100 gegen 55; 18 Zustände: 289 gegen 153 (= (V − 1)² und V(V − 1)/2, Rang numerisch bis n = 7) | `test_c6` |
| 5-Kreis mit Drift: Abweichung von Detailed Balance | π · \|2p − 1\| = 0,0400 / 0,0800 / 0,1600 / 0,2000 für p = 0,6 / 0,7 / 0,9 / 1; Kolmogorov-Produkt bei p = 0,7: vorwärts 0,16807 gegen rückwärts 0,00243 (16807/100000 und 243/100000); Zeitumkehr dreht die Drift | `test_c7` |
| 5-Kreis: Nettofluss je Kante und Schritt | exakt **0,0800** bei p = 0,7; gemessen (eine Kette, 10⁶ Schritte) 0,0799 bei Zufalls-Seed 11001 und 0,0802 bei Zufalls-Seed 11002 (als grobe obere Schranke der Standardabweichung dient etwa 0,0004, die tatsächliche Streuung ist kleiner; beide Werte liegen innerhalb ±4 dieser Schranken auf allen fünf Kanten) | `test_c8` |
| 5-Kreis: Ankunft | t_mix = 5 / 6 / 15 / nie für p = 0,5 / 0,7 / 0,9 / 1; \|λ₂\| = 0,8090 / 0,8425 / 0,9357 / 1; bei p = 0,7 ein komplexes Eigenwertpaar (die Verteilung nach t Schritten schwingt, d(t) selbst fällt); p = 1: d(t) = 0,8 für t = 0 bis 200; träge Drehung (½ bleiben, ½ drehen): t_mix = 5 | `test_c9` |
| Drehung als Zug (Mischung, 30 Zellen n = 4 bis 8, λ = 2 und 5, q = 0,05; 0,2; 0,5) | t_mix gegen reines Metropolis: Gewinn **1,41- bis 11,2fach** in allen 90 Zeilen (n = 6, λ = 5: 87 → 27, 32, 51 bei q = 0,05; 0,2; 0,5); der Unterschied von t_mix zwischen p = 0,5 (reversibel), 0,7 und 1 beträgt bei geradem n höchstens 1 Schritt (17 von 18 Zellen gleich), bei ungeradem n höchstens 2 Schritte (5 von 12 gleich); Abweichung von Detailed Balance q · \|2p − 1\| · max π über die bewegten Zustände | `test_c10` |
| Kreisströme (zufällig, größte zulässige Stärke, Basisfaktor ½, je 20 Ströme, Zufalls-Seed 11001) | Änderung von t_mix gegen den reversiblen Kern: n = 5, λ = 2 (t_mix 25): 0 bis +3; n = 5, λ = 5 (61): +1 bis +7; n = 6, λ = 2 (48): −2 bis 0; n = 6, λ = 5 (174): −1 bis +1; n = 7, λ = 2 (48): 0 bis +1; n = 7, λ = 5 (137): +1 bis +2; insgesamt **−2 bis +7 Schritte**; mit den Zufalls-Seeds 11001 bis 11005 insgesamt −6 bis +8; bei n = 6, λ = 2 sind 15 von 20 Strömen kürzer, bei n = 5, λ = 5 keiner; \|λ₂\| ist in 120 von 120 Fällen höchstens so groß wie beim reversiblen Kern (auf vier Stellen) | `test_c11` |
| Ein Kreisstrom von Hand (n = 4, λ = 2) | halber Metropolis-Leitwert, dazu s = 1/136 auf ∅ → {0} → {0,2} → {2} → ∅: aus ∅ gehen 1/4 nach {0}, 0 nach {2}, 1/2 bleiben; Abweichung von Detailed Balance 1/68; t_mix 29 mit und ohne Kreisstrom | `test_c12` |
| Kern gegen Ankunft (n = 6, λ = 2, Start ∅) | TV(μ_t, π) bei t = 0 / 1 / 2 / 5 / 10 / 20 / 40 / 60: Metropolis (Abweichung des Kerns 0) 0,9846 / 0,8154 / 0,3000 / 0,1163 / 0,0513 / 0,0125 / 0,0008 / 0,0000; Mischung mit Drehung und Drift (q = 0,2, p = 1, Abweichung des Kerns **0,0123** bei jedem t) 0,9846 / 0,8000 / 0,4800 / 0,1570 / 0,0713 / 0,0226 / 0,0025 / 0,0003; von Hand n = 4: 16/17, 9/17, 15/68, 69/544, 297/4352, 1305/34816 | `test_c13` |
| Reine Drehung des Rings | Klassen = Bahnen der Drehung: 2, 3, 3, 5, 5, 8, 10, 15 für n = 3 bis 10; n = 5, λ = 2: 3 Klassen (1, 5 und 5 Zustände), d(t) = 30/31 = 0,9677 für alle t, eine zweite stationäre Verteilung (⅓ auf ∅, je 1/15 auf den zehn anderen Zuständen) | `test_c14` |
| Zeitanteile einer Kette (n = 6, λ = 2, 200 000 Schritte, Zufalls-Seeds 11081 bis 11084, Start aus π) | TV zu π: Metropolis 0,0142 / 0,0112 / 0,0165 / 0,0137, Mischung (q = 0,2, p = 0,7) 0,0080 / 0,0049 / 0,0088 / 0,0054; Rauschgrenze unabhängiger Ziehungen 0,0037: in diesen vier Läufen Metropolis das 3,0- bis 4,4fache, Mischung das 1,3- bis 2,4fache (das Verhältnis streut zwischen Läufen stark; ein Befund aus vier Läufen, kein Band) | `test_c15` |
| Burn-in und der Kern | der Kern ändert sich mit t nicht (Abweichung 0 bzw. 0,0123 bei jedem t), TV(μ_t, π) fällt in beiden Fällen streng; nach 60 Schritten liegt TV(μ_t, π) der Mischung bei 0,0003, die Abweichung ihres Kerns bei 0,0123 (mehr als das 40fache); die Aussage „5000 Burn-in-Schritte stellen Detailed Balance sicher“ ist falsch | `test_c13`, `test_c16` |

Messungen dieser README, die **nicht** im Test stehen: die Rechenzeiten (Abschnitt Methodik).

**Negativbefund (bleibt stehen):** Nichtreversibel ist hier nicht von selbst schneller: gegenüber dem reversiblen Kern mit halbem Metropolis-Leitwert (der Basis der Kreisströme, selbst langsamer als reines Metropolis) machen die zufälligen Kreisströme t_mix mal kürzer, mal länger (−2 bis +7 Schritte bei Zufalls-Seed 11001, −6 bis +8 über fünf Zufalls-Seeds); gegenüber reinem Metropolis ist in diesen sechs Zeilen und fünf Zufalls-Seeds kein Kreisstrom schneller. Der Gewinn der Drehung (bis 11,2fach)
kommt von der Drehung als Zug, nicht von der Drift (Kontrolle p = 0,5). Gehobene Ketten auf dem Pfad und dem Ring sind in der Literatur schneller (Vucelja); optimierte Kreisströme sind hier nicht gemessen.

## Befunde und Korrekturen gegenüber der ersten Rechnung

* **Zufällige Kreisströme:** die erste Rechnung (numpy-Zufall) nannte −3 bis +6 Schritte; mit dem SplitMix64 der Demo sind es −2 bis +7 (Zufalls-Seed 11001) und −6 bis +8 über fünf Zufalls-Seeds. Die Aussage bleibt („nicht von selbst schneller“), steht aber jetzt mit den Zählungen
  (kürzer, länger, gleich) und ohne feste Grenze.
* **Gleiche Leitwerte, „mittlerer Quotient“:** die erste Rechnung meinte den gemeinsamen Leitwert geteilt durch das **Mittel** der Metropolis-Leitwerte (0,302 und 0,015); der Mittelwert der Quotienten je Kante ist 0,394 und 0,091. Die README nennt die erste Größe mit ihrer Definition.
* **Zeitanteile:** die Zustandsnummerierung der Demo (erst nach Zahl der belegten Plätze) gibt andere Zufallsfolgen als die erste Rechnung; Rauschgrenze 0,0037 statt 0,0025 bis 0,0034, Zeitanteile von Metropolis das 3,0- bis 4,4fache (erste Rechnung das 3- bis 5fache). Der Nettofluss des 5-Kreises
  (0,0799 und 0,0802) reproduziert sich dagegen bitgleich.
* **Fehlerfälle bei λ = 1:** alle drei Fehlerfälle sind wirkungslos (π ist dann die Gleichverteilung); die App meldet das statt eines Effekts.
* **Auswahl der Kerne:** Regel und Fehlerfall stehen in einer einzigen Auswahl; für „eigene Leitwerte“ gibt es zwei Faktoren (f₀ für die Kanten vom leeren Ring, f₁ für alle höheren Stufen) statt eines Faktors je Stufe.
* **Experiment 3:** die Kernanalyse mit Kreisstrom läuft für n = 3 bis 10 (n = 3 meldet „Baum“); die Tabelle der Freiheitsgrade deckt n = 3 bis 12 ab, der Rang ist bis n = 9 gerechnet. Ab n = 9 werden 10 statt 20 Kreisströme gerechnet (Rechenzeit).

## Quellen

Nur so zitiert, wie sie gelesen sind:

* Levin, Peres, Wilmer, *Markov Chains and Mixing Times*, 2. Auflage, AMS 2017 (ganzes Buch gelesen): Detailed Balance (1.29) und Satz 1.20 (Seite 13), Beispiel 1.22 (Seite 14), Metropolis-Matrix (3.5, Seite 41) und die größtmögliche Annahme π(x) ∧ π(y) (Seite 39),
  Definition von d(t) und t_mix (Kapitel 4). Die Nummer des Konvergenzsatzes (irreduzibel, aperiodisch) ist hier nicht nachgeschlagen.
* Vucelja, „Lifting – A nonreversible Markov chain Monte Carlo algorithm“, American Journal of Physics 84 (2016), arXiv 1412.8762 (Abstract und Abschnitt zum Ring gelesen): gehobene Ketten brauchen auf dem Ring mit Gleichverteilung O(N) statt ∝ N² Schritte. Diaconis/Holmes/Neal (2000) ist nur über den Abstract bekannt.
* Suwa, Todo, arXiv 1007.2262 (Abstract): globales Gleichgewicht ohne Detailed Balance, im Potts-Modell eine mehr als 6-mal kürzere Autokorrelationszeit (Messung dieser Arbeit, kein allgemeines Verhältnis).
* Andrieu, Livingstone, arXiv 1906.06197 (Abstract): Vergleichssätze für nichtreversible Ketten werden für eine bestimmte Klasse bewiesen.
* OEIS A000032 (Lucas-Zahlen).
* **Nicht gelesen:** Peskun (1973) und Tierney (1998) zur Ordnung nach Leitwerten, Norris, Robert/Casella, das Handbook als Buch; sie werden hier nicht als Beleg geführt.

## Abgrenzung zu Nachbar-Demos

* **Das Rezept Metropolis, Vorschlagsfallen, die Annahmerate und alle Ursachen falsch gebauter Ketten** samt dem Fall „Irrfahrt ohne Ablehnen“: [`metropolis-kleiner-raum-demo`](https://github.com/sebastian-hanisch/metropolis-kleiner-raum-demo). Hier nur die Menge der Kerne und die Prüfung als Zahl.
* **Heat-bath gegen Metropolis, der Takt (Heat-bath als Metropolis mit gleichmäßig kleinerem Leitwert), Reihenfolge der Plätze:** [`gibbs-glauber-demo`](https://github.com/sebastian-hanisch/gibbs-glauber-demo). **Mischzeit systematisch** (Spektrallücke, Schranken, d̄ mit dem Satz d ≤ d̄ ≤ 2d): [`mischzeit-exakt-demo`](https://github.com/sebastian-hanisch/mischzeit-exakt-demo).
* **Gate-Kette und Gamma–Poisson** (die Richtung „Kette gegeben“): [`exakt-oder-abtasten-demo`](https://github.com/sebastian-hanisch/exakt-oder-abtasten-demo) und [`markov-queue-demo`](https://github.com/sebastian-hanisch/markov-queue-demo).
* **Metropolis, pCN, ESS und R̂ im stetigen Raum:** Inverse Probleme, [`posterior-sampling-demo`](https://github.com/sebastian-hanisch/posterior-sampling-demo); Zeitanteile einer Kette sind keine unabhängigen Ziehungen, ESS und die Diagnostik zeigt [`diagnose-taeuscht-demo`](https://github.com/sebastian-hanisch/diagnose-taeuscht-demo).
* **Gibbs-Maß als stationäre Verteilung der Geburts-Sterbe-Kette** (immer reversibel): [`koordinationsspiel-demo`](https://github.com/sebastian-hanisch/koordinationsspiel-demo) und [`gleichgewichtsauswahl-demo`](https://github.com/sebastian-hanisch/gleichgewichtsauswahl-demo).
* **Metropolis-Annahme als Optimierer** ([`simulated-annealing-demo`](https://github.com/sebastian-hanisch/simulated-annealing-demo), [`parallel-tempering-demo`](https://github.com/sebastian-hanisch/parallel-tempering-demo)): dort ist die Annahmeregel die Metropolis-Regel; hier ist die Menge der Kerne größer als Metropolis.
* **Irrfahrt auf Graphen als Eigenschaft eines Netzes** ([`meinungsbildung-demo`](https://github.com/sebastian-hanisch/meinungsbildung-demo)), **Ketten, die ein vorgegebenes π nur richtig gebaut erreichen** ([`konfigurationsmodell-demo`](https://github.com/sebastian-hanisch/konfigurationsmodell-demo), [`zulaessige-reihenfolgen-demo`](https://github.com/sebastian-hanisch/zulaessige-reihenfolgen-demo)): Verweise, kein Nachbau.

## Ehrliche Grenzen

* **Das Modell ist ausgedacht.** Der Hard-Core-Ring und der 5-Kreis sind kleine, exakt lösbare Räume; was die Zahlen für ein echtes Abtastproblem heißen, wird hier nicht behauptet, und es gibt keine Messung in den gelesenen Quellen, dass MCMC im Betrieb bessere Entscheidungen liefert.
* **Die Menge der Kerne zu π** ist eine Dimensionszahl des linearen Raums der Flüsse (Standardargument 2E − (V − 1), numerisch bestätigt für n = 3 bis 9); die zulässigen Kerne bilden darin ein Polytop (Einträge ≥ 0, Zeilensumme ≤ 1), das den Raum nur ausfüllt, wenn es innere Punkte gibt (die Kerne mit f < 1 sind welche).
  Gezeigt am Beispiel, nicht als Satz.
* **Nichtreversibel schneller** ist hier nur für zufällige Kreisströme (sechs Zeilen, Zufalls-Seeds 11001 bis 11005) und die Drehung gemessen; optimierte Kreisströme, andere Graphen (Pfad, Gitter) und andere nichtreversible Konstruktionen sind nicht gemessen.
* **Schranken aus der Spektrallücke** (Mischzeit-Sandwich) zeigt die Demo für nichtreversible Kerne nicht; die Voraussetzung „reversibel“ ist in den gelesenen Quellen nicht im Wortlaut belegt. Die Ordnung nach Leitwerten (Peskun, Tierney) ist nicht gelesen; „mehr Leitwert, schneller“ steht nirgends als Satz.
* **Zufallsergebnisse** (Nettofluss, Zeitanteile, Kreisströme) sind Bänder mit festen Zufalls-Seeds; je Zelle stehen ein bis vier Zufalls-Seeds in der README (Kreisströme fünf), die Bandbreite über mehr Zufalls-Seeds ist nicht gemessen.
* **Eine einzige Kette** liefert die Zeitanteile; sie sind keine unabhängigen Ziehungen (Faktor 3,0 bis 4,4 über der Rauschgrenze bei Metropolis in vier Läufen; das Verhältnis streut zwischen Läufen stark), die Autokorrelation wird hier nicht ausgewertet.
* **Browser-Verhalten** (Kleben des Kopfes, Darstellung der Plotly-Diagramme, die ausgegrauten Regler) sieht der AppTest nicht; angesehen wurden im Browser nur der Kopf, die Preset-Leiste, das Ergebnis mit der Matrix, Experiment 2 (Kreis) und Experiment 4 (Verlauf); Experiment 3 und das helle gegen das dunkle Design nicht.
* **Plattform:** Zahlen unter Windows, Python 3.14; Ganzzahlen und Brüche sind plattformfest, Mischzeiten als Ganzzahlen aus Gleitkomma-Potenzen (der Randfall d(t) = 1/4 genau ist im Test an einer Zwei-Zustands-Kette geprüft); Zufallsergebnisse mit dem exakten SplitMix64 ebenfalls (Linux-Prüfung siehe Verifikation).

## Verifikation

Die Suite hat **224 Tests** (`pytest --collect-only -q tests`); der Test `test_readme_nennt_die_testzahl_der_suite` hält die Zahl aktuell.

- **Handbeispiele** (Brüche, Mengen, Ganzzahlen; Beispiel von Hand n = 4, λ = 2 und n = 5): Zustände, Gewichte, Leitwerte, Matrizen, t_mix und d(1), Verteilung nach t Schritten, 5-Kreis, Freiheitsgrade, Kreisstrom, Fehlerfälle, reine Drehung (`tests/test_hand_beispiele.py`).
- **Orakel** (`tests/test_oracle_kette_gleichgewicht.py`): Kerne gegen den Aufbau in Brüchen und gegen Metropolis und Heat-bath aus den Regeln (n = 4 bis 6); stationäre Verteilung auf drei Wegen (Formel, lineares System, linker Eigenvektor); Abweichung und Nettofluss gegen π(2p − 1) und
  q|2p − 1|·max π; t_mix gegen Brüche (n = 4, 5), gegen `numpy.linalg.matrix_power` (die kleinste Schwelle) und gegen Vektoriteration; Rang gegen die Formel (n = 3 bis 9); Klassen gegen die Bahnenzahl der Drehung; Zufallsergebnisse als Band gegen die exakte Größe.
- **Eigenschaften** (`tests/test_eigenschaften_gleichgewicht.py`): zulässige Leitwerte geben Kerne mit π; gemeinsames Skalieren der Gewichte, Umnummerieren, Drehen des Rings; Zeitumkehr; Kreisströme bis zur größten Stärke; d(t) fällt nicht; Dimension der Flüsse wächst mit der Zahl der Kanten.
- **Literatur** (`tests/test_literatur_gleichgewicht.py`): Beispiel 1.22, Satz 1.20, Lucas-Zahlen, Definition von d(t) an einer Zwei-Zustands-Kette.
- **Verdikte** an handgesetzten Werten (`tests/test_verdicts.py`), **Randfälle** (n = 3, n = 10, λ = 0,01 bis 1000, Leitwertfaktor 0, unzulässige Leitwerte, ein Schritt, Burn-in 0; `tests/test_edge_cases.py`), **Kerne und Prüfer** (`tests/test_kernels.py`),
  **Zufall** (`tests/test_simulation.py`), **Claims** (jede Zahl dieser README; `tests/test_claims.py`), **Texte** (`tests/test_texte.py`).
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an beiden Rändern, jede Auswahl), je Preset die erwartete Verdikt-Zeile, Permalink (`tests/test_app.py`); **Layout:** Schnellstart oben in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben, jede Reglerwirkung
  hebt die Markierung auf, ausgegraute Regler (`tests/test_layout.py`). **Presets, Footer, Laufzeit-Importe:** `tests/test_presets.py`, `tests/test_footer.py`, `tests/test_runtime_imports.py`.
- **Linux-Prüfung:** `demo_linux_check.py` (Docker, Linux, Python 3.12, `requirements-dev.txt`): alle 224 Tests bestanden (2026-11-18, vor dem ersten Push).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Ergebnis, Kernfragen, vier Experimente, Grenzen, Formeln |
| `kgw_checks.py` | Prüfungen: Zeilensummen, πP = π, Detailed Balance, stationäre Verteilung, Klassen, Zeitumkehr, d(t), t_mix, Freiheitsgrade |
| `kgw_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `kgw_cycle5.py` | Der 5-Kreis mit Drift: Kern, Abweichung, Nettofluss exakt und gemessen |
| `kgw_evaluation.py` | Auswertungen der vier Experimente und Verdikte (reine Funktionen) |
| `kgw_format.py` | Zahlen im deutschen Format |
| `kgw_kernels.py` | Kerne aus Leitwerten, Fehlerfälle, Drehung, Mischung, Kreisströme |
| `kgw_layout.py` | Fester Kopf (Stylesheet) |
| `kgw_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `kgw_ring.py` | Hard-Core-Ring: Zustände, π, Zustandsgraph, Drehung |
| `kgw_rng.py` | SplitMix64 (geprüfte Vorlage, unverändert kopiert) |
| `kgw_simulation.py` | Vektorisierte SplitMix64-Ausgaben, Zeitanteile, Rauschgrenze |
| `kgw_texts.py` | Feste Texte der App (Einleitung, Erklärung, Grenzen, Formeln) |
| `kgw_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): streamlit, numpy, scipy, pandas, plotly |
| `tests/` | Handbeispiele, Orakel, Eigenschaften, Literatur, Kerne, Zufall, Verdikte, Randfälle, Claims, Texte, Presets, Layout, AppTest, Footer, Laufzeit-Importe; `tests/exakt.py` rechnet in Brüchen |

## Bewusst nicht umgesetzt

* **Beweise** (Detailed Balance ⇒ stationär, die Dimension der Flüsse): nur Lehrbuchverweis und numerische Bestätigung.
* **Die Ursachen der Fehlerfälle, der Takt, d̄, ESS und R̂, Kopplung:** gehören zu den Nachbar-Demos (Abschnitt Abgrenzung).
* **Optimierte Kreisströme, andere Graphen, Schranken für nichtreversible Kerne:** nicht gemessen, siehe Ehrliche Grenzen.
* **PDF-Export:** nicht vorgesehen; die Seite ist ein Rechenwerkzeug, keine Auswertung für den Versand.
* **Vorgerechnete oder eingefrorene Tabellen:** keine, alles rechnet live.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, SciPy, pandas und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [MCMC: von der Kette zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-mcmc.html).

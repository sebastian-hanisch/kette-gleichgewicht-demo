"""Textregeln für README, App-Texte, Docstrings und Tests: keine Wörter der eigenen Planung, kein ASCII-Ersatz für Umlaute in Anzeigetexten, einheitliche Begriffe.

Planungswörter gehören nicht in die Demo (der Leser kennt weder die Planung noch die Nummerierung); statt dessen „frühere Messung“, „erste Rechnung“, „Nachbar-Demo“. Die Zeile
„Mehr zur Reihe: …“ im Portfolio-Footer wird von set_demo_footer.py erzeugt und ist kein Planungstext; sie liegt nicht in diesen Dateien, wird aber vorsorglich ausgenommen."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLANUNG = re.compile(r"\b(Stück|Stücke|Stücken|Linie|Linien|Vormessung|Plan|Pläne|V1|V2|V3|mc_v\w*|Phase 0|Kantenabgleich)\b")
VERBOTEN = {
    "Saatwert": "heißt „Zufalls-Seed“",
    "Wärmebad": "heißt „Heat-bath“",
    "trägt": "reicht, gilt, zeigt",
    "tragen": "reicht, gilt, zeigt",
    "läuft lokal": "keine Aussage über die Umgebung",
    "geplant": "kein „geplant“-Zusatz in Verweisen",
}


def _dateien():
    eigene = Path(__file__).resolve()
    out = [ROOT / "README.md", ROOT / "app.py"] + sorted(ROOT.glob("kgw_*.py")) + [p for p in sorted((ROOT / "tests").glob("*.py")) if p.resolve() != eigene]
    return [p for p in out if p.name != "kgw_rng.py"]       # die Vorlage des Zufallsgenerators bleibt unverändert


def _zeilen(path):
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if "Mehr zur Reihe" in line:
            continue
        yield i, line


def test_keine_planungswoerter_in_readme_app_modulen_und_tests():
    treffer = [f"{p.name}:{i}: {m.group(0)}" for p in _dateien() for i, line in _zeilen(p) for m in PLANUNG.finditer(line)]
    assert not treffer, treffer


def test_keine_verbotenen_begriffe():
    treffer = [f"{p.name}:{i}: {w} ({hint})" for p in _dateien() for i, line in _zeilen(p) for w, hint in VERBOTEN.items() if re.search(rf"\b{w}\b", line)]
    assert not treffer, treffer


def test_zufalls_seed_wird_mit_bindestrich_geschrieben():
    falsch = [f"{p.name}:{i}" for p in _dateien() for i, line in _zeilen(p) for m in re.finditer(r"Zufalls[- ]?[Ss]eed|Zufallsseed|Zufalls Seed", line) if m.group(0) != "Zufalls-Seed"]
    assert not falsch, falsch


def test_readme_nennt_die_modellannahme_und_die_keine_neuheit_aussage():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "ausgedacht" in readme and "keine reale Datenquelle" in readme
    assert "Nichts davon ist inhaltlich neu" in readme and "neu ist nur die Vorführbarkeit mit Zähleinheit" in readme


def test_readme_nennt_keine_einbindung_als_offen_und_nur_die_todo_marke_des_live_links():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    todos = [line for line in readme.splitlines() if "TODO" in line]
    assert len(todos) == 1 and "TODO nach dem Deploy" in todos[0], todos
    assert not re.search(r"Einbindung[^.\n]*\boffen\b|\boffen\b[^.\n]*Einbindung", readme)

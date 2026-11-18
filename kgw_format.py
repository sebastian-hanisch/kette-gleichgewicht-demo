"""Zahlen im deutschen Format: Dezimalkomma, Gruppierung mit Leerzeichen ab fünf Stellen (DIN 5008)."""

from fractions import Fraction


def ganz(n) -> str:
    """16796 → „16 796“, 1430 → „1430“."""
    n = int(n)
    s = str(abs(n))
    if len(s) >= 5:
        groups = []
        while s:
            groups.append(s[-3:])
            s = s[:-3]
        s = " ".join(reversed(groups))
    return ("-" if n < 0 else "") + s


def dez(x: float, digits: int = 2) -> str:
    """Dezimalkomma und echtes Minuszeichen; ein Wert, der auf die gezeigten Stellen zu null gerundet wird, bekommt kein Vorzeichen."""
    text = f"{x:.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    return text.replace(".", ",").replace("-", "−")


def schritte(t) -> str:
    """Mischzeit in Schritten; None heißt: die Kette kommt in der gerechneten Zahl von Schritten nicht an."""
    return "nie" if t is None else ganz(t)


def mit_einheit(t) -> str:
    """„15 Schritte“; ohne Ankunft „nie“."""
    return "nie" if t is None else f"{ganz(t)} Schritte"


def bruch(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator)


def prozent(x: float, digits: int = 1) -> str:
    return f"{dez(100 * x, digits)} %"


def zahl(x: float) -> str:
    """Ganze Zahlen ohne Nachkommastellen („2“), sonst mit Komma („0,5“)."""
    return str(int(x)) if float(x) == int(x) else dez(x, 2).rstrip("0").rstrip(",")


def vorzeichen(d: int) -> str:
    """Änderung mit Vorzeichen: „+6“, „−3“, „0“ (echtes Minuszeichen)."""
    return f"+{d}" if d > 0 else (f"−{-d}" if d < 0 else "0")


def spanne(lo: float, hi: float, digits: int = 4) -> str:
    """Ein Wert, wenn beide Enden auf die gezeigten Stellen gleich sind, sonst „lo bis hi“."""
    a, b = dez(lo, digits), dez(hi, digits)
    return a if a == b else f"{a} bis {b}"

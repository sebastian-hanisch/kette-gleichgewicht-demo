"""Exakte Rechnung in Brüchen für die Tests: ein zweiter, unabhängiger Aufbau des Hard-Core-Rings und seiner Kerne.

Zustände sind hier sortierte Tupel belegter Plätze (die App benutzt Bitmasken), Wahrscheinlichkeiten sind `Fraction`. Metropolis und Heat-bath werden aus den Regeln
(Platz wählen, Vorschlag, Annahme) aufgebaut, nicht aus Leitwerten wie in der App; die Leitwert-Sicht steht daneben. Matrixprodukte, Totalvariationsabstand und
Mischzeit laufen in Brüchen, damit Randfälle wie d(t) = 1/4 genau entschieden werden. Nur für kleine Räume (bis etwa 30 Zustände) gedacht."""

from fractions import Fraction as F
from itertools import combinations

import numpy as np


def states(n):
    """Unabhängige Mengen des n-Rings als Tupel, nach Größe, dann lexikographisch."""
    out = []
    for k in range(n // 2 + 1):
        for c in combinations(range(n), k):
            if all(((a + 1) % n) not in c for a in c):
                out.append(c)
    return out


class Raum:
    def __init__(self, n, lam):
        self.n, self.lam = n, F(lam)
        self.st = states(n)
        self.S = len(self.st)
        self.idx = {x: i for i, x in enumerate(self.st)}
        self.w = [self.lam ** len(x) for x in self.st]
        self.Z = sum(self.w)
        self.pi = [wi / self.Z for wi in self.w]
        self.edges = []
        for i, x in enumerate(self.st):
            for v in range(n):
                j = self.idx.get(tuple(sorted(set(x) ^ {v})))
                if j is not None and j > i:
                    self.edges.append((i, j))
        self.deg = [0] * self.S
        for a, b in self.edges:
            self.deg[a] += 1
            self.deg[b] += 1

    def kern_aus_leitwert(self, c):
        """c(a, b) = π_a P(a, b) = π_b P(b, a); die Diagonale ergänzt die Zeilensumme auf 1."""
        P = [[F(0)] * self.S for _ in range(self.S)]
        for a, b in self.edges:
            P[a][b] = c(a, b) / self.pi[a]
            P[b][a] = c(a, b) / self.pi[b]
        for i in range(self.S):
            P[i][i] = 1 - sum(P[i])
        return P

    def metropolis(self, f=F(1)):
        return self.kern_aus_leitwert(lambda a, b: f * min(self.pi[a], self.pi[b]) / self.n)

    def metropolis_nach_regel(self):
        """Platz v mit Wahrscheinlichkeit 1/n, Belegung umkehren, wenn zulässig, annehmen mit min(1, w(y)/w(x)); sonst bleiben."""
        P = [[F(0)] * self.S for _ in range(self.S)]
        for i, x in enumerate(self.st):
            for v in range(self.n):
                j = self.idx.get(tuple(sorted(set(x) ^ {v})))
                if j is None:
                    P[i][i] += F(1, self.n)
                else:
                    a = min(F(1), self.w[j] / self.w[i])
                    P[i][j] += a / self.n
                    P[i][i] += (1 - a) / self.n
        return P

    def waermebad_nach_regel(self, p_beleg=None):
        """Platz gleichverteilt; hat kein Nachbar ein Teilchen: belegen mit p_beleg (Standard λ/(1+λ)), sonst leeren."""
        p = self.lam / (1 + self.lam) if p_beleg is None else F(p_beleg)
        P = [[F(0)] * self.S for _ in range(self.S)]
        for i, x in enumerate(self.st):
            s = set(x)
            for v in range(self.n):
                if ((v - 1) % self.n) in s or ((v + 1) % self.n) in s:
                    P[i][self.idx[tuple(sorted(s - {v}))]] += F(1, self.n)
                else:
                    P[i][self.idx[tuple(sorted(s | {v}))]] += p / self.n
                    P[i][self.idx[tuple(sorted(s - {v}))]] += (1 - p) / self.n
        return P

    def gleiche_leitwerte(self):
        c = min(self.pi[i] / self.deg[i] for i in range(self.S))
        return self.kern_aus_leitwert(lambda a, b: c), c

    def feste_annahme_halb(self):
        P = [[F(0)] * self.S for _ in range(self.S)]
        for a, b in self.edges:
            P[a][b] = F(1, 2 * self.n)
            P[b][a] = F(1, 2 * self.n)
        for i in range(self.S):
            P[i][i] = 1 - sum(P[i])
        return P

    def umgekehrtes_verhaeltnis(self):
        """Annahme min(1, w(x)/w(y)) statt min(1, w(y)/w(x))."""
        P = [[F(0)] * self.S for _ in range(self.S)]
        for i, x in enumerate(self.st):
            for v in range(self.n):
                j = self.idx.get(tuple(sorted(set(x) ^ {v})))
                if j is None:
                    P[i][i] += F(1, self.n)
                else:
                    a = min(F(1), self.w[i] / self.w[j])
                    P[i][j] += a / self.n
                    P[i][i] += (1 - a) / self.n
        return P

    def drehung(self):
        R = [[F(0)] * self.S for _ in range(self.S)]
        for i, x in enumerate(self.st):
            R[i][self.idx[tuple(sorted((v + 1) % self.n for v in x))]] = F(1)
        return R


def mm(A, B):
    return [[sum(A[i][t] * B[t][j] for t in range(len(B)) if A[i][t] != 0) for j in range(len(B[0]))] for i in range(len(A))]


def vm(v, A):
    return [sum(v[i] * A[i][j] for i in range(len(v)) if v[i] != 0) for j in range(len(A[0]))]


def eye(n):
    return [[F(int(i == j)) for j in range(n)] for i in range(n)]


def tv(u, v):
    return sum(abs(a - b) for a, b in zip(u, v)) / 2


def d_t(P, pi, t):
    M = eye(len(P))
    for _ in range(t):
        M = mm(M, P)
    return max(tv(M[i], pi) for i in range(len(P)))


def t_mix(P, pi, tmax=400):
    """Kleinstes t mit d(t) ≤ 1/4, exakt; None, wenn bis tmax nicht erreicht."""
    M = eye(len(P))
    for t in range(tmax + 1):
        if max(tv(M[i], pi) for i in range(len(P))) <= F(1, 4):
            return t
        M = mm(M, P)
    return None


def stationaer(P, pi):
    return vm(pi, P) == pi


def db_abw(P, pi):
    return max(abs(pi[i] * P[i][j] - pi[j] * P[j][i]) for i in range(len(P)) for j in range(len(P)))


def zeilensumme_eins(P):
    return all(sum(r) == 1 and min(r) >= 0 for r in P)


def rang(A):
    """Rang einer Matrix aus Brüchen (Gauß)."""
    A = [r[:] for r in A]
    rk = 0
    for c in range(len(A[0])):
        piv = next((r for r in range(rk, len(A)) if A[r][c] != 0), None)
        if piv is None:
            continue
        A[rk], A[piv] = A[piv], A[rk]
        for r in range(len(A)):
            if r != rk and A[r][c] != 0:
                f = A[r][c] / A[rk][c]
                A[r] = [a - f * b for a, b in zip(A[r], A[rk])]
        rk += 1
    return rk


def erhaltung(S, paare):
    A = [[F(0)] * len(paare) for _ in range(S)]
    for k, (x, y) in enumerate(paare):
        A[y][k] += 1
        A[x][k] -= 1
    return A


def kreis5(p):
    P = [[F(0)] * 5 for _ in range(5)]
    for i in range(5):
        P[i][(i + 1) % 5] += p
        P[i][(i - 1) % 5] += 1 - p
    return P


def bahnen(P):
    """Kommunizierende Klassen einer Permutationsmatrix (Bahnen) als Listen von Zuständen."""
    seen = [False] * len(P)
    out = []
    for i in range(len(P)):
        if seen[i]:
            continue
        comp, stack = [], [i]
        seen[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for j in range(len(P)):
                if P[u][j] != 0 and not seen[j]:
                    seen[j] = True
                    stack.append(j)
        out.append(sorted(comp))
    return out


def als_float(P):
    return np.array([[float(x) for x in r] for r in P])


def app_nach_exakt(ring, raum):
    """Zuordnung App-Zustand → exakter Zustand (über die Menge der belegten Plätze); dieselben Zustände, andere Nummerierung."""
    perm = []
    for m in ring.masks:
        perm.append(raum.idx[tuple(v for v in range(ring.n) if (m >> v) & 1)])
    return perm


def umnummeriert(P_app, perm, P_exakt):
    """Größter Abstand zwischen der App-Matrix und der exakten Matrix, beide im Zustandsraum der App geordnet."""
    exakt = als_float(P_exakt)
    return float(np.abs(P_app - exakt[np.ix_(perm, perm)]).max())

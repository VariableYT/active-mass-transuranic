"""ra_atomic.py - atomic sector: what precision spectroscopy would need to see the paper's force.

No isotope-shift or transition-frequency data are ingested in this pass, so every atomic observable is reported as
DATA_GAP. What this module does compute is the size of the effect, which decides whether ingesting such data (and
building a King-plot analyser for it) could ever pay off for this particular force.

The paper's interaction between two bodies is V = -(G m_a m_b / r) q_a q_b exp(-r / lambda), with lambda of order
0.5 m. On atomic scales the exponential is 1, so between an electron and a nucleus it is a 1/r potential, like
the Coulomb term, with relative strength

    eps = G m_e M_nuc q_e q_nuc / (Z alpha hbar c) = [G m_e u / (alpha hbar c)] (A / Z) (q_e q_nuc).

A 1/r term of relative size eps acts like a change of the nuclear charge by Z eps. To first order a valence
electron's binding energy shifts by 2 (Z eps) / Z_eff of itself; Z_eff = 1 is used, which overestimates the shift
for most optical transitions only mildly and is good enough for an order-of-magnitude reach.

The electron carries the charge of ordinary matter, q_e = q0, only in the paper's mass-like reading; in the
baryon-number reading it carries none and the effect is exactly zero. The numbers below are therefore upper limits
(mass-like reading, q_e q_nuc = P_X, the same product a torsion balance measures).
"""
import math

G_N, M_E, U_AMU = 6.67430e-11, 9.1093837015e-31, 1.66053906660e-27
ALPHA, HBARC = 7.2973525693e-3, 3.1615267734966903e-26          # hbar c in J m
PREFACTOR = G_N * M_E * U_AMU / (ALPHA * HBARC)                 # gravity / Coulomb for an electron and one u
BEST_CLOCK_FRACTION = 1e-18   # order of the best optical frequency comparisons (stable or long-lived species only)

ZC, NC = 114, 184
D2U = (92 - ZC) ** 2 + (146 - NC) ** 2


def max_P(Z, N, Pc_ceiling, B_U):
    """Largest P_X for nuclide X under the paper's Gaussian switch: the centre is capped at Pc_ceiling (superheavy
    Q_alpha) and uranium at B_U (B_U = 1 means no uranium bound, the mass-like reading). Maximised over width."""
    d2 = (Z - ZC) ** 2 + (N - NC) ** 2
    if d2 == 0:
        return Pc_ceiling
    best = 0.0
    for i in range(50, 2001):                       # widths 0.5 .. 20 nucleons
        s = i / 100
        pc = min(Pc_ceiling, B_U * math.exp(min(D2U / (2 * s * s), 700.0)))
        best = max(best, pc * math.exp(-d2 / (2 * s * s)))
    return best


def reach(Z, N, P):
    A = Z + N
    eps = PREFACTOR * (A / Z) * P
    frac = 2 * Z * eps
    return dict(P_X=P, coulomb_fraction=eps, fractional_frequency_shift=frac,
                short_of_best_clocks_by=BEST_CLOCK_FRACTION / frac if frac > 0 else math.inf)


def report(nuclides, Pc_ceiling_mass_like):
    """nuclides: list of (label, Z, N). Returns one DATA_GAP record per nuclide."""
    out = []
    for label, Z, N in nuclides:
        P = max_P(Z, N, Pc_ceiling_mass_like, 1.0)
        r = reach(Z, N, P)
        r.update(nuclide=label, Z=Z, N=N, status="DATA_GAP",
                 observable="optical transition frequencies / isotope shifts (King plot)",
                 note="no dataset ingested; effect is an upper limit (mass-like reading, zero in the "
                      "baryon-number reading)")
        out.append(r)
    return out

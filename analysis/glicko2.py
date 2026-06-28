"""Glicko-2 rating system implementation.

Reference: Mark E. Glickman, "Example of the Glicko-2 system" (2012).
http://www.glicko.net/glicko/glicko2.pdf

All arithmetic uses native Python floats. No JAX or NumPy dependency.
"""

from __future__ import annotations

import math
from typing import List, Tuple


# Default system constant. Controls how quickly volatility can change.
# Glickman recommends 0.3–1.2; 0.5 is a common default.
DEFAULT_TAU = 0.5
DEFAULT_INITIAL_RATING = 1500.0
DEFAULT_INITIAL_RD = 350.0
DEFAULT_INITIAL_VOLATILITY = 0.06

# Convergence tolerance for the Illinois algorithm (Step 5 in Glickman 2012).
_EPSILON = 1e-6

# Conversion factor between Glicko-2 internal scale and external rating.
_SCALE = 173.7178

# Maximum rating change allowed in a single rating period. Caps `mu_prime - mu`
# to `_MAX_RATING_STEP / _SCALE`. Guards against runaway when a player's prior
# (mu, phi) sits far from the population mean and a single update overshoots
# into the float64-saturated region of E (where v_inv → 0 and the player can
# no longer be re-rated).  ~1000 ≈ 5.76 mu units, big enough that normal
# updates are untouched but small enough to keep ratings inside the
# well-conditioned region.
_MAX_RATING_STEP = 1000.0


def _to_glicko2(rating: float, rd: float) -> Tuple[float, float]:
    """Convert external (Elo-like) rating and RD to Glicko-2 internal scale."""
    mu = (rating - 1500.0) / _SCALE
    phi = rd / _SCALE
    return mu, phi


def _from_glicko2(mu: float, phi: float) -> Tuple[float, float]:
    """Convert Glicko-2 internal scale back to external rating and RD."""
    rating = _SCALE * mu + 1500.0
    rd = _SCALE * phi
    return rating, rd


def _g(phi: float) -> float:
    """Glicko-2 g function."""
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    """Glicko-2 E (expected score) function."""
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def update_glicko2(
    rating: float,
    rd: float,
    volatility: float,
    opponents: List[Tuple[float, float, float]],
    tau: float = DEFAULT_TAU,
) -> Tuple[float, float, float]:
    """Apply one rating period and return updated (rating, rd, volatility).

    Args:
        rating: Current external rating (Elo-like scale, e.g. 1500).
        rd: Current rating deviation (external scale, e.g. 350).
        volatility: Current volatility (dimensionless, e.g. 0.06).
        opponents: List of (opp_rating, opp_rd, score) tuples.
            score is 1.0 for a win, 0.5 for draw, 0.0 for a loss.
            Passing an empty list applies Step 6 only (rd grows, no update).
        tau: System constant (τ). Controls max volatility change per period.

    Returns:
        (new_rating, new_rd, new_volatility) in external scale.
    """
    mu, phi = _to_glicko2(rating, rd)

    if not opponents:
        # Step 6: player did not compete; increase RD toward prior.
        phi_star = math.sqrt(phi * phi + volatility * volatility)
        new_rating, new_rd = _from_glicko2(mu, phi_star)
        return new_rating, new_rd, volatility

    # Convert opponents to internal scale.
    opp_internal = [
        (_to_glicko2(r, rj)[0], _to_glicko2(r, rj)[1], s)
        for r, rj, s in opponents
    ]

    # Step 3: compute v (estimated variance of the player's rating).
    v_inv = 0.0
    for mu_j, phi_j, _ in opp_internal:
        g_j = _g(phi_j)
        e_j = _E(mu, mu_j, phi_j)
        v_inv += g_j * g_j * e_j * (1.0 - e_j)
    if v_inv == 0.0:
        # Degenerate case: no variance information, skip update.
        return rating, rd, volatility
    v = 1.0 / v_inv

    # Step 4: compute delta (improvement in rating scaled by v).
    delta_sum = 0.0
    for mu_j, phi_j, s_j in opp_internal:
        g_j = _g(phi_j)
        e_j = _E(mu, mu_j, phi_j)
        delta_sum += g_j * (s_j - e_j)
    delta = v * delta_sum

    # Step 5: determine new volatility σ' using the Illinois algorithm.
    sigma = volatility
    a = math.log(sigma * sigma)
    phi2 = phi * phi

    def _f(x: float) -> float:
        ex = math.exp(x)
        d2 = phi2 + v + ex
        lhs = ex * (delta * delta - d2) / (2.0 * d2 * d2)
        rhs = (x - a) / (tau * tau)
        return lhs - rhs

    # Bracket the root.
    A = a
    if delta * delta > phi2 + v:
        B = math.log(delta * delta - phi2 - v)
    else:
        k = 1
        while _f(a - k * tau) < 0:
            k += 1
        B = a - k * tau

    fA = _f(A)
    fB = _f(B)
    while abs(B - A) > _EPSILON:
        C = A + (A - B) * fA / (fB - fA)
        fC = _f(C)
        if fC * fB <= 0:
            A = B
            fA = fB
        else:
            fA /= 2.0
        B = C
        fB = fC

    sigma_prime = math.exp(A / 2.0)

    # Step 6: update RD to new pre-rating value.
    phi_star = math.sqrt(phi2 + sigma_prime * sigma_prime)

    # Step 7: update rating and RD.
    phi_prime = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    mu_prime = mu + phi_prime * phi_prime * delta_sum

    # Clamp single-period rating change to keep mu inside the float64-resolvable
    # region of E. Without this, a fresh player (large phi) inserted into a
    # high-rated population can overshoot far enough that exp(-g·(mu−mu_j))
    # underflows to 0, making E exactly 1 and v_inv exactly 0 forever after.
    max_mu_step = _MAX_RATING_STEP / _SCALE
    mu_step = mu_prime - mu
    if mu_step > max_mu_step:
        mu_prime = mu + max_mu_step
    elif mu_step < -max_mu_step:
        mu_prime = mu - max_mu_step

    new_rating, new_rd = _from_glicko2(mu_prime, phi_prime)
    return new_rating, new_rd, sigma_prime


def rd_after_inactivity(rd: float, volatility: float, periods: float) -> float:
    """Increase RD to account for inactivity (periods of no games).

    Args:
        rd: Current RD in external scale.
        volatility: Current volatility.
        periods: Number of rating periods elapsed without games.

    Returns:
        New RD in external scale, capped at DEFAULT_INITIAL_RD.
    """
    phi = rd / _SCALE
    phi_star = math.sqrt(phi * phi + volatility * volatility * periods)
    new_rd = _SCALE * phi_star
    return min(new_rd, DEFAULT_INITIAL_RD)

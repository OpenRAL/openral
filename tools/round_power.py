#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""How many validation-matrix runs does a comparison actually need?

Issue #217: decide the round count from the effect size worth detecting before
running anything. Motivated by two under-powered conclusions: the n=1 reading
on #176, and the 20-run 2026-08-26 vs 2026-09-04 comparison, which had under
30% power against the effect it observed (25% -> 5%) — "p = 0.18, not
separable" (#217) was correct arithmetic on a sample that could not separate
anything.

* ``required`` — runs per arm for a target power against a stated effect.
* ``power`` — power of a battery already planned.

Exact (not Monte Carlo), stdlib-only, deterministic (CLAUDE.md §1.8): power is
the sum, over every ``(a, b)`` outcome pair under the two binomials, of the
pairs Fisher's exact test would reject. ``O(n²)`` Fisher evaluations, each
``O(n)`` — sub-second to n=100, seconds by n=300; the ladder stops at 800
(unaffordable beyond that).

Example:
    $ python tools/round_power.py --baseline 0.25 --alternative 0.05
    baseline 25.0% vs alternative 5.0%, alpha 0.05, two-sided Fisher
      n =  20/arm  power 0.30
      n =  30/arm  power 0.44
      n =  40/arm  power 0.65
      n =  60/arm  power 0.85  <- first n at or above 80% power
    80% power needs 60 runs per arm (15 four-scene rounds per arm).
"""

from __future__ import annotations

import argparse
from math import comb, isclose

#: Scene runs in one validation-matrix round. `tools/validation_matrix.py`
#: runs the four-scene matrix, so a round contributes four runs to an arm.
RUNS_PER_ROUND = 4

#: The ladder `required()` walks. Beyond this a battery is not affordable on
#: any host this programme uses, and the honest output is "more than 800".
_LADDER = (20, 30, 40, 60, 80, 100, 120, 160, 200, 260, 320, 400, 500, 650, 800)


def fisher_exact_two_sided(a: int, n1: int, b: int, n2: int) -> float:
    """Two-sided Fisher p-value for successes ``a``/``n1`` against ``b``/``n2``.

    The conventional two-sided form: sum the hypergeometric probability of
    every table at least as extreme as the observed one, "as extreme" meaning
    "no more probable", which is what ``scipy.stats.fisher_exact`` computes.

    Args:
        a: Successes in the first arm.
        n1: Runs in the first arm.
        b: Successes in the second arm.
        n2: Runs in the second arm.

    Returns:
        The p-value, in ``[0.0, 1.0]``.

    Example:
        >>> round(fisher_exact_two_sided(5, 20, 1, 20), 4)
        0.1818
    """
    total = n1 + n2
    successes = a + b
    if successes in (0, total):
        return 1.0
    denominator = comb(total, successes)
    observed = comb(n1, a) * comb(n2, b) / denominator
    # A strict `<` would drop the observed table's own mirror image whenever
    # floating point lands a hair apart; `isclose` keeps the test conservative.
    tail = 0.0
    low = max(0, successes - n2)
    high = min(n1, successes)
    for k in range(low, high + 1):
        probability = comb(n1, k) * comb(n2, successes - k) / denominator
        if probability < observed or isclose(probability, observed, rel_tol=1e-12):
            tail += probability
    return min(1.0, tail)


def power(baseline: float, alternative: float, n: int, *, alpha: float = 0.05) -> float:
    """Probability that ``n`` runs per arm reject at ``alpha``, computed exactly.

    Enumerates every ``(a, b)`` outcome under ``Binomial(n, baseline)`` and
    ``Binomial(n, alternative)`` and sums the probability of the pairs Fisher
    would call significant.

    Args:
        baseline: True success rate of the first arm.
        alternative: True success rate of the second arm.
        n: Runs in each arm.
        alpha: Significance threshold.

    Returns:
        Power, in ``[0.0, 1.0]``.

    Example:
        >>> round(power(0.25, 0.05, 20), 3)
        0.299
    """
    first = [comb(n, k) * baseline**k * (1 - baseline) ** (n - k) for k in range(n + 1)]
    second = [comb(n, k) * alternative**k * (1 - alternative) ** (n - k) for k in range(n + 1)]
    # Fisher's p depends only on (a, b), so evaluate each cell once.
    detected = 0.0
    for a, pa in enumerate(first):
        if pa == 0.0:
            continue
        for b, pb in enumerate(second):
            if pb == 0.0:
                continue
            if fisher_exact_two_sided(a, n, b, n) < alpha:
                detected += pa * pb
    return detected


def required(
    baseline: float, alternative: float, *, alpha: float = 0.05, target: float = 0.80
) -> tuple[int | None, float]:
    """Smallest ladder ``n`` per arm reaching ``target`` power, or ``(None, best)``.

    Args:
        baseline: True success rate of the first arm.
        alternative: True success rate of the second arm.
        alpha: Significance threshold.
        target: Power to reach.

    Returns:
        ``(n, power_at_n)``, or ``(None, power_at_800)`` when the ladder tops out.

    Example:
        >>> n, _ = required(0.25, 0.05)
        >>> n
        60
    """
    achieved = 0.0
    for n in _LADDER:
        achieved = power(baseline, alternative, n, alpha=alpha)
        if achieved >= target:
            return n, achieved
    return None, achieved


def main(argv: list[str] | None = None) -> int:
    """Print the power ladder for one comparison."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=float, required=True, help="Rate of arm A, 0..1.")
    parser.add_argument("--alternative", type=float, required=True, help="Rate of arm B, 0..1.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80, dest="target")
    args = parser.parse_args(argv)

    print(
        f"baseline {args.baseline:.1%} vs alternative {args.alternative:.1%}, "
        f"alpha {args.alpha}, two-sided Fisher"
    )
    hit: int | None = None
    for n in _LADDER:
        achieved = power(args.baseline, args.alternative, n, alpha=args.alpha)
        marker = ""
        if hit is None and achieved >= args.target:
            hit = n
            marker = f"  <- first n at or above {args.target:.0%} power"
        print(f"  n = {n:3d}/arm  power {achieved:.2f}{marker}")
        if hit is not None:
            break
    if hit is None:
        print(f"no ladder size reaches {args.target:.0%} power; more than {_LADDER[-1]} per arm.")
        return 1
    print(
        f"{args.target:.0%} power needs {hit} runs per arm "
        f"({hit // RUNS_PER_ROUND} four-scene rounds per arm)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

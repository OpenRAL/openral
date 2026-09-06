#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""How many validation-matrix runs does a comparison actually need?

Issue #217, step 1: *"Power first. Decide the round count from the effect size
worth detecting before running anything."*

This exists because the collision programme has now twice drawn a conclusion
from a battery that could not have supported one, and had to take it back:
the n=1 reading on #176, and the 2026-08-26 vs 2026-09-04 success comparison
that #217 was opened to settle. The second is the sharper lesson — a 20-run
arm has **under 30 % power** against the very effect it observed (25 % → 5 %), so
"p = 0.18, not separable" was the arithmetic working correctly on a sample
that was never going to separate anything. Running it again, the same size,
answers nothing again.

Two numbers come out, and both are worth having before a battery rather than
after:

* ``required``  — runs per arm for a target power against a stated effect.
* ``power``     — what a battery you have already planned can actually see.

Exact, not Monte Carlo, and stdlib-only. The test that a battery reports is
Fisher's exact test on the 2×2 of successes, so power is computed by
enumerating every outcome pair ``(a, b)`` under the two binomials and summing
the probability of those the test would reject. That makes the answer
deterministic — the same inputs give the same number on every host and in
every re-run, which a sampled estimate does not (CLAUDE.md §1.8) — and it
avoids putting scipy in the workspace for a planning script.

Cost is ``O(n²)`` Fisher evaluations, each ``O(n)``. Sub-second to ``n = 100``,
a few seconds by ``n = 300``; the ladder stops at 800 because a comparison
needing more runs than that is not a comparison this programme can afford, and
saying so is the useful answer.

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

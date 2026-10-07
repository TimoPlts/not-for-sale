"""How much does a Sharpe ratio prove? The probabilistic and deflated Sharpe ratios.

A Sharpe ratio measured on a sample is itself uncertain. It is more
uncertain when the sample is short, the returns are skewed or they have fat
tails (crypto returns usually have both).

* **Probabilistic Sharpe ratio** (Bailey and López de Prado, 2012): the
  probability that the true Sharpe ratio is above a threshold (0 by
  default), given the measured one, the number of returns, their skewness
  and their kurtosis::

      PSR = Φ( (SR - SR*) * sqrt(n - 1) / sqrt(1 - γ3 * SR + (γ4 - 1) / 4 * SR²) )

  where SR is the per-bar Sharpe ratio (mean / sample std), γ3 the skewness
  and γ4 the (non-excess) kurtosis.
* **Deflated Sharpe ratio** (Bailey and López de Prado, 2014): the PSR of
  the best of N tried configurations, measured against the Sharpe ratio the
  best of N would reach by luck alone::

      SR0 = sqrt(V) * ((1 - γ) * Φ⁻¹(1 - 1/N) + γ * Φ⁻¹(1 - 1/(N e)))

  where V is the variance of the N trials' per-bar Sharpe ratios and γ the
  Euler-Mascheroni constant. Trying more settings raises the bar. A DSR
  of 0.95 or more is the usual standard; below 0.5 the winner is no better
  than the luckiest of the trials would be.

Everything is per bar. The annualised Sharpe ratio in the metrics is the
per-bar one times sqrt(bars per year).
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Sequence

import numpy as np

EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


def per_bar_sharpe(returns: Sequence[float]) -> float | None:
    values = np.asarray(returns, dtype="float64")
    if values.size < 2:
        return None
    std = float(np.std(values, ddof=1))
    return None if std <= 0 else float(np.mean(values)) / std


def probabilistic_sharpe(returns: Sequence[float], threshold: float = 0.0) -> float | None:
    """P(true per-bar Sharpe > ``threshold``); None when undefined (fewer than 3 returns, no variance)."""
    values = np.asarray(returns, dtype="float64")
    sr = per_bar_sharpe(values)
    if sr is None or values.size < 3:
        return None
    centred = values - values.mean()
    m2 = float(np.mean(centred**2))
    if m2 <= 0:
        return None
    skew = float(np.mean(centred**3)) / m2**1.5
    kurt = float(np.mean(centred**4)) / m2**2
    variance = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if not math.isfinite(variance) or variance <= 0:
        return None
    return _NORMAL.cdf((sr - threshold) * math.sqrt(values.size - 1) / math.sqrt(variance))


def expected_max_sharpe(n_trials: int, variance: float) -> float:
    """The per-bar Sharpe ratio the best of ``n_trials`` reaches by luck (their true Sharpe being 0)."""
    if n_trials < 2 or variance <= 0:
        return 0.0
    return math.sqrt(variance) * ((1 - EULER_GAMMA) * _NORMAL.inv_cdf(1 - 1 / n_trials)
                                  + EULER_GAMMA * _NORMAL.inv_cdf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe(returns: Sequence[float], trial_sharpes: Sequence[float | None]) -> float | None:
    """PSR of ``returns`` against the luck-only best of the trials (per-bar Sharpe ratios, this one included)."""
    trials = [s for s in trial_sharpes if s is not None and math.isfinite(s)]
    variance = float(np.var(trials, ddof=1)) if len(trials) >= 2 else 0.0
    return probabilistic_sharpe(returns, expected_max_sharpe(len(trials), variance))

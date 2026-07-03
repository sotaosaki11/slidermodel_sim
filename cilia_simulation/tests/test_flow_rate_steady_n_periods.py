"""
steady_n_periods による流量平均のテスト。
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.flow_rate import FlowCalculator
from core.solver import TwoSliderResult


def _make_periodic_two_slider_result(
    *,
    n_periods: int,
    n_eval_per_period: int,
    omega: float,
    amplitude: float,
) -> TwoSliderResult:
    period = 2.0 * math.pi / omega
    n_eval = n_periods * n_eval_per_period + 1
    t = np.linspace(0.0, n_periods * period, n_eval)
    s1 = amplitude * np.sin(omega * t)
    s2 = amplitude * np.cos(omega * t)
    f1_total = np.sin(omega * t)
    f2_total = np.cos(omega * t)
    steady_start_index = int(np.searchsorted(t, (n_periods - 1) * period, side="left"))
    return TwoSliderResult(
        t=t,
        s1=s1,
        s2=s2,
        f1_total=f1_total,
        f2_total=f2_total,
        period=period,
        n_periods=n_periods,
        steady_start_index=steady_start_index,
    )


class TestFlowRateSteadyNPeriods(unittest.TestCase):
  def test_five_period_average_matches_one_period_for_periodic_signal(self) -> None:
    flow = FlowCalculator(mu=1.0, h=1.0)
    result = _make_periodic_two_slider_result(
      n_periods=10,
      n_eval_per_period=200,
      omega=math.pi,
      amplitude=0.1,
    )
    phi = math.pi / 6.0

    _, _, q_one = flow.compute_two_slider_Q_from_result(
      result=result,
      phi=phi,
      use_steady_window=True,
      steady_n_periods=1,
    )
    _, _, q_five = flow.compute_two_slider_Q_from_result(
      result=result,
      phi=phi,
      use_steady_window=True,
      steady_n_periods=5,
    )
    self.assertAlmostEqual(q_one, q_five, places=10)

  def test_invalid_steady_n_periods_raises(self) -> None:
    flow = FlowCalculator(mu=1.0, h=1.0)
    result = _make_periodic_two_slider_result(
      n_periods=3,
      n_eval_per_period=50,
      omega=math.pi,
      amplitude=0.1,
    )
    with self.assertRaises(ValueError):
      flow.compute_two_slider_Q_from_result(
        result=result,
        phi=0.0,
        steady_n_periods=0,
      )


if __name__ == "__main__":
  unittest.main()

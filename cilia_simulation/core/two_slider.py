"""
2本スライダー実験向けの薄いラッパ / Thin helpers for two-slider experiments.

1ケース分の積分と周期平均流量 Q をまとめて返す。
"""

from __future__ import annotations

from core.flow_rate import FlowCalculator
from core.hydrodynamics import TwoSliderMobility
from core.solver import SolverConfig, TwoSliderTimeStepper


def compute_two_slider_Q(
    *,
    mu: float,
    a: float,
    k: float,
    F_0: float,
    omega: float,
    phi: float,
    h: float,
    l: float,
    delta: float,
    s1_0: float,
    s2_0: float,
    solver_config: SolverConfig,
) -> float:
    """
    2本スライダー1ケースを積分し、定常窓1周期の平均流量 Q を返す。
    """
    mobility = TwoSliderMobility(mu=mu, a=a)
    flow = FlowCalculator(mu=mu, h=h)
    stepper = TwoSliderTimeStepper(
        mobility=mobility,
        omega=omega,
        k=k,
        F_0=F_0,
        phi=phi,
        l=l,
        h=h,
        delta=delta,
        s1_0=s1_0,
        s2_0=s2_0,
        config=solver_config,
    )
    result = stepper.run()
    _, _, Q = flow.compute_two_slider_Q_from_result(
        result=result,
        phi=phi,
        use_steady_window=True,
    )
    return float(Q)

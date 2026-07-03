"""
2本スライダー実験向けの薄いラッパ / Thin helpers for two-slider experiments.

1ケース分の積分と周期平均流量 Q をまとめて返す。

- compute_two_slider_Q: 壁なし Oseen（exp02/exp03）
- compute_two_slider_Q_blakelet: 壁あり Blakelet（exp04）
"""

from __future__ import annotations

from core.flow_rate import FlowCalculator
from core.hydrodynamics import BlakeletTwoSliderMobility, TwoSliderMobility
from core.solver import SolverConfig, TwoSliderResult, TwoSliderTimeStepper


def integrate_two_slider_case(
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
    use_blakelet: bool = False,
    layout_theta: float = 0.0,
) -> TwoSliderResult:
    """
    2本スライダー1ケースを積分し、軌道時系列 TwoSliderResult を返す。

    Parameters
    ----------
    use_blakelet : bool
        True なら BlakeletTwoSliderMobility（exp05）、False なら Oseen（exp03）。
    layout_theta : float
        x-y 平面内の相対配置角（exp06）。0 で論文配置。

    Returns
    -------
    TwoSliderResult
        全積分区間の s1, s2 時系列と定常窓インデックス。
    """
    if use_blakelet:
        mobility = BlakeletTwoSliderMobility(mu=mu, a=a)
    else:
        mobility = TwoSliderMobility(mu=mu, a=a)
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
        layout_theta=layout_theta,
        config=solver_config,
    )
    return stepper.run()


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
    layout_theta: float = 0.0,
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
        layout_theta=layout_theta,
        config=solver_config,
    )
    result = stepper.run()
    _, _, Q = flow.compute_two_slider_Q_from_result(
        result=result,
        phi=phi,
        use_steady_window=True,
    )
    return float(Q)


def compute_two_slider_Q_blakelet(
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
    layout_theta: float = 0.0,
    steady_n_periods: int = 1,
) -> float:
    """
    2本スライダー1ケースを Blakelet 移動度で積分し、定常窓の平均流量 Q を返す。

    compute_two_slider_Q と同一の引数・戻り値。移動度のみ
    TwoSliderMobility（Oseen）から BlakeletTwoSliderMobility に差し替える。
    流量評価は FlowCalculator（論文 式(2.61)(4.65)）をそのまま用いる。

    Parameters
    ----------
    mu, a, k, F_0, omega, phi, h, l, delta, s1_0, s2_0
        物理・幾何パラメータ（無次元）。
    solver_config : SolverConfig
        積分器設定（周期数、刻み、RK45/Euler など）。
    layout_theta : float
        x-y 平面内の相対配置角（exp06）。0 で論文配置。
    steady_n_periods : int
        流量評価に使う定常窓の周期数（デフォルト 1）。

    Returns
    -------
    float
        定常窓における周期平均流量 Q。
    """
    mobility = BlakeletTwoSliderMobility(mu=mu, a=a)
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
        layout_theta=layout_theta,
        config=solver_config,
    )
    result = stepper.run()
    _, _, Q = flow.compute_two_slider_Q_from_result(
        result=result,
        phi=phi,
        use_steady_window=True,
        steady_n_periods=steady_n_periods,
    )
    return float(Q)

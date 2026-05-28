"""
進捗表示ユーティリティ / Progress helpers for long sweeps.

実験スクリプトから再利用できるよう、tqdm と ETA(EMA) 計算を共通化する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tqdm import tqdm


@dataclass
class SweepProgressTracker:
    """
    掃引計算向けの進捗表示トラッカー。

    Parameters
    ----------
    total_cases : int
        全ケース数。
    alpha : float, optional
        EMA の係数。0<alpha<=1。
    desc : str, optional
        tqdm の説明テキスト。
    """

    total_cases: int
    alpha: float = 0.2
    desc: str = "sweep"

    def __post_init__(self) -> None:
        if self.total_cases <= 0:
            raise ValueError("total_cases must be positive.")
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha must satisfy 0 < alpha <= 1.")
        self._done_cases = 0
        self._ema_case_seconds: float | None = None
        self._bar: tqdm | None = None

    def __enter__(self) -> "SweepProgressTracker":
        self._bar = tqdm(
            total=self.total_cases,
            desc=self.desc,
            dynamic_ncols=True,
            unit="case",
            leave=False,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def update(self, *, l_value: float, delta_rad: float, case_seconds: float) -> None:
        """
        1ケース完了ごとに進捗と ETA を更新する。
        """
        if self._bar is None:
            raise RuntimeError("SweepProgressTracker must be used as a context manager.")

        self._done_cases += 1
        if self._ema_case_seconds is None:
            self._ema_case_seconds = case_seconds
        else:
            self._ema_case_seconds = (
                self.alpha * case_seconds + (1.0 - self.alpha) * self._ema_case_seconds
            )

        remaining = max(self.total_cases - self._done_cases, 0)
        eta_seconds = remaining * self._ema_case_seconds
        self._bar.set_postfix(
            l=f"{l_value:.3f}",
            delta_deg=f"{math.degrees(delta_rad):.1f}",
            eta_ema_s=f"{eta_seconds:.1f}",
        )
        self._bar.update(1)

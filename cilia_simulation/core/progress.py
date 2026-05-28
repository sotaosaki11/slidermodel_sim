"""
進捗表示ユーティリティ / Progress helpers for long sweeps.

実験スクリプトから再利用できるよう、tqdm と ETA(EMA) 計算を共通化する。
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass

from tqdm import tqdm


def _format_duration(seconds: float) -> str:
    """
    秒を読みやすい時間表記へ変換する（s / m+s / h+m+s）。
    """
    total_seconds = max(int(round(seconds)), 0)
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h{minutes}m{secs}s"
    if minutes > 0:
        return f"{minutes}m{secs}s"
    return f"{secs}s"


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
    update_every_cases: int = 10
    mininterval: float = 0.5

    def __post_init__(self) -> None:
        if self.total_cases <= 0:
            raise ValueError("total_cases must be positive.")
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha must satisfy 0 < alpha <= 1.")
        if self.update_every_cases <= 0:
            raise ValueError("update_every_cases must be positive.")
        if self.mininterval < 0.0:
            raise ValueError("mininterval must be non-negative.")
        self._done_cases = 0
        self._ema_case_seconds: float | None = None
        self._bar: tqdm | None = None
        self._start_time = 0.0

    def __enter__(self) -> "SweepProgressTracker":
        self._bar = tqdm(
            total=self.total_cases,
            dynamic_ncols=False,
            leave=False,
            mininterval=self.mininterval,
            bar_format="{postfix}",
        )
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def update(self, *, case_seconds: float) -> None:
        """
        1ケース完了ごとに進捗と ETA を更新する。

        表示は tqdm の postfix のみ:
        progress=done/total, time_passed=..., time_left=...
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
        self._bar.update(1)
        should_refresh = (
            self._done_cases % self.update_every_cases == 0
            or self._done_cases == self.total_cases
        )
        if should_refresh:
            elapsed_seconds = max(time.perf_counter() - self._start_time, 0.0)
            self._bar.set_postfix(
                ordered_dict=OrderedDict(
                    [
                        ("progress", f"{self._done_cases}/{self.total_cases}"),
                        ("time_passed", _format_duration(elapsed_seconds)),
                        ("time_left", _format_duration(eta_seconds)),
                    ]
                ),
                refresh=True,
            )

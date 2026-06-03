"""
進捗表示ユーティリティ / Progress helpers for long sweeps.

実験スクリプトから再利用できるよう、tqdm と ETA 表示を共通化する。
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


def _format_eta_duration(
    seconds: float,
    *,
    round_minutes_above: float = 600.0,
) -> str:
    """
    残り時間表示向けの時間表記。長時間 ETA は粗く丸めて揺らぎを抑える。

    - seconds >= 3600: 5 分単位
    - seconds >= round_minutes_above (default 600): 分単位（秒なし）
    - それ以外: _format_duration と同じ
    """
    seconds = max(float(seconds), 0.0)
    if seconds >= 3600.0:
        rounded = int(round(seconds / 300.0)) * 300
        hours, rem = divmod(rounded, 3600)
        minutes = rem // 60
        if minutes > 0:
            return f"{hours}h{minutes}m"
        return f"{hours}h"
    if seconds >= round_minutes_above:
        rounded = int(round(seconds / 60.0)) * 60
        hours, rem = divmod(rounded, 3600)
        minutes = rem // 60
        if hours > 0:
            return f"{hours}h{minutes}m"
        return f"{minutes}m"
    return _format_duration(seconds)


def _throughput_eta_seconds(*, done: int, total: int, elapsed_seconds: float) -> float:
    """スループットベースの生 ETA [s]。"""
    remaining = max(total - done, 0)
    if done <= 0:
        return 0.0
    return remaining * (elapsed_seconds / done)


def _effective_eta_min_cases(*, total_cases: int, eta_min_cases: int) -> int:
    """
    eta_min_cases=0 のとき自動閾値を返す。

    全件数の約 0.3%（total//300）を目安に、15〜60 件にクリップする。
    fine（152k 件）でも 60 件程度で ETA 表示に切り替わる。
    """
    if eta_min_cases > 0:
        return eta_min_cases
    return min(60, max(15, total_cases // 300))


def _smooth_eta_display(
    eta_raw: float,
    *,
    eta_display_ema: float | None,
    eta_alpha: float,
) -> tuple[float, float]:
    """
    表示用 ETA の EMA を更新する。

    Returns
    -------
    (new_eta_display_ema, smoothed_eta_seconds)
    """
    if eta_display_ema is None:
        return eta_raw, eta_raw
    smoothed = eta_alpha * eta_raw + (1.0 - eta_alpha) * eta_display_ema
    return smoothed, smoothed


@dataclass
class SweepProgressTracker:
    """
    掃引計算向けの進捗表示トラッカー。

    Parameters
    ----------
    total_cases : int
        全ケース数。
    alpha : float, optional
        case_seconds 用 EMA 係数（0<alpha<=1）。
    eta_alpha : float, optional
        表示 time_left 用 EMA 係数。小さいほど滑らか。
    eta_min_cases : int, optional
        この件数未満では time_left=estimating。0 なら min(60, max(15, total//300))。
    eta_round_minutes_above : float, optional
        この秒数以上の ETA は分単位表示（3600 以上は 5 分単位）。
    desc : str, optional
        tqdm の説明テキスト。
    """

    total_cases: int
    alpha: float = 0.2
    eta_alpha: float = 0.08
    eta_min_cases: int = 0
    eta_round_minutes_above: float = 600.0
    desc: str = "sweep"
    update_every_cases: int = 10
    mininterval: float = 0.5

    def __post_init__(self) -> None:
        if self.total_cases <= 0:
            raise ValueError("total_cases must be positive.")
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha must satisfy 0 < alpha <= 1.")
        if not (0.0 < self.eta_alpha <= 1.0):
            raise ValueError("eta_alpha must satisfy 0 < eta_alpha <= 1.")
        if self.eta_min_cases < 0:
            raise ValueError("eta_min_cases must be non-negative.")
        if self.eta_round_minutes_above < 0.0:
            raise ValueError("eta_round_minutes_above must be non-negative.")
        if self.update_every_cases <= 0:
            raise ValueError("update_every_cases must be positive.")
        if self.mininterval < 0.0:
            raise ValueError("mininterval must be non-negative.")
        self._done_cases = 0
        self._ema_case_seconds: float | None = None
        self._eta_display_ema: float | None = None
        self._bar: tqdm | None = None
        self._start_time = 0.0
        self._eta_min_cases_effective = _effective_eta_min_cases(
            total_cases=self.total_cases,
            eta_min_cases=self.eta_min_cases,
        )

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

        time_left はスループット ETA を eta_alpha で平滑化して表示する。
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

        elapsed_seconds = max(time.perf_counter() - self._start_time, 0.0)
        eta_raw = _throughput_eta_seconds(
            done=self._done_cases,
            total=self.total_cases,
            elapsed_seconds=elapsed_seconds,
        )

        self._bar.update(1)
        should_refresh = (
            self._done_cases % self.update_every_cases == 0
            or self._done_cases == self.total_cases
        )
        if should_refresh:
            self._refresh_postfix(
                elapsed_seconds=elapsed_seconds,
                eta_raw=eta_raw,
            )

    def set_status(self, message: str) -> None:
        """
        完了件数 0 の段階向けステータス表示（ワーカー起動待ちなど）。
        """
        if self._bar is None:
            raise RuntimeError("SweepProgressTracker must be used as a context manager.")
        self._bar.set_postfix(
            ordered_dict=OrderedDict([("status", message)]),
            refresh=True,
        )

    def _time_left_label(self, *, eta_raw: float) -> str:
        if self._done_cases < self._eta_min_cases_effective:
            return "estimating"
        self._eta_display_ema, eta_smoothed = _smooth_eta_display(
            eta_raw,
            eta_display_ema=self._eta_display_ema,
            eta_alpha=self.eta_alpha,
        )
        return _format_eta_duration(
            eta_smoothed,
            round_minutes_above=self.eta_round_minutes_above,
        )

    def _refresh_postfix(
        self,
        *,
        elapsed_seconds: float,
        eta_raw: float,
    ) -> None:
        if self._bar is None:
            return
        self._bar.set_postfix(
            ordered_dict=OrderedDict(
                [
                    ("progress", f"{self._done_cases}/{self.total_cases}"),
                    ("time_passed", _format_duration(elapsed_seconds)),
                    ("time_left", self._time_left_label(eta_raw=eta_raw)),
                ]
            ),
            refresh=True,
        )

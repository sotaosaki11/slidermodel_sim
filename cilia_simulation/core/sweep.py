"""
パラメータ掃引用の並列実行ユーティリティ / Parallel sweep helpers.

ProcessPoolExecutor によるケース並列と SweepProgressTracker 連携を共通化する。
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from typing import TypeVar

from core.progress import SweepProgressTracker

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def default_worker_count(*, reserve: int = 1) -> int:
    """
    並列ワーカー数のデフォルト値を返す。

    Parameters
    ----------
    reserve : int
        OS / IDE 用に空けておく論理 CPU 数（デフォルト 1）。
    """
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - reserve)


def run_parallel_sweep(
    *,
    cases: Sequence[T],
    worker_fn: Callable[[T], R],
    workers: int,
    progress: SweepProgressTracker,
    on_result: Callable[[R], None],
    status_message: str | None = None,
) -> None:
    """
    完了順に worker_fn(case) の結果を on_result へ渡す。

    executor.map(chunksize>1) は入力順でしか返せず先頭チャンク待ちになるため、
    wait(FIRST_COMPLETED) で完了順に処理する。
    """
    total_cases = len(cases)
    if total_cases == 0:
        return

    max_in_flight = max(workers * 2, workers)
    case_iter = iter(cases)
    pending: set[Future[R]] = set()

    def submit_next(executor: ProcessPoolExecutor) -> None:
        case = next(case_iter, None)
        if case is not None:
            pending.add(executor.submit(worker_fn, case))

    if status_message is not None:
        progress.set_status(status_message)
    logger.info(
        "Spawning ProcessPoolExecutor with %d workers (%d cases)",
        workers,
        total_cases,
    )

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for _ in range(min(max_in_flight, total_cases)):
            submit_next(executor)

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.remove(future)
                on_result(future.result())
                progress.update(case_seconds=0.0)
                submit_next(executor)


def sweep_with_progress(
    *,
    cases: Sequence[T],
    worker_fn: Callable[[T], R],
    workers: int,
    on_result: Callable[[R], None],
    desc: str = "sweep",
    alpha: float = 0.2,
    progress_kwargs: dict | None = None,
) -> None:
    """
    ケース列を直列または並列で実行し、各結果を on_result に渡す。

    workers <= 1 なら直列（ケース時間を progress に記録）。
    workers >= 2 なら ProcessPoolExecutor で並列。
    """
    total_cases = len(cases)
    tracker_kwargs = {"alpha": alpha, "desc": desc}
    if progress_kwargs:
        tracker_kwargs.update(progress_kwargs)

    with SweepProgressTracker(total_cases=total_cases, **tracker_kwargs) as progress:
        if workers <= 1:
            for case in cases:
                case_start = time.perf_counter()
                on_result(worker_fn(case))
                progress.update(case_seconds=time.perf_counter() - case_start)
            return

        status = f"starting {workers} workers..."
        run_parallel_sweep(
            cases=cases,
            worker_fn=worker_fn,
            workers=workers,
            progress=progress,
            on_result=on_result,
            status_message=status,
        )

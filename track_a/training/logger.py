"""Run logger: TensorBoard scalars with a graceful no-op fallback.

Every loss component, the recall/hallucination stats, LR, and tokens seen are
logged as separate scalars (PLAN 7.3) so ablations are interpretable.
"""

from __future__ import annotations

from pathlib import Path


class RunLogger:
    """TensorBoard scalar logger; degrades to a no-op if unavailable."""

    def __init__(self, log_dir: str):
        self._writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=log_dir)
        except Exception:  # tensorboard optional at runtime
            self._writer = None

    @property
    def enabled(self) -> bool:
        return self._writer is not None

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

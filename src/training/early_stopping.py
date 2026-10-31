"""Early stopping on a validation ranking metric."""

from __future__ import annotations

from dataclasses import dataclass

from ..utils.logging import get_logger

__all__ = ["EarlyStopping", "ValidationMonitor"]

logger = get_logger(__name__)


@dataclass
class EarlyStopping:
    """Stops when the monitored metric has not improved for ``patience`` epochs.

    Monitors a ranking metric rather than validation loss: BPR loss on sampled
    negatives keeps improving well after ranking quality has plateaued, so loss
    is the wrong stopping signal for this objective.
    """

    patience: int = 10
    min_delta: float = 1e-4
    mode: str = "max"
    best: float | None = None
    best_epoch: int = -1
    counter: int = 0
    should_stop: bool = False

    def step(self, value: float, epoch: int) -> bool:
        """Record an epoch's metric. Returns True if this was the best so far."""
        if self.best is None:
            self.best, self.best_epoch, self.counter = value, epoch, 0
            return True

        improved = (value > self.best + self.min_delta) if self.mode == "max" \
            else (value < self.best - self.min_delta)
        if improved:
            self.best, self.best_epoch, self.counter = value, epoch, 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
            logger.info("Early stopping at epoch %d; best %.6f at epoch %d.",
                        epoch, self.best, self.best_epoch)
        return False


class ValidationMonitor:
    """Early stopping plus a snapshot of the best weights.

    Wraps :class:`EarlyStopping` for models whose parameters are plain tensors
    (the baselines). ``update`` is called once per validation measurement;
    ``restore`` puts the best snapshot back, so the model that is evaluated is
    the one that validated best, not the one that happened to be current when
    training ended.
    """

    def __init__(self, tensors, patience: int = 8, min_delta: float = 1e-4) -> None:
        self.tensors = list(tensors)
        self.stopper = EarlyStopping(patience=patience, min_delta=min_delta)
        self.snapshot = None
        self.best_val: float | None = None
        self.best_epoch: int | None = None

    def update(self, value: float, epoch: int) -> bool:
        """Record a measurement. Returns True when training should stop."""
        if self.stopper.step(value, epoch):
            self.best_val, self.best_epoch = value, epoch
            self.snapshot = [t.detach().clone() for t in self.tensors]
        return self.stopper.should_stop

    def restore(self) -> None:
        if self.snapshot is None:
            return
        for tensor, saved in zip(self.tensors, self.snapshot):
            tensor.data.copy_(saved)

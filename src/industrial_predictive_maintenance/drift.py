"""Deliberate changes to replayed real signals for testing drift detection.

The injector never creates data. It only transforms real recordings in
fixed and repeatable ways without random numbers. Every changed message is
marked, so it can never be mistaken for a real measurement.

Kinds of drift:
    gain    the sensor sensitivity changes: samples are multiplied by a factor
    offset  a constant bias appears: samples are shifted by a share of their RMS
    clip    the sensor range shrinks: samples are cut off below their real peak
    stuck   the sensor freezes: the last real recording is sent again and again
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

KINDS = ("gain", "offset", "clip", "stuck")


@dataclass
class DriftInjector:
    kind: str
    start: int  # first message number (counted from 1) that is changed
    strength: float = 0.5
    ramp: int = 1  # number of messages over which the change grows to full strength
    _frozen: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, not {self.kind!r}")
        if self.start < 1 or self.ramp < 1:
            raise ValueError("start and ramp must be at least 1")
        if self.strength <= 0:
            raise ValueError("strength must be positive")
        if self.kind == "clip" and self.strength >= 1:
            raise ValueError("clip strength must be below 1, otherwise the signal disappears")

    def apply(self, sequence: int, samples: np.ndarray) -> tuple[np.ndarray, dict | None]:
        """Return the samples to send and a description of the change, or None if unchanged."""
        if sequence < self.start:
            self._frozen = samples.copy()
            return samples, None

        progress = min(1.0, (sequence - self.start + 1) / self.ramp)
        amount = self.strength * progress
        if self.kind == "gain":
            changed = samples * (1 + amount)
        elif self.kind == "offset":
            changed = samples + amount * np.sqrt(np.mean(np.square(samples), axis=0))
        elif self.kind == "clip":
            limit = np.abs(samples).max(axis=0) * (1 - amount)
            changed = np.clip(samples, -limit, limit)
        else:
            if self._frozen is None:
                self._frozen = samples.copy()
            changed = self._frozen

        info = {
            "kind": self.kind,
            "strength": self.strength,
            "progress": round(progress, 4),
            "since_sequence": self.start,
        }
        return changed.astype(np.float32), info
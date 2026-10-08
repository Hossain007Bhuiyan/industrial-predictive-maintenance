"""Schema for the vibration messages sent over MQTT.

Every message is checked before its samples are used. A message is rejected
when any of these is true:
    a field is missing or unknown
    a value is out of range
    the shape does not match the channel names
    the samples cannot be decoded or do not match the shape
    the samples contain NaN or infinite values
"""

from __future__ import annotations

import base64
import binascii
from datetime import datetime
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

MIN_SAMPLES = 256


class InjectedDrift(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["gain", "offset", "clip", "stuck"]
    strength: float = Field(gt=0)
    progress: float = Field(gt=0, le=1)
    since_sequence: int = Field(ge=1)


class VibrationMessage(BaseModel):
    # Unknown fields are rejected, so ground truth such as a label can never slip in.
    model_config = ConfigDict(extra="forbid")

    source: Literal["femto", "sca"]
    asset: str = Field(min_length=1)
    placement: str | None = None
    sensor: Literal["accelerometer"]
    channels: list[str] = Field(min_length=1)
    unit: Literal["g", "m/s^2"]
    sampling_hz: float = Field(gt=0)
    rpm: float = Field(ge=0)
    elapsed_s: float | None = Field(default=None, ge=0)
    recorded_at: datetime | None = None
    shape: list[int] = Field(min_length=1, max_length=2)
    samples: str
    injected: InjectedDrift | None
    sequence: int = Field(ge=1)
    sent_at: datetime

    _signal: np.ndarray = PrivateAttr()

    @model_validator(mode="after")
    def check_and_decode(self) -> VibrationMessage:
        if self.elapsed_s is None and self.recorded_at is None:
            raise ValueError("either elapsed_s or recorded_at is required")
        channels = self.shape[1] if len(self.shape) == 2 else 1
        if channels != len(self.channels):
            raise ValueError(f"shape has {channels} channels but {len(self.channels)} channel names are given")
        if self.shape[0] < MIN_SAMPLES:
            raise ValueError(f"at least {MIN_SAMPLES} samples per channel are required")
        try:
            raw = base64.b64decode(self.samples, validate=True)
        except binascii.Error as error:
            raise ValueError("samples are not valid base64") from error
        if len(raw) != 4 * int(np.prod(self.shape)):
            raise ValueError("the number of samples does not match the shape")
        signal = np.frombuffer(raw, dtype="<f4").reshape(self.shape)
        if not np.isfinite(signal).all():
            raise ValueError("samples contain NaN or infinite values")
        self._signal = signal
        return self

    @property
    def signal(self) -> np.ndarray:
        """The decoded samples, shape (samples,) or (samples, channels)."""
        return self._signal


def parse(payload: bytes | str) -> VibrationMessage:
    return VibrationMessage.model_validate_json(payload)
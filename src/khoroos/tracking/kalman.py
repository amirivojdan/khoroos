"""A minimal constant-velocity Kalman filter for bounding boxes.

Written in-package rather than pulled from ``filterpy``/``norfair``: those pin
``numpy<2``, which is incompatible with the modern torch stack this project needs.
The filter below is the standard SORT formulation and depends only on numpy.

State vector: ``[cx, cy, w, h, vcx, vcy, vw]`` — centre, size, and the velocities that
matter for birds moving in a fixed camera frame.
"""

from __future__ import annotations

import numpy as np

STATE_DIM = 7
MEAS_DIM = 4


def box_to_measurement(box: np.ndarray) -> np.ndarray:
    """Convert ``xyxy`` to the ``[cx, cy, w, h]`` measurement space."""
    x1, y1, x2, y2 = box
    return np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2), x2 - x1, y2 - y1], dtype=np.float64)


def measurement_to_box(z: np.ndarray) -> np.ndarray:
    """Convert ``[cx, cy, w, h]`` back to ``xyxy``, guarding against collapsed boxes."""
    cx, cy, w, h = z[:4]
    w = max(float(w), 1.0)
    h = max(float(h), 1.0)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


class BoxKalmanFilter:
    """Constant-velocity Kalman filter over ``[cx, cy, w, h]``."""

    def __init__(
        self,
        box: np.ndarray,
        process_noise: float = 0.28,
        measurement_noise: float = 2.0,
    ) -> None:
        self.x = np.zeros(STATE_DIM, dtype=np.float64)
        self.x[:MEAS_DIM] = box_to_measurement(box)

        # State transition: position += velocity, size += size-velocity.
        self.F = np.eye(STATE_DIM)
        self.F[0, 4] = 1.0  # cx += vcx
        self.F[1, 5] = 1.0  # cy += vcy
        self.F[2, 6] = 1.0  # w  += vw

        # Measurement matrix: we observe centre and size directly.
        self.H = np.zeros((MEAS_DIM, STATE_DIM))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        self.P = np.eye(STATE_DIM) * 10.0
        self.P[4:, 4:] *= 100.0  # velocities start highly uncertain

        self.Q = np.eye(STATE_DIM) * process_noise
        self.Q[4:, 4:] *= 0.01  # velocities evolve slowly

        self.R = np.eye(MEAS_DIM) * measurement_noise

    def predict(self) -> np.ndarray:
        """Advance the state one frame and return the predicted box."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # Keep sizes physically meaningful.
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)
        return self.box

    def update(self, box: np.ndarray) -> None:
        """Correct the state with an observed box."""
        z = box_to_measurement(box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        identity = np.eye(STATE_DIM)
        self.P = (identity - K @ self.H) @ self.P

    @property
    def box(self) -> np.ndarray:
        return measurement_to_box(self.x)

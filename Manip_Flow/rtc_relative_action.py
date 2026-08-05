from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation

from Manip_Flow.common.pose_util import mat_to_rot6d, rot6d_to_mat


class RTCActionShapeError(ValueError):
    pass


class RTCInputs(NamedTuple):
    prefix: np.ndarray | None
    inference_delay: int
    current_bases: np.ndarray


class RTCInferenceState:
    def __init__(self, action_fps: float, target_fps: float = 30.0) -> None:
        if action_fps <= 0.0 or target_fps <= 0.0:
            raise RTCActionShapeError(
                f"RTC frequencies must be positive, got {action_fps}, "
                f"{target_fps}"
            )
        self._action_fps = action_fps
        self._target_fps = target_fps
        self._previous_action: np.ndarray | None = None
        self._previous_bases: np.ndarray | None = None
        self._previous_start: int | None = None
        self._last_latency_s = 0.0

    def prepare(
        self,
        env_obs: dict[str, np.ndarray],
        start: int,
    ) -> RTCInputs:
        current_bases = eef_pose_matrices(env_obs)
        if (
            self._previous_action is None
            or self._previous_bases is None
            or self._previous_start is None
            or start <= self._previous_start
        ):
            return RTCInputs(None, 0, current_bases)
        shift_float = (
            (start - self._previous_start)
            * self._action_fps
            / self._target_fps
        )
        shift_tokens = int(round(shift_float))
        prefix = reanchor_relative_action_prefix(
            self._previous_action,
            self._previous_bases,
            current_bases,
            shift_tokens=shift_tokens,
        )
        inference_delay = int(
            math.ceil(self._last_latency_s * self._action_fps)
        )
        return RTCInputs(prefix, inference_delay, current_bases)

    def complete(
        self,
        action: np.ndarray,
        start: int,
        current_bases: np.ndarray,
        latency_s: float,
    ) -> None:
        self._previous_action = np.asarray(action, dtype=np.float32).copy()
        self._previous_bases = np.asarray(
            current_bases,
            dtype=np.float64,
        ).copy()
        self._previous_start = int(start)
        self._last_latency_s = max(0.0, float(latency_s))


def eef_pose_matrices(env_obs: dict[str, np.ndarray]) -> np.ndarray:
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    for arm in range(2):
        position = np.asarray(env_obs[f"robot{arm}_eef_pos"])[-1]
        rotvec = np.asarray(
            env_obs[f"robot{arm}_eef_rot_axis_angle"]
        )[-1]
        matrices[arm, :3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
        matrices[arm, :3, 3] = position
    return matrices


def reanchor_relative_action_prefix(
    previous_action: np.ndarray,
    previous_bases: np.ndarray,
    current_bases: np.ndarray,
    *,
    shift_tokens: int,
) -> np.ndarray:
    action = np.asarray(previous_action, dtype=np.float64)
    old_bases = np.asarray(previous_bases, dtype=np.float64)
    new_bases = np.asarray(current_bases, dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != 20:
        raise RTCActionShapeError(
            f"previous_action must be (T,20), got {action.shape}"
        )
    if old_bases.shape != (2, 4, 4) or new_bases.shape != (2, 4, 4):
        raise RTCActionShapeError(
            "previous_bases and current_bases must both be (2,4,4)"
        )
    if shift_tokens < 0:
        raise RTCActionShapeError(
            f"shift_tokens must be non-negative, got {shift_tokens}"
        )
    leftovers = action[shift_tokens:].copy()
    for arm in range(2):
        offset = arm * 10
        relative = np.repeat(
            np.eye(4, dtype=np.float64)[None],
            leftovers.shape[0],
            axis=0,
        )
        relative[:, :3, :3] = rot6d_to_mat(
            leftovers[:, offset + 3 : offset + 9]
        )
        relative[:, :3, 3] = leftovers[:, offset : offset + 3]
        world = np.einsum("ij,tjk->tik", old_bases[arm], relative)
        rebased = np.einsum(
            "ij,tjk->tik",
            np.linalg.inv(new_bases[arm]),
            world,
        )
        leftovers[:, offset : offset + 3] = rebased[:, :3, 3]
        leftovers[:, offset + 3 : offset + 9] = mat_to_rot6d(
            rebased[:, :3, :3]
        )
    return leftovers.astype(np.float32)

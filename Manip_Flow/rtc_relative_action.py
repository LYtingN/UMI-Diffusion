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


class _RTCChunk(NamedTuple):
    action: np.ndarray
    bases: np.ndarray
    start: int
    latency_s: float


class RTCInferenceState:
    def __init__(self, action_fps: float, target_fps: float = 30.0) -> None:
        if action_fps <= 0.0 or target_fps <= 0.0:
            raise RTCActionShapeError(
                f"RTC frequencies must be positive, got {action_fps}, "
                f"{target_fps}"
            )
        self._action_fps = action_fps
        self._target_fps = target_fps
        self._latest: _RTCChunk | None = None
        self._anchor: _RTCChunk | None = None

    def prepare(
        self,
        env_obs: dict[str, np.ndarray],
        start: int,
    ) -> RTCInputs:
        current_bases = eef_pose_matrices(env_obs)
        latest = self._latest
        if latest is not None and start < latest.start:
            self._latest = None
            self._anchor = None
            latest = None
        if latest is None:
            return RTCInputs(None, 0, current_bases)
        if start > latest.start:
            self._anchor = latest
        anchor = self._anchor
        if anchor is None:
            return RTCInputs(None, 0, current_bases)
        shift_float = (
            (start - anchor.start)
            * self._action_fps
            / self._target_fps
        )
        shift_tokens = int(round(shift_float))
        prefix = reanchor_relative_action_prefix(
            anchor.action,
            anchor.bases,
            current_bases,
            shift_tokens=shift_tokens,
        )
        inference_delay = int(
            math.ceil(anchor.latency_s * self._action_fps)
        )
        return RTCInputs(prefix, inference_delay, current_bases)

    def complete(
        self,
        action: np.ndarray,
        start: int,
        current_bases: np.ndarray,
        latency_s: float,
    ) -> None:
        self._latest = _RTCChunk(
            action=np.asarray(action, dtype=np.float32).copy(),
            bases=np.asarray(current_bases, dtype=np.float64).copy(),
            start=int(start),
            latency_s=max(0.0, float(latency_s)),
        )


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

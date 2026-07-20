"""P5 glue: flow policy checkpoint -> ``dp_infer_fn`` for the Prior_Recon bridge.

Loads a workspace checkpoint (UMI BaseWorkspace payload format: ``{'cfg',
'state_dicts': {'model', 'ema_model', ...}, 'pickles'}``, saved with dill),
rebuilds the policy (EMA weights by default), and exposes exactly the callback
``bridge/dp_base_anchor.DPKeypointProvider`` expects:

    dp_infer_fn(start, seg_len) -> (Ta, 20) float RELATIVE bimanual action

IMPORTANT — what stays relative and what becomes world:
  * The policy is trained with ``action_pose_repr: relative`` (rel to the
    last-obs EE pose). This wrapper returns that RAW relative action. It must
    NOT be run through UMI's ``get_real_umi_action`` (which bakes in the
    obs-frame base): the bridge's dp_adapter does ``world = base @ rel`` itself,
    with ``base`` supplied by P2 from executed robot state.
  * CONSISTENCY INVARIANT: the EE poses in ``env_obs`` must come from the same
    FK chain P2 uses (HandPoseFK raw wrist frames on executed qpos, world
    frame), so that "relative to the last obs frame" (training convention) and
    "relative to P2's base" (bridge convention) are the same transform. Feed
    ``robot{i}_eef_pos`` / ``robot{i}_eef_rot_axis_angle`` from those frames.
  * Both hands are expressed in ONE world frame (G1 FK), so the inter-robot
    transform ``tx_robot1_robot0`` defaults to identity (UMI needs it because
    its two arms have separate bases).

``env_obs`` format (what ``obs_provider()`` must return): dict of numpy arrays
already at the policy's obs cadence, matching eval_real conventions:
    camera{i}_rgb:                (img_obs_horizon, H, W, 3) uint8 or float
    robot{i}_eef_pos:             (low_dim_obs_horizon, 3) world
    robot{i}_eef_rot_axis_angle:  (low_dim_obs_horizon, 3) world, axis-angle
    robot{i}_gripper_width:       (low_dim_obs_horizon, 1)
Last row = most recent frame. Relative conversion (obs_pose_repr) and the
wrt-other-robot features are computed here via UMI's get_real_umi_obs_dict.

Requires torch; the sys.path bootstrap below puts the repo root on the path so
the vendored ``pipeline.Manip_Flow...`` modules import (the
``universal_manipulation_interface`` repo is no longer needed).
"""

from __future__ import annotations

import pathlib
import sys
from typing import Callable, Dict, Optional

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np


class FlowPolicyInference:
    """Checkpoint -> callable policy returning raw relative (Ta, 20) actions."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        use_ema: bool = True,
        num_inference_steps: Optional[int] = None,
        tx_robot1_robot0: Optional[np.ndarray] = None,
    ) -> None:
        import dill
        import hydra
        import torch
        from omegaconf import OmegaConf

        OmegaConf.register_new_resolver("eval", eval, replace=True)

        payload = torch.load(
            open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill
        )
        cfg = payload["cfg"]

        policy = hydra.utils.instantiate(cfg.policy)
        key = "ema_model" if (use_ema and "ema_model" in payload["state_dicts"]) else "model"
        policy.load_state_dict(payload["state_dicts"][key])
        policy.eval()
        self.policy = policy.to(device)
        self.device = device

        if num_inference_steps is not None:
            self.policy.num_inference_steps = int(num_inference_steps)

        self.shape_meta = OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
        self.obs_pose_repr = str(cfg.task.pose_repr.obs_pose_repr)
        self.action_pose_repr = str(cfg.task.pose_repr.action_pose_repr)
        assert self.action_pose_repr == "relative", (
            "dp_adapter assumes relative actions (world = base @ rel); got "
            f"action_pose_repr={self.action_pose_repr!r}"
        )
        # Both G1 hands live in one world frame -> identity inter-robot tx.
        self.tx_robot1_robot0 = (
            np.eye(4) if tx_robot1_robot0 is None else np.asarray(tx_robot1_robot0)
        )
        self.action_horizon = int(self.shape_meta["action"]["horizon"])
        self.action_dim = int(self.shape_meta["action"]["shape"][0])

    def frames_at_target_fps(self, dp_fps: float, target_fps: float = 30.0) -> int:
        """Planner frames one chunk yields after dp_adapter resampling."""
        if abs(dp_fps - target_fps) < 1e-6:
            return self.action_horizon
        return int(np.floor((self.action_horizon - 1) / dp_fps * target_fps)) + 1

    def assert_planner_budget(
        self,
        dp_fps: float,
        seg_len: int,
        kp_window_len: int,
        target_fps: float = 30.0,
        *,
        replan_stride: int,
        history_len: int = 2,
    ) -> None:
        """Fail fast if the chunk cannot fill the planner's EE-condition window.

        Call at integration time with the planner segment/window lengths and the
        actual replan stride. The effective provider window must cover the segment
        condition plus every preview that will execute before the next replan.
        """
        if history_len < 1:
            raise ValueError(f"history_len must be >= 1, got {history_len}")
        if replan_stride < 1:
            raise ValueError(f"replan_stride must be >= 1, got {replan_stride}")
        n = self.frames_at_target_fps(dp_fps, target_fps) + history_len - 1
        lookahead_len = max(int(kp_window_len) - int(seg_len), 0)
        required = max(
            int(seg_len), int(replan_stride) + history_len + lookahead_len
        )
        if n < required:
            raise ValueError(
                f"action chunk = {n} frames @ {target_fps:g} fps < planner "
                f"requirement {required} for seg_len={seg_len}, stride="
                f"{replan_stride}, history={history_len}, lookahead="
                f"{lookahead_len}; retrain with a longer action horizon."
            )
        if n < int(kp_window_len):
            import warnings

            warnings.warn(
                f"action chunk = {n} frames < kp_window_len {kp_window_len}: "
                f"trailing primitives' lookahead previews will be "
                f"look_valid-truncated (safe through stride {replan_stride}; "
                f"the truncated primitives are never executed).",
                RuntimeWarning,
                stacklevel=2,
            )

    def predict_relative_action(self, env_obs: Dict[str, np.ndarray]) -> np.ndarray:
        """env_obs (see module docstring) -> RAW relative action (Ta, 20)."""
        import torch
        from pipeline.Manip_Flow.common.pytorch_util import dict_apply
        from pipeline.Manip_Flow.common.real_inference_util import get_real_umi_obs_dict

        obs_dict_np = get_real_umi_obs_dict(
            env_obs=env_obs,
            shape_meta=self.shape_meta,
            obs_pose_repr=self.obs_pose_repr,
            tx_robot1_robot0=self.tx_robot1_robot0,
        )
        obs_dict = dict_apply(
            obs_dict_np,
            lambda x: torch.from_numpy(np.ascontiguousarray(x))
            .unsqueeze(0)
            .to(self.device, dtype=self.policy.dtype),
        )
        with torch.no_grad():
            result = self.policy.predict_action(obs_dict, None)
        action = result["action"][0].detach().to("cpu").numpy()
        assert action.shape == (self.action_horizon, self.action_dim)
        return action

    def make_dp_infer_fn(
        self, obs_provider: Callable[[], Dict[str, np.ndarray]]
    ) -> Callable[[int, int], np.ndarray]:
        """Build the ``dp_infer_fn(start, seg_len) -> (Ta, 20)`` P4/P2 hook.

        ``obs_provider()`` returns the CURRENT env_obs; it is sampled at call
        time, i.e. once per planned segment. ``start``/``seg_len`` index the
        planner's 30 fps timeline; the policy horizon is fixed, so they are
        accepted (for the DPKeypointProvider signature) but the chunk length is
        always ``action_horizon`` — dp_adapter resamples it to 30 fps and the
        planner's lookahead logic tolerates a short window (partially invalid
        preview), consistent with kp_window_len semantics.
        """

        def dp_infer_fn(start: int, seg_len: int) -> np.ndarray:  # noqa: ARG001
            return self.predict_relative_action(obs_provider())

        return dp_infer_fn

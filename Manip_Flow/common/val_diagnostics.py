"""Val-time metrics that the flow loss structurally cannot report.

``val_loss`` is ``E[|| v(xt,t,obs) - (x1-x0) ||^2]`` with ``t`` uniform, averaged
over every ``action_horizon x action_dim`` entry. Two consequences:

* it scores the conditional MEAN velocity field, while deploy executes a single
  ``num_inference_steps`` Euler draw -- sampling spread is invisible to it;
* the chunk-end position channels are a few 800ths of the number, so a 0.4 m
  swing in the predicted hand height barely moves it.

The metrics here measure that spread directly, in metres. Kept out of the
workspace module so they import without ``wandb``/``accelerate``.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from Manip_Flow.common.pytorch_util import dict_apply

# The flat action_dim splits into arms x [pos(3), rot_6d(6), gripper_width(1)],
# matching FlowTimmPolicy's (B, Ta, 20) output for the bimanual shape_meta.
ACTION_CHANNELS_PER_ARM = 10
_WIDTH_CHANNEL = 9


def log_draw_dispersion(
    step_log: Dict[str, Any],
    policy: Any,
    obs_dict: Dict[str, torch.Tensor],
    n_obs: int,
    k: int,
) -> None:
    """Spread of ``k`` independent draws from ONE observation.

    ``action_pose_repr='relative'`` means every draw for a given observation
    shares one base pose, so the std below is directly in metres of predicted
    end-effector displacement -- no frame algebra needed.
    """
    obs = dict_apply(obs_dict, lambda x: x[:n_obs].repeat_interleave(k, dim=0))
    pred = policy.predict_action(obs, None)["action_pred"]
    # (n_obs, k, Ta, arms, [pos3, rot6, width1])
    pred = pred.reshape(n_obs, k, pred.shape[1], -1, ACTION_CHANNELS_PER_ARM)
    std = pred.std(dim=1)
    pos = std[..., :3].norm(dim=-1)  # metres, per (n_obs, Ta, arm)
    step_log["val_draw_std_pos_mean_m"] = pos.mean().item()
    step_log["val_draw_std_pos_last_m"] = pos[:, -1].mean().item()
    step_log["val_draw_std_pos_max_m"] = pos.max().item()
    step_log["val_draw_std_width_last_m"] = std[:, -1, :, _WIDTH_CHANNEL].mean().item()


def log_prefix_consistency(
    step_log: Dict[str, Any],
    policy: Any,
    obs_dict: Dict[str, torch.Tensor],
    n_obs: int,
    prefix_steps: int,
) -> None:
    """Does a fresh draw agree with a chunk it was already committed to?

    Deploy commits ``stride`` frames and then re-predicts under RTC soft-prefix
    guidance. Both draws here use the same observation, hence the same base pose,
    so the chunks are comparable with no frame algebra: the divergence in the
    UNGUIDED tail is the offline counterpart of the chunk-end swing seen on the
    robot. The guided-prefix residual is reported separately so that "RTC did not
    bite" and "the model is inconsistent about the future" stay distinguishable --
    they need different fixes.

    This does NOT test observation staleness. Comparing chunks anchored at ``t``
    and ``t+dt`` needs the SE(3) map between their two base poses
    (``action_pose_repr='relative'``), which is not done here.
    """
    obs = dict_apply(obs_dict, lambda x: x[:n_obs])
    first = policy.predict_action(obs, None)["action_pred"]
    horizon = first.shape[1]
    # Leave at least one unguided step: an empty tail slice reports nan, and a nan
    # in step_log would propagate into topk's metric dict.
    prefix_steps = max(1, min(int(prefix_steps), horizon - 1))
    second = policy.predict_action(
        obs,
        rtc_action_prefix=first[:, :prefix_steps],
        rtc_inference_delay=0,
    )["action_pred"]
    delta = (second - first).reshape(n_obs, horizon, -1, ACTION_CHANNELS_PER_ARM)
    pos = delta[..., :3].norm(dim=-1)
    step_log["val_prefix_guided_pos_m"] = pos[:, :prefix_steps].mean().item()
    step_log["val_prefix_tail_pos_mean_m"] = pos[:, prefix_steps:].mean().item()
    step_log["val_prefix_tail_pos_max_m"] = pos[:, prefix_steps:].max().item()
    step_log["val_prefix_tail_pos_last_m"] = pos[:, -1].mean().item()


def shuffled_obs_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Pair each observation with another sample's action.

    The flow target ``x1-x0`` has an irreducible conditional-variance floor, so
    ``val_loss`` carries no scale on its own. This is the "conditioning is
    uninformative" reference: ``val_loss`` rising toward it means the visual
    conditioning has stopped paying. Rolling by one keeps the marginal action
    distribution intact, so only the pairing is destroyed.
    """
    return {"obs": batch["obs"], "action": batch["action"].roll(1, dims=0)}

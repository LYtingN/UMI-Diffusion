"""Smoke test: flow policy + bridge adapter + (optionally) the real planner.

Run ON THE TRAINING BOX (needs torch/timm; local env has no torch):
    python pipeline/Manip_Flow/scripts/smoke_flow_policy.py
    python pipeline/Manip_Flow/scripts/smoke_flow_policy.py \
        --ckpt /path/to/delta87_look.ckpt      # adds the true end-to-end leg

Checks, with a tiny untrained encoder and random data:
  1. compute_loss forward + backward (training path the workspace exercises)
  2. predict_action returns (B, 12, 20) with the DiT backbone (12 tokens @
     10 Hz span 1.10 s, the config default); UNet accepts 12 (%4==0) but must
     reject 14 -- both checked
  3. prefix inpainting pins the conditioned steps exactly
  4. frame budget: 12 actions @ 10 Hz resample to 34 frames @ 30 fps;
     2 executed history frames replace the duplicate current frame -> 35
     provider frames (the "execute 16, replan"
     loop: executed prims 0-1 get FULL 16-frame lookahead previews,
     prims 2-3 previews are look_valid-truncated and never executed)
  5. (Ta, 20) chunk -> dp_adapter -> (T, 2, 7) keypoints + (T, 2) grippers
  6. DPKeypointProvider rejects a too-short chunk (min_window_len=seg_len)
     and passes an exactly-seg_len one [needs mujoco]
  7. [--ckpt] policy -> DPKeypointProvider -> plan_segment x2 segments with
     next_seed(stride=16) carry: the original receding-horizon streaming
     loop on real planner code
"""

import argparse
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from Manip_Flow.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from Manip_Flow.model.vision.timm_obs_encoder import TimmObsEncoder
from Manip_Flow.policy.flow_timm_policy import FlowTimmPolicy
from pipeline.Deploy.bridge.dp_adapter import dp_action_to_keypoints

IMG_H = 2  # img_obs_horizon
LOW_H = 2  # low_dim_obs_horizon
ACT_H = 12  # action tokens @ 10 Hz (down_sample 3 on 30 fps data) span 1.10 s
ACT_FPS = 10.0
RESAMPLED = 34  # floor((12-1)/10*30)+1 frames @ 30 fps after dp_adapter
PROVIDER_WINDOW = RESAMPLED + LOW_H - 1
SEG_LEN = 34
KP_WINDOW_LEN = 50  # what the bridge requests (seg 34 + lookahead 16)
STRIDE = 16  # replan after executing 16 new frames (prims 0-1)
B = 2


def build_shape_meta(act_h: int = ACT_H) -> dict:
    """umi_bimanual shape_meta with horizons resolved (no omegaconf)."""
    obs = {}
    for i in range(2):
        obs[f"camera{i}_rgb"] = {"shape": [3, 224, 224], "horizon": IMG_H, "type": "rgb"}
        obs[f"robot{i}_eef_pos"] = {"shape": [3], "horizon": LOW_H, "type": "low_dim"}
        obs[f"robot{i}_eef_rot_axis_angle"] = {
            "shape": [6], "horizon": LOW_H, "type": "low_dim",
        }
        obs[f"robot{i}_gripper_width"] = {"shape": [1], "horizon": LOW_H, "type": "low_dim"}
    obs["robot0_eef_pos_wrt1"] = {"shape": [3], "horizon": LOW_H, "type": "low_dim"}
    obs["robot0_eef_rot_axis_angle_wrt1"] = {"shape": [6], "horizon": LOW_H, "type": "low_dim"}
    obs["robot1_eef_pos_wrt0"] = {"shape": [3], "horizon": LOW_H, "type": "low_dim"}
    obs["robot1_eef_rot_axis_angle_wrt0"] = {"shape": [6], "horizon": LOW_H, "type": "low_dim"}
    return {"obs": obs, "action": {"shape": [20], "horizon": act_h}}


def build_policy(
    shape_meta: dict, backbone: str = "dit", obs_encoder=None
) -> FlowTimmPolicy:
    if obs_encoder is None:
        obs_encoder = TimmObsEncoder(
            shape_meta=shape_meta,
            model_name="vit_tiny_patch16_224",
            pretrained=False,
            frozen=False,
            global_pool="",
            feature_aggregation="attention_pool_2d",
            position_encording="sinusoidal",
            downsample_ratio=32,
            transforms=None,
            use_group_norm=True,
            share_rgb_model=False,
            imagenet_norm=True,
        )
    policy = FlowTimmPolicy(
        shape_meta=shape_meta,
        obs_encoder=obs_encoder,
        num_inference_steps=4,
        backbone=backbone,
        dit_d_model=128,
        dit_depth=2,
        dit_n_heads=4,
        # 3 levels -> divisor 2**(3-1)=4, matching the real config's
        # down_dims=[256,512,1024]; the UNet-parity check below (accepts 12,
        # rejects 14) depends on this divisor being 4, not 2.
        down_dims=(64, 128, 256),
        diffusion_step_embed_dim=64,
    )
    normalizer = LinearNormalizer()
    for key in list(shape_meta["obs"].keys()) + ["action"]:
        normalizer[key] = SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(normalizer)
    return policy


def random_obs(shape_meta: dict, device) -> dict:
    return {
        k: torch.randn(B, v["horizon"], *v["shape"], device=device)
        for k, v in shape_meta["obs"].items()
    }


def make_dp_infer_fn(policy, obs, device):
    """(start, window_len) -> (Ta, 20) numpy, like inference.FlowPolicyInference."""

    def dp_infer_fn(start, window_len):  # noqa: ARG001
        with torch.no_grad():
            result = policy.predict_action(obs, None)
        return result["action"][0].cpu().numpy().astype(np.float64)

    return dp_infer_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, help="delta87+lookahead planner ckpt")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shape_meta = build_shape_meta()
    policy = build_policy(shape_meta).to(device)
    obs = random_obs(shape_meta, device)
    batch = {"obs": obs, "action": torch.randn(B, ACT_H, 20, device=device)}

    # 1. training path
    loss = policy(batch)
    loss.backward()
    assert torch.isfinite(loss), f"non-finite loss: {loss}"
    print(f"[1] compute_loss + backward OK (dit, horizon {ACT_H}), loss={loss.item():.4f}")

    # 2. inference path + unet divisibility guard
    policy.eval()
    with torch.no_grad():
        result = policy.predict_action(obs, None)
    action = result["action"]
    assert action.shape == (B, ACT_H, 20), action.shape
    assert torch.isfinite(action).all()
    # unet parity: accepts 12 (%4==0), rejects 14 (skip-concat breaks)
    build_policy(shape_meta, backbone="unet", obs_encoder=policy.obs_encoder)
    try:
        build_policy(build_shape_meta(14), backbone="unet",
                     obs_encoder=policy.obs_encoder)
        raise AssertionError("unet backbone must reject horizon 14 (% 4 != 0)")
    except ValueError:
        pass
    print(f"[2] predict_action OK {tuple(action.shape)}; "
          f"unet accepts {ACT_H} / rejects 14 OK")

    # 3. prefix inpainting
    policy.inpaint_fixed_action_prefix = True
    prefix = batch["action"][:, :4]
    with torch.no_grad():
        result = policy.predict_action(obs, fixed_action_prefix=prefix)
    err = (result["action"][:, :4] - prefix).abs().max().item()
    assert err < 1e-4, f"prefix not pinned, max err {err}"
    policy.inpaint_fixed_action_prefix = False
    print(f"[3] prefix inpainting OK, max err={err:.2e}")

    # 4. frame budget: 34 resampled DP frames become 35 provider frames after
    #    replacing action[0] with the two executed history frames.
    dp_action = result["action"][0].cpu().numpy().astype(np.float64)
    from pipeline.Prior_Recon.Masked_Flow.visual.recon_delta69 import _HAND_OFFSETS

    keypoints, grippers = dp_action_to_keypoints(
        dp_action,
        base_left=np.eye(4),
        base_right=np.eye(4),
        tcp_offsets=np.asarray(_HAND_OFFSETS, dtype=np.float64),
        dp_fps=ACT_FPS,
        target_fps=30.0,
    )
    assert keypoints.shape == (RESAMPLED, 2, 7), keypoints.shape
    assert grippers.shape == (RESAMPLED, 2), grippers.shape
    assert np.isfinite(keypoints).all() and np.isfinite(grippers).all()
    assert PROVIDER_WINDOW >= SEG_LEN and PROVIDER_WINDOW >= STRIDE + LOW_H + 16
    print(f"[4] frame budget OK: {ACT_H} tokens @ {ACT_FPS:g} Hz -> "
          f"{keypoints.shape[0]} frames >= max(seg_len {SEG_LEN}, "
          f"stride {STRIDE} + 18)")
    print(f"[5] dp_adapter chain OK, keypoints={keypoints.shape}, grippers={grippers.shape}")

    # 6. provider frame-budget enforcement (needs mujoco for FK)
    try:
        from pipeline.Deploy.bridge.dp_base_anchor import (
            DPKeypointProvider,
            HandPoseFK,
        )

        fk = HandPoseFK()
        base_qpos = np.zeros(36)
        base_qpos[3] = 1.0  # identity root quat (wxyz)

        # 6a. UMI-legacy-sized chunk (16 @ 20 fps -> 23 frames < seg_len) must raise
        ident = np.zeros((16, 20))
        for off in (3, 13):  # identity rot6d rows [1,0,0],[0,1,0] per arm
            ident[:, off] = 1.0
            ident[:, off + 4] = 1.0
        short_fn = lambda s, k: ident  # noqa: E731
        provider = DPKeypointProvider(
            short_fn,
            fk,
            dp_fps=20.0,
            tcp_offsets=np.asarray(_HAND_OFFSETS, dtype=np.float64),
            min_window_len=SEG_LEN,
        )
        provider.update_base_qpos(np.repeat(base_qpos[None, :], LOW_H, axis=0))
        try:
            provider.kp_window(0, KP_WINDOW_LEN)
            raise AssertionError("short chunk must raise with min_window_len set")
        except ValueError:
            pass

        # 6b. exactly-seg_len chunk passes; short lookahead tail is allowed
        #     (plan_segment look_valid-masks the missing preview frames)
        provider = DPKeypointProvider(
            make_dp_infer_fn(policy, obs, device), fk, dp_fps=ACT_FPS,
            tcp_offsets=np.asarray(_HAND_OFFSETS, dtype=np.float64),
            min_window_len=SEG_LEN,
        )
        provider.update_base_qpos(np.repeat(base_qpos[None, :], LOW_H, axis=0))
        window = provider.kp_window(0, KP_WINDOW_LEN)
        assert window.shape == (PROVIDER_WINDOW, 2, 7), window.shape
        assert provider.last_n_valid == PROVIDER_WINDOW
        assert provider.last_grippers.shape == (PROVIDER_WINDOW, 2)
        print(f"[6] DPKeypointProvider budget enforcement OK "
              f"(short chunk raises; seg_len chunk passes, "
              f"n_valid={provider.last_n_valid} of {KP_WINDOW_LEN} requested)")
    except ImportError as e:
        print(f"[6] SKIPPED (mujoco not available: {e})")
        provider = None

    # 7. true end-to-end against the real planner
    if args.ckpt is not None:
        assert provider is not None, "end-to-end leg needs mujoco"
        from pipeline.Deploy.bridge.online_planner import (
            OnlinePrimitivePlanner,
        )

        planner = OnlinePrimitivePlanner(args.ckpt, device=device)
        assert PROVIDER_WINDOW >= planner.seg_len, (
            f"chunk covers {PROVIDER_WINDOW} frames < planner seg_len "
            f"{planner.seg_len} (zero-pad hazard)"
        )
        assert PROVIDER_WINDOW >= STRIDE + planner.hist_len + planner.lookahead_len, (
            "executed primitives would not get full lookahead previews"
        )
        seed = planner.initial_seed()
        seg0 = planner.plan_segment(
            0, 0, seed, kp_window=provider.kp_window(0, planner.kp_window_len)
        )
        assert seg0.window_len == planner.seg_len, (
            f"segment truncated: {seg0.window_len} < {planner.seg_len} "
            f"(kp window too short reached the planner)"
        )
        # original receding-horizon loop: execute STRIDE new frames, replan
        seed1 = planner.next_seed(seg0, stride=STRIDE)
        start1 = STRIDE
        provider.update_base_qpos(seg0.qpos[STRIDE : STRIDE + planner.hist_len])
        seg1 = planner.plan_segment(
            1, start1, seed1,
            kp_window=provider.kp_window(start1, planner.kp_window_len),
        )
        assert seg1.window_len == planner.seg_len
        assert np.isfinite(seg1.qpos).all()
        print(f"[7] end-to-end OK: 2 segments, replan stride={STRIDE}, "
              f"window_len={seg1.window_len}, lookahead={planner.lookahead_len}")
    else:
        print("[7] SKIPPED (pass --ckpt for the real planner end-to-end leg)")

    print("SMOKE PASS")


if __name__ == "__main__":
    main()

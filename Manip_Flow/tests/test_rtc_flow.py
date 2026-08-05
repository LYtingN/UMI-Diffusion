from __future__ import annotations

import numpy as np
import torch

from Manip_Flow.policy import rtc_flow
from Manip_Flow.policy.rtc_flow import rtc_guided_velocity, rtc_prefix_weights
from Manip_Flow.rtc_relative_action import (
    RTCInferenceState,
    reanchor_relative_action_prefix,
)


def test_rtc_prefix_weights_freeze_delay_and_taper_overlap() -> None:
    # Given: three delayed tokens and a six-token RTC execution horizon.
    # When: exponential prefix attention is constructed for a ten-token chunk.
    weights = rtc_prefix_weights(
        inference_delay=3,
        execution_horizon=6,
        total_horizon=10,
        schedule="exp",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Then: committed actions are frozen and editable overlap decays to zero.
    torch.testing.assert_close(weights[:3], torch.ones(3))
    assert torch.all(weights[3:6] < 1.0)
    assert torch.all(weights[3:6] > 0.0)
    torch.testing.assert_close(weights[6:], torch.zeros(4))


def test_relative_rtc_prefix_is_reanchored_to_current_eef_pose() -> None:
    # Given: an old relative chunk whose world target is x=1.2 metres.
    previous_action = np.zeros((4, 20), dtype=np.float32)
    previous_action[:, 3] = 1.0
    previous_action[:, 7] = 1.0
    previous_action[:, 13] = 1.0
    previous_action[:, 17] = 1.0
    previous_action[:, 0] = 0.2
    previous_action[:, 10] = 0.2
    previous_bases = np.repeat(np.eye(4)[None], 2, axis=0)
    previous_bases[:, 0, 3] = 1.0
    current_bases = previous_bases.copy()
    current_bases[:, 0, 3] = 1.1

    # When: one consumed token is removed and leftovers use the new base.
    prefix = reanchor_relative_action_prefix(
        previous_action,
        previous_bases,
        current_bases,
        shift_tokens=1,
    )

    # Then: the same world target becomes x=0.1 relative to the current EE.
    assert prefix.shape == (3, 20)
    np.testing.assert_allclose(prefix[:, 0], 0.1, atol=1e-6)
    np.testing.assert_allclose(prefix[:, 10], 0.1, atol=1e-6)


def test_rtc_guidance_moves_flow_velocity_toward_previous_chunk() -> None:
    # Given: an unguided zero velocity and a prefix target at one.
    state = torch.zeros((1, 3, 2), dtype=torch.float32)
    prefix = torch.ones_like(state)
    weights = torch.tensor([1.0, 0.5, 0.0]).view(1, 3, 1)

    # When: RTC guides the midpoint of the flow with beta five.
    velocity = rtc_guided_velocity(
        state=state,
        time=torch.tensor(0.5),
        velocity_fn=lambda value: torch.zeros_like(value),
        prefix=prefix,
        weights=weights,
        max_guidance_weight=5.0,
    )

    # Then: guidance follows prefix weights and leaves unconstrained tokens free.
    torch.testing.assert_close(velocity[0, :, 0], torch.tensor([2.0, 1.0, 0.0]))


def test_rtc_state_maps_planner_stride_and_latency_to_policy_tokens() -> None:
    # Given: a 15 Hz action policy driven by a 30 Hz planner.
    state = RTCInferenceState(action_fps=15.0, target_fps=30.0)
    env_obs = {}
    for arm in range(2):
        env_obs[f"robot{arm}_eef_pos"] = np.zeros((1, 3))
        env_obs[f"robot{arm}_eef_rot_axis_angle"] = np.zeros((1, 3))
    action = np.zeros((36, 20), dtype=np.float32)
    action[:, 3] = 1.0
    action[:, 7] = 1.0
    action[:, 13] = 1.0
    action[:, 17] = 1.0
    first = state.prepare(env_obs, start=0)
    state.complete(action, 0, first.current_bases, latency_s=0.21)

    # When: the planner advances sixteen 30 Hz frames.
    second = state.prepare(env_obs, start=16)

    # Then: eight action tokens are consumed and four cover measured latency.
    assert second.prefix is not None
    assert second.prefix.shape == (28, 20)
    assert second.inference_delay == 4


def test_rtc_state_accepts_fractional_planner_to_policy_stride() -> None:
    # Given: a 10 Hz DP chunk consumed for 32 frames by a 30 Hz planner.
    state = RTCInferenceState(action_fps=10.0, target_fps=30.0)
    env_obs = {}
    for arm in range(2):
        env_obs[f"robot{arm}_eef_pos"] = np.zeros((1, 3))
        env_obs[f"robot{arm}_eef_rot_axis_angle"] = np.zeros((1, 3))
    action = np.zeros((40, 20), dtype=np.float32)
    action[:, (3, 7, 13, 17)] = 1.0
    first = state.prepare(env_obs, start=0)
    state.complete(action, 0, first.current_bases, latency_s=0.1)

    # When: the next H8/F32/P2 segment starts after the 32-frame stride.
    second = state.prepare(env_obs, start=32)

    # Then: RTC advances to the nearest policy token without rejecting the stride.
    assert second.prefix is not None
    assert second.prefix.shape == (29, 20)


def test_official_rtc_does_not_hard_latch_tail_channels() -> None:
    # Given: old task progress lies beyond the official soft execution horizon.
    condition = torch.zeros((1, 4, 4), dtype=torch.float32)
    condition_mask = torch.zeros_like(condition, dtype=torch.bool)
    prefix = torch.zeros((1, 3, 4), dtype=torch.float32)
    prefix[0, 2, 0] = 0.75
    prefix[0, 2, 3] = 0.02

    class ZeroVelocity(torch.nn.Module):
        def forward(
            self,
            value: torch.Tensor,
            _time: torch.Tensor,
            local_cond: torch.Tensor | None,
            global_cond: torch.Tensor | None,
        ) -> torch.Tensor:
            del local_cond, global_cond
            return torch.zeros_like(value)

    # When: a new sample is generated with official soft-prefix RTC guidance.
    sampled = rtc_flow.flow_euler_sample(
        model=ZeroVelocity(),
        condition_data=condition,
        condition_mask=condition_mask,
        global_cond=None,
        generator=torch.Generator().manual_seed(7),
        rtc_action_prefix=prefix,
        rtc_inference_delay=0,
        config=rtc_flow.FlowSamplingConfig(
            inference_steps=2,
            time_embed_scale=1.0,
            action_horizon=4,
            execution_horizon=1,
            max_guidance_weight=5.0,
            prefix_schedule="linear",
        ),
    )

    # Then: the tail stays free instead of being overwritten by custom latches.
    assert not torch.isclose(sampled[0, 2, 0], prefix[0, 2, 0])

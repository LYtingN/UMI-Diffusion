from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from Manip_Flow.common import real_inference_util
from Manip_Flow.inference import DeploymentConfigError, FlowPolicyInference
from Manip_Flow.mock_policy_server import MockDPPolicyServer
from Manip_Flow.policy_server import DPPolicyServer, build_parser
from pipeline.Deploy.bridge import dp_wire


def test_flow_inference_forwards_rtc_prefix_and_latency(monkeypatch) -> None:
    # Given: RTC has re-anchored leftovers from the previous relative chunk.
    prefix = np.ones((3, 20), dtype=np.float32)
    bases = np.repeat(np.eye(4)[None], 2, axis=0)

    class FakeRTCState:
        completed_start = None

        def prepare(self, env_obs, start):
            del env_obs, start
            return SimpleNamespace(
                prefix=prefix,
                inference_delay=2,
                current_bases=bases,
            )

        def complete(self, action, start, current_bases, latency_s):
            del action, current_bases, latency_s
            self.completed_start = start

    class FakePolicy:
        dtype = torch.float32
        received_prefix = None
        received_delay = None

        def predict_action(
            self,
            obs_dict,
            fixed_action_prefix=None,
            rtc_action_prefix=None,
            rtc_inference_delay=0,
        ):
            del obs_dict, fixed_action_prefix
            self.received_prefix = rtc_action_prefix
            self.received_delay = rtc_inference_delay
            return {"action": torch.zeros((1, 4, 20))}

    monkeypatch.setattr(
        real_inference_util,
        "get_real_umi_obs_dict",
        lambda **_kwargs: {"observation": np.zeros((1,), dtype=np.float32)},
    )
    inference = FlowPolicyInference.__new__(FlowPolicyInference)
    inference.shape_meta = {}
    inference.obs_pose_repr = "relative"
    inference.tx_robot1_robot0 = np.eye(4)
    inference.device = "cpu"
    inference.policy = FakePolicy()
    inference.action_horizon = 4
    inference.action_dim = 20
    inference._rtc_state = FakeRTCState()

    # When: a new action chunk is inferred at planner frame 32.
    inference.predict_relative_action({}, start=32)

    # Then: the policy receives official RTC inputs and state advances atomically.
    assert inference.policy.received_prefix.shape == (1, 3, 20)
    assert inference.policy.received_delay == 2
    assert inference._rtc_state.completed_start == 32


def test_policy_server_cli_and_inference_enable_rtc_by_default() -> None:
    # Given / When: the deploy server parses its default and explicit opt-out.
    enabled = build_parser().parse_args(["--ckpt", "policy.ckpt"])
    disabled = build_parser().parse_args(["--ckpt", "policy.ckpt", "--no-rtc"])

    # Then: RTC is the normal path and remains explicitly switchable for A/B QA.
    assert enabled.rtc_enabled is True
    assert disabled.rtc_enabled is False
    assert "rtc_enabled" in inspect.signature(DPPolicyServer).parameters
    assert "rtc_enabled" in inspect.signature(FlowPolicyInference).parameters


def test_policy_server_ping_reports_protocol_and_action_frequency() -> None:
    # Given
    server = DPPolicyServer.__new__(DPPolicyServer)
    server.infer = SimpleNamespace(
        shape_meta={"action": {"horizon": 36, "shape": [20]}},
        action_horizon=36,
        action_dim=20,
        obs_pose_repr="relative",
        action_pose_repr="relative",
        action_fps=15.0,
        rtc_enabled=True,
    )

    # When
    message = dp_wire.decode_reply(server._handle(dp_wire.encode_ping()))

    # Then
    assert message["protocol_version"] == dp_wire.DP_PROTOCOL_VERSION
    assert message["action_fps"] == 15.0
    assert message["rtc_enabled"] is True


def test_reset_rtc_replaces_the_carried_chunk_only_when_rtc_is_on() -> None:
    # Given: a live RTC state holding the previous episode's chunk.
    inference = FlowPolicyInference.__new__(FlowPolicyInference)
    inference.rtc_enabled = True
    inference.action_fps = 10.0
    inference.reset_rtc()
    carried = inference._rtc_state

    # When
    inference.reset_rtc()

    # Then: a fresh state replaces it, and the off switch still means no state.
    assert inference._rtc_state is not None
    assert inference._rtc_state is not carried
    inference.rtc_enabled = False
    inference.reset_rtc()
    assert inference._rtc_state is None


def _predict_buf(start: int, episode_token: str) -> bytes:
    return dp_wire.encode_predict_request(
        req_id=1,
        cameras={"camera0_rgb": [b"\xff\xd8jpeg"]},
        lowdim={"robot0_gripper_width": np.zeros((2, 1), dtype=np.float32)},
        start=start,
        window_len=72,
        episode_token=episode_token,
    )


def _episode_token_server() -> DPPolicyServer:
    server = DPPolicyServer.__new__(DPPolicyServer)
    server._episode_token = None
    resets = []
    server.infer = SimpleNamespace(
        reset_rtc=lambda: resets.append(True),
        predict_relative_action=lambda env_obs, start: np.zeros(
            (4, 20), dtype=np.float32
        ),
    )
    server._assemble_env_obs = lambda cameras, lowdim: {}
    server.resets = resets
    return server


def test_policy_server_drops_rtc_state_when_the_episode_token_changes() -> None:
    # Given: a server that outlives the bridge, so `start` alone cannot mark the
    # boundary -- both episodes replan from planner frame 0 onwards.
    server = _episode_token_server()

    # When: two episodes of the same bridge session each stream a few requests.
    for buf in (
        _predict_buf(0, "sess:0:0"),
        _predict_buf(32, "sess:0:0"),
        _predict_buf(0, "sess:1:1"),
        _predict_buf(32, "sess:1:1"),
    ):
        server._handle(buf)

    # Then: RTC state is dropped exactly once per episode, not per request.
    assert len(server.resets) == 2


def test_policy_server_keeps_rtc_state_for_a_tokenless_client() -> None:
    # Given: a pre-protocol-4 bridge that cannot report episode identity.
    server = _episode_token_server()

    # When
    server._handle(_predict_buf(0, ""))
    server._handle(_predict_buf(32, ""))

    # Then: the server never guesses a boundary; RTC's own start-regression guard
    # stays the only fallback.
    assert server.resets == []


def test_mock_policy_server_reports_deploy_action_frequency() -> None:
    # Given
    server = MockDPPolicyServer(action_fps=15.0)

    # When
    message = dp_wire.decode_reply(server._handle(dp_wire.encode_ping()))

    # Then
    assert message["protocol_version"] == dp_wire.DP_PROTOCOL_VERSION
    assert message["action_fps"] == 15.0
    assert message["rtc_enabled"] is False


def test_local_budget_rejects_action_short_after_replan_lead() -> None:
    # Given: startup sees 65 action frames for a 72-frame segment and stride 16.
    inference = FlowPolicyInference.__new__(FlowPolicyInference)
    inference.action_horizon = 65
    inference.action_fps = 30.0

    # When / Then: local mode rejects the chunk before its first planning call.
    with pytest.raises(DeploymentConfigError, match="leaves 49 frames"):
        inference.assert_planner_budget(
            dp_fps=30.0,
            seg_len=72,
            kp_window_len=72,
            replan_stride=16,
            history_len=8,
        )

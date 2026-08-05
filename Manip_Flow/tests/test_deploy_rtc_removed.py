from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from Manip_Flow.inference import FlowPolicyInference
from Manip_Flow.mock_policy_server import MockDPPolicyServer
from Manip_Flow.policy_server import DPPolicyServer, build_parser
from pipeline.Deploy.bridge import dp_wire


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


def test_mock_policy_server_reports_deploy_action_frequency() -> None:
    # Given
    server = MockDPPolicyServer(action_fps=15.0)

    # When
    message = dp_wire.decode_reply(server._handle(dp_wire.encode_ping()))

    # Then
    assert message["protocol_version"] == dp_wire.DP_PROTOCOL_VERSION
    assert message["action_fps"] == 15.0
    assert message["rtc_enabled"] is False

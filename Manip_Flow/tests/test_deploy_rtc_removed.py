from __future__ import annotations

import inspect

import pytest

from Manip_Flow.inference import FlowPolicyInference
from Manip_Flow.policy_server import DPPolicyServer, build_parser


def test_policy_server_cli_and_inference_do_not_expose_rtc() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--ckpt", "policy.ckpt", "--rtc"])

    assert "rtc_enabled" not in inspect.signature(DPPolicyServer).parameters
    assert "rtc_enabled" not in inspect.signature(FlowPolicyInference).parameters

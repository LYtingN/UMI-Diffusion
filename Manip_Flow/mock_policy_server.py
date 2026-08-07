"""Mock DP policy server for end-to-end wiring tests (NO torch/cv2/hydra).

Speaks the exact same ZMQ REQ/REP protocol as ``policy_server.py`` but returns
an IDENTITY relative action chunk: zero translation, identity rot6d, mid gripper.
Fed through the bridge's dp_adapter (world = base @ identity), this makes the DP
"hands hold" -- a deterministic reference for verifying the full decoupled loop
(wire codec, client, provider, planner, sonic stream) without a checkpoint, a
GPU, or the heavy deps.

``ping`` returns a hard-coded ``shape_meta`` matching config/task/umi_bimanual.yaml
(bimanual, action shape [20], horizon 16, obs horizons 2) so the bridge sizes its
obs provider + planner budget identically to the real server.

Usage:
    python -m Manip_Flow.mock_policy_server --port 5570 [--action-horizon 16]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

_UMI_ROOT = pathlib.Path(__file__).resolve().parents[1]
_REPO_ROOT = _UMI_ROOT.parents[1]
for _p in (str(_UMI_ROOT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.Deploy.bridge import dp_wire  # noqa: E402

# rot6d identity = first two rows of I3 (matches dp_adapter._IDENTITY_ROT6D).
_IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def _lowdim_meta(horizon: int) -> dict:
    return {"shape": [3], "horizon": horizon, "type": "low_dim"}


def mock_shape_meta(img_horizon: int = 2, low_horizon: int = 2, action_horizon: int = 16) -> dict:
    """Minimal bimanual shape_meta the bridge's obs provider + budget need."""
    obs = {}
    for i in range(2):
        obs[f"camera{i}_rgb"] = {"shape": [3, 224, 224], "horizon": img_horizon, "type": "rgb"}
        obs[f"robot{i}_eef_pos"] = {"shape": [3], "horizon": low_horizon, "type": "low_dim"}
        obs[f"robot{i}_eef_rot_axis_angle"] = {
            "raw_shape": [3], "shape": [6], "horizon": low_horizon,
            "type": "low_dim", "rotation_rep": "rotation_6d",
        }
        obs[f"robot{i}_gripper_width"] = {"shape": [1], "horizon": low_horizon, "type": "low_dim"}
    return {"obs": obs, "action": {"shape": [20], "horizon": action_horizon, "rotation_rep": "rotation_6d"}}


class MockDPPolicyServer:
    def __init__(self, port: int = 5570, host: str = "*", action_horizon: int = 16,
                 action_fps: float = 15.0, gripper_mid: float = 0.05):
        self.port = int(port)
        self.host = host
        self.action_horizon = int(action_horizon)
        self.action_fps = float(action_fps)
        self.action_dim = 20
        self.shape_meta = mock_shape_meta(action_horizon=self.action_horizon)
        self.gripper_mid = float(gripper_mid)

    def _identity_action(self) -> np.ndarray:
        a = np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)
        for arm in range(2):
            off = arm * 10
            a[:, off + 3 : off + 9] = _IDENTITY_ROT6D  # rot6d identity
            a[:, off + 9] = self.gripper_mid           # gripper width
        return a

    def _handle(self, buf: bytes) -> bytes:
        msg = dp_wire.decode_any(buf)
        mtype = msg.get("type")
        if mtype == "ping":
            return dp_wire.encode_pong(
                dp_wire.DPPongMetadata(
                    shape_meta=self.shape_meta,
                    action_horizon=self.action_horizon,
                    action_dim=self.action_dim,
                    obs_pose_repr="relative",
                    action_pose_repr="relative",
                    action_fps=self.action_fps,
                    rtc_enabled=False,
                )
            )
        if mtype == "predict":
            request = dp_wire.decode_predict_request(buf)
            t0 = time.perf_counter()
            act = self._identity_action()
            return dp_wire.encode_action_reply(
                request.req_id, act, (time.perf_counter() - t0) * 1e3
            )
        return dp_wire.encode_error(int(msg.get("req_id", -1)), f"unknown type {mtype!r}")

    def serve_forever(self) -> None:
        import zmq

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REP)
        sock.bind(f"tcp://{self.host}:{self.port}")
        print(f"[mock_policy_server] REP bound tcp://{self.host}:{self.port} "
              f"(identity action horizon={self.action_horizon})", flush=True)
        try:
            while True:
                sock.send(self._handle(sock.recv()))
        except KeyboardInterrupt:
            print("[mock_policy_server] interrupted; closing.")
        finally:
            sock.close(0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mock DP policy server (identity action)")
    p.add_argument("--port", type=int, default=5570)
    p.add_argument("--host", default="*")
    p.add_argument("--action-horizon", type=int, default=16)
    p.add_argument("--action-fps", type=float, default=15.0)
    p.add_argument("--gripper-mid", type=float, default=0.05)
    return p


def main() -> None:
    args = build_parser().parse_args()
    MockDPPolicyServer(port=args.port, host=args.host,
                       action_horizon=args.action_horizon, action_fps=args.action_fps,
                       gripper_mid=args.gripper_mid).serve_forever()


if __name__ == "__main__":
    main()

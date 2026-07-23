"""DP policy server: ZMQ REQ/REP wrapper around ``FlowPolicyInference``.

Runs in the env that has the heavy deps (torch/hydra/dill/diffusers/cv2). Owns
the flow-policy checkpoint; the bridge (``motion_prior``) talks to it over ZMQ
so the bridge itself needs none of those deps.

  bridge --(predict: JPEG cams + lowdim EE/width)--> server
  server --(action: (Ta,20) RAW relative)---------> bridge

The server decodes the JPEG camera frames (the ONLY place cv2/PIL touches
pixels), assembles the ``env_obs`` dict ``FlowPolicyInference.predict_relative_
action`` expects, and returns the raw relative action UNCHANGED (the bridge's
dp_adapter does ``world = base @ rel`` itself). On ``ping`` it replies with the
checkpoint's ``shape_meta`` + action horizon/dim so the bridge can size its obs
provider and planner budget WITHOUT loading a checkpoint.

CONSISTENCY INVARIANT: the EE pose in ``env_obs`` is computed on the BRIDGE
(HandPoseFK on executed qpos) and shipped as arrays; the server NEVER recomputes
EE from robot state -- it only merges the arrays the bridge sent with the images
it decoded. See gripper_obs_provider.py / inference.py docstrings.

Usage:
    python -m Manip_Flow.policy_server --ckpt <policy.ckpt> --device cuda:0 --port 5570
(run from the UMI-Diffusion repo root, or with it on PYTHONPATH.)
"""

from __future__ import annotations

import argparse
import io
import pathlib
import sys
import time
from typing import Dict, List

import numpy as np

# Repo root on sys.path so ``pipeline.Deploy.bridge.dp_wire`` imports. inference.py
# also inserts UMI-Diffusion for the ``Manip_Flow.*`` package name.
_UMI_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_UMI_ROOT) not in sys.path:
    sys.path.insert(0, str(_UMI_ROOT))
_REPO_ROOT = _UMI_ROOT.parents[1]  # pipeline/UMI-Diffusion -> repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.Deploy.bridge import dp_wire  # noqa: E402


def _decode_jpeg(value) -> np.ndarray:
    """JPEG bytes (or base64 str) -> RGB uint8 (H, W, 3). Server-side only."""
    import base64

    from PIL import Image

    if isinstance(value, str):
        raw = base64.b64decode(value)
    elif isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        if raw[:2] != b"\xff\xd8":  # not a JPEG SOI -> assume base64 text
            raw = base64.b64decode(value)
    else:
        raise TypeError(f"unsupported image payload type: {type(value)}")
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


class DPPolicyServer:
    """REQ/REP server wrapping ``FlowPolicyInference``."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        num_inference_steps: int | None = None,
        port: int = 5570,
        host: str = "*",
    ) -> None:
        from Manip_Flow.inference import FlowPolicyInference

        self.infer = FlowPolicyInference(
            ckpt_path, device=device, num_inference_steps=num_inference_steps
        )
        self.port = int(port)
        self.host = host
        self._n_robots = self._count_robots(self.infer.shape_meta)
        self._sock = None

    @staticmethod
    def _count_robots(shape_meta: dict) -> int:
        obs = shape_meta.get("obs", {})
        n = 0
        while f"robot{n}_eef_pos" in obs:
            n += 1
        return n or 2

    def _assemble_env_obs(
        self, cameras: Dict[str, List[bytes]], lowdim: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """JPEG lists + lowdim arrays -> the env_obs dict predict_ wants.

        Cameras decode to (img_horizon, H, W, 3) uint8 (newest last, as sent);
        lowdim arrays pass through unchanged. Missing/short handling is left to
        get_real_umi_obs_dict downstream (it resizes + slices to the horizons).
        """
        env_obs: Dict[str, np.ndarray] = {}
        for name, jpegs in cameras.items():
            frames = [_decode_jpeg(j) for j in jpegs]
            if not frames:
                raise ValueError(f"camera '{name}' had zero frames")
            env_obs[name] = np.stack(frames, axis=0)
        for k, v in lowdim.items():
            env_obs[k] = np.asarray(v)
        return env_obs

    def _handle(self, buf: bytes) -> bytes:
        msg = dp_wire.decode_any(buf)
        mtype = msg.get("type")
        if mtype == "ping":
            return dp_wire.encode_pong(
                shape_meta=self.infer.shape_meta,
                action_horizon=self.infer.action_horizon,
                action_dim=self.infer.action_dim,
                obs_pose_repr=self.infer.obs_pose_repr,
                action_pose_repr=self.infer.action_pose_repr,
            )
        if mtype == "predict":
            req_id, cameras, lowdim, _start, _win = dp_wire.decode_predict_request(buf)
            t0 = time.perf_counter()
            try:
                env_obs = self._assemble_env_obs(cameras, lowdim)
                action = self.infer.predict_relative_action(env_obs)  # (Ta,20) raw rel
            except Exception as exc:  # keep the server alive; report to the client
                return dp_wire.encode_error(req_id, f"{type(exc).__name__}: {exc}")
            server_ms = (time.perf_counter() - t0) * 1e3
            return dp_wire.encode_action_reply(req_id, action, server_ms)
        return dp_wire.encode_error(int(msg.get("req_id", -1)), f"unknown type {mtype!r}")

    def serve_forever(self) -> None:
        import zmq

        ctx = zmq.Context.instance()
        self._sock = ctx.socket(zmq.REP)
        self._sock.bind(f"tcp://{self.host}:{self.port}")
        print(
            f"[policy_server] REP bound on tcp://{self.host}:{self.port}; "
            f"n_robots={self._n_robots} action=({self.infer.action_horizon},"
            f"{self.infer.action_dim}) obs_repr={self.infer.obs_pose_repr} "
            f"action_repr={self.infer.action_pose_repr}",
            flush=True,
        )
        try:
            while True:
                buf = self._sock.recv()
                self._sock.send(self._handle(buf))
        except KeyboardInterrupt:
            print("[policy_server] interrupted; closing.")
        finally:
            self._sock.close(0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DP flow-policy ZMQ REQ/REP server")
    p.add_argument("--ckpt", required=True, help="flow/diffusion policy workspace .ckpt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--infer-steps", type=int, default=None,
                   help="override policy num_inference_steps (default: ckpt value)")
    p.add_argument("--port", type=int, default=5570)
    p.add_argument("--host", default="*")
    return p


def main() -> None:
    args = build_parser().parse_args()
    DPPolicyServer(
        args.ckpt, device=args.device, num_inference_steps=args.infer_steps,
        port=args.port, host=args.host,
    ).serve_forever()


if __name__ == "__main__":
    main()

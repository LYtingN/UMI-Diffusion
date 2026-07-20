"""Feed a LeRobot v2.1 UMI dataset to UMI's bimanual policy WITHOUT a .zarr.zip.

``UmiDataset`` is format-agnostic below its constructor: everything after it
builds a UMI ``ReplayBuffer`` only consumes ``replay_buffer[key]`` /
``episode_ends`` / ``n_episodes`` (see ``diffusion_policy/dataset/umi_dataset.py``
and ``common/sampler.py``). The only UMI-specific thing is the *source*: UMI
opens a ``.zarr.zip`` via ``zarr.ZipStore``. This module swaps that one step:
it reads the LeRobot ``data/*.parquet`` (proprio) + ``videos/**/*.mp4`` (rgb)
and packs them into an in-memory ``ReplayBuffer`` with EXACTLY the keys UMI's
``umi_bimanual`` task expects, then reuses 100% of UMI's sampler / normalizer /
``__getitem__``. The UMI repo is not modified.

LeRobot (this dataset)              ->  UMI ReplayBuffer key
  observation.eef_pose_left  (7)    ->  robot0_eef_pos (3) + robot0_eef_rot_axis_angle (3)
  observation.eef_pose_right (7)    ->  robot1_eef_pos (3) + robot1_eef_rot_axis_angle (3)
  observation.gripper       [L,R]   ->  robot0_gripper_width (1), robot1_gripper_width (1)
  observation.images.left  (mp4)    ->  camera0_rgb (T,224,224,3) uint8
  observation.images.right (mp4)    ->  camera1_rgb (T,224,224,3) uint8

Key facts verified against the actual data / UMI source:
  * eef_pose is pos(3) + quat in WXYZ order (parquet col-3 is the scalar w,
    |quat| == 1). UMI stores/consumes rotation as an axis-angle rotvec(3)
    (``robot*_eef_rot_axis_angle``; the dataset later Slerps it and calls
    ``st.Rotation.from_rotvec`` in the sampler), so we convert wxyz -> rotvec.
  * ``action`` is NOT stored: SequenceSampler auto-builds it as
    concat([eef_pos, eef_rot_axis_angle, gripper_width]) per robot when
    'action' is absent from the replay buffer -> 2 x 7 = 14-D raw, which UMI's
    ``__getitem__`` then turns into the 20-D rot6d action. Matches shape_meta.
  * gripper_width feeds the sampler's grasp heuristic (threshold 0.08) and is
    range-normalized -- raw metric width is exactly what UMI expects.
  * Images are center-crop-resized to 224x224 with UMI's own transform when
    cv2 is available (identical to scripts_slam_pipeline/07), else Pillow.

Hydra usage: point the task's dataset ``_target_`` at this class and give it
the LeRobot dataset ROOT dir instead of a .zarr.zip. See
``config/train_flow_lerobot_umi_pnp.yaml``.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import scipy.spatial.transform as st

from Manip_Flow.common.replay_buffer import ReplayBuffer
from Manip_Flow.dataset.umi_dataset import UmiDataset


# ---- LeRobot field layout (bimanual UMI DAS-gripper) -----------------------
# robot0 == left == "master", robot1 == right == "sub" (matches umi_to_lerobot.py)
_ROBOT_TO_POSE_KEY = {
    0: "observation.eef_pose_left",
    1: "observation.eef_pose_right",
}
_ROBOT_TO_VIDEO_KEY = {
    0: "observation.images.left",
    1: "observation.images.right",
}
_GRIPPER_KEY = "observation.gripper"  # (T, 2) -> [:,0]=left/robot0, [:,1]=right/robot1


def _quat_wxyz_to_rotvec(quat_wxyz: np.ndarray) -> np.ndarray:
    """(T,4) WXYZ quaternion -> (T,3) axis-angle rotvec (scipy uses XYZW)."""
    q_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    # guard against tiny denorm from resampling before feeding scipy
    q_xyzw = q_xyzw / np.linalg.norm(q_xyzw, axis=1, keepdims=True)
    return st.Rotation.from_quat(q_xyzw).as_rotvec().astype(np.float32)


def _make_image_transform(in_hw, out_res):
    """Center-crop-resize (ih,iw)->out_res, prefer UMI's cv2 transform."""
    ih, iw = in_hw
    ow, oh = out_res
    try:
        from Manip_Flow.common.cv2_util import get_image_transform

        tf = get_image_transform(input_res=(iw, ih), output_res=(ow, oh), bgr_to_rgb=False)
        return tf
    except Exception:
        # Pillow fallback: same "resize short side then center crop" geometry.
        from PIL import Image

        def tf(img: np.ndarray) -> np.ndarray:
            if (iw / ih) >= (ow / oh):
                rh, rw = oh, int(np.ceil(oh / ih * iw))
            else:
                rw, rh = ow, int(np.ceil(ow / iw * ih))
            im = Image.fromarray(img).resize((rw, rh), Image.BILINEAR)
            left = (rw - ow) // 2
            top = (rh - oh) // 2
            return np.asarray(im)[top:top + oh, left:left + ow, :]

        return tf


def _read_video_frames(mp4_path: str, expected_len: int, transform) -> np.ndarray:
    """Decode an mp4 to (T,H,W,3) uint8, resized by ``transform``.

    Lengths are asserted against the parquet row count so proprio and rgb stay
    aligned frame-for-frame (both were written on the same fps grid by
    ``umi_to_lerobot.py``). A single trailing/leading off-by-one from the codec
    is clamped by repeating the edge frame, anything larger is a hard error.
    """
    import imageio

    reader = imageio.get_reader(mp4_path, "ffmpeg")
    frames = []
    for fr in reader:
        frames.append(transform(np.asarray(fr)))
    reader.close()
    arr = np.stack(frames, axis=0).astype(np.uint8)

    if arr.shape[0] != expected_len:
        diff = arr.shape[0] - expected_len
        if abs(diff) > 2:
            raise ValueError(
                f"{os.path.basename(mp4_path)}: {arr.shape[0]} video frames vs "
                f"{expected_len} proprio rows (diff {diff}); refusing to guess alignment."
            )
        if diff > 0:  # too many frames -> drop from the end
            arr = arr[:expected_len]
        else:  # too few -> pad by repeating last frame
            pad = np.repeat(arr[-1:], -diff, axis=0)
            arr = np.concatenate([arr, pad], axis=0)
    return arr


def build_umi_replay_buffer_from_lerobot(
    dataset_path: str,
    out_res=(224, 224),
    max_episodes: Optional[int] = None,
    verbose: bool = True,
) -> ReplayBuffer:
    """Read a LeRobot v2.1 UMI dataset dir into an in-memory UMI ReplayBuffer.

    Args:
        dataset_path: LeRobot dataset ROOT (the dir holding data/ meta/ videos/).
        out_res: (W, H) target for camera frames (UMI shape_meta is 224x224).
        max_episodes: cap episodes (debug/smoke); None = all.
    """
    import json

    root = dataset_path
    info = json.load(open(os.path.join(root, "meta", "info.json")))
    data_tmpl = info["data_path"]      # data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
    video_tmpl = info["video_path"]    # videos/chunk-{...}/{video_key}/episode_{...}.mp4
    chunks_size = info.get("chunks_size", 1000)
    n_ep = info["total_episodes"]
    if max_episodes is not None:
        n_ep = min(n_ep, max_episodes)

    replay_buffer = ReplayBuffer.create_empty_numpy()
    img_tf = None  # built lazily once we know the source resolution

    for ep in range(n_ep):
        chunk = ep // chunks_size
        pq = os.path.join(root, data_tmpl.format(episode_chunk=chunk, episode_index=ep))
        df = pd.read_parquet(pq)
        T = len(df)

        episode = {}
        for robot_id in (0, 1):
            pose = np.stack(df[_ROBOT_TO_POSE_KEY[robot_id]].values).astype(np.float32)  # (T,7)
            episode[f"robot{robot_id}_eef_pos"] = pose[:, 0:3].astype(np.float32)
            episode[f"robot{robot_id}_eef_rot_axis_angle"] = _quat_wxyz_to_rotvec(pose[:, 3:7])

        gripper = np.stack(df[_GRIPPER_KEY].values).astype(np.float32)  # (T,2)
        episode["robot0_gripper_width"] = gripper[:, 0:1].astype(np.float32)
        episode["robot1_gripper_width"] = gripper[:, 1:2].astype(np.float32)

        for robot_id in (0, 1):
            vkey = _ROBOT_TO_VIDEO_KEY[robot_id]
            mp4 = os.path.join(root, video_tmpl.format(episode_chunk=chunk, video_key=vkey, episode_index=ep))
            # peek source resolution to build the transform exactly once
            if img_tf is None:
                import imageio

                r0 = imageio.get_reader(mp4, "ffmpeg")
                h0, w0 = np.asarray(r0.get_next_data()).shape[:2]
                r0.close()
                img_tf = _make_image_transform((h0, w0), out_res)
            episode[f"camera{robot_id}_rgb"] = _read_video_frames(mp4, T, img_tf)

        replay_buffer.add_episode(episode, compressors=None)
        if verbose:
            print(f"[lerobot->umi] episode {ep + 1}/{n_ep}: {T} frames", flush=True)

    return replay_buffer


class LeRobotUmiDataset(UmiDataset):
    """UmiDataset backed by a LeRobot v2.1 dir instead of a .zarr.zip.

    Drop-in for ``diffusion_policy.dataset.umi_dataset.UmiDataset`` in the UMI
    ``umi_bimanual`` task: same shape_meta, pose_repr, val_ratio, etc. Only the
    replay-buffer source differs, so ``get_normalizer`` / ``get_validation_dataset``
    / ``__getitem__`` are all inherited unchanged.
    """

    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        cache_dir: Optional[str] = None,          # accepted & ignored (kept in-memory)
        pose_repr: dict = {},
        action_padding: bool = False,
        temporally_independent_normalization: bool = False,
        repeat_frame_prob: float = 0.0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_duration: Optional[float] = None,
        max_episodes: Optional[int] = None,        # extra: cap episodes for smoke runs
    ):
        from Manip_Flow.common.sampler import SequenceSampler, get_val_mask

        self.pose_repr = pose_repr
        self.obs_pose_repr = self.pose_repr.get("obs_pose_repr", "rel")
        self.action_pose_repr = self.pose_repr.get("action_pose_repr", "rel")

        # infer target rgb res from shape_meta (defaults to 224x224)
        out_res = (224, 224)
        for _k, _attr in shape_meta["obs"].items():
            if _attr.get("type") == "rgb":
                c, h, w = _attr["shape"]
                out_res = (w, h)
                break

        replay_buffer = build_umi_replay_buffer_from_lerobot(
            dataset_path, out_res=out_res, max_episodes=max_episodes
        )

        # ---- everything below mirrors UmiDataset.__init__ verbatim ----------
        self.num_robot = 0
        rgb_keys, lowdim_keys = [], []
        key_horizon, key_down_sample_steps, key_latency_steps = {}, {}, {}
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            typ = attr.get("type", "low_dim")
            if typ == "rgb":
                rgb_keys.append(key)
            elif typ == "low_dim":
                lowdim_keys.append(key)
            if key.endswith("eef_pos"):
                self.num_robot += 1
            key_horizon[key] = shape_meta["obs"][key]["horizon"]
            key_latency_steps[key] = shape_meta["obs"][key]["latency_steps"]
            key_down_sample_steps[key] = shape_meta["obs"][key]["down_sample_steps"]

        key_horizon["action"] = shape_meta["action"]["horizon"]
        key_latency_steps["action"] = shape_meta["action"]["latency_steps"]
        key_down_sample_steps["action"] = shape_meta["action"]["down_sample_steps"]

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask

        self.sampler_lowdim_keys = [k for k in lowdim_keys if "wrt" not in k]
        for key in replay_buffer.keys():
            if key.endswith("_demo_start_pose") or key.endswith("_demo_end_pose"):
                self.sampler_lowdim_keys.append(key)
                query_key = key.split("_")[0] + "_eef_pos"
                key_horizon[key] = shape_meta["obs"][query_key]["horizon"]
                key_latency_steps[key] = shape_meta["obs"][query_key]["latency_steps"]
                key_down_sample_steps[key] = shape_meta["obs"][query_key]["down_sample_steps"]

        sampler = SequenceSampler(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            rgb_keys=rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=key_horizon,
            key_latency_steps=key_latency_steps,
            key_down_sample_steps=key_down_sample_steps,
            episode_mask=train_mask,
            action_padding=action_padding,
            repeat_frame_prob=repeat_frame_prob,
            max_duration=max_duration,
        )
        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        self.val_mask = val_mask
        self.action_padding = action_padding
        self.repeat_frame_prob = repeat_frame_prob
        self.max_duration = max_duration
        self.sampler = sampler
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False

#!/usr/bin/env python
"""Offline conversion: LeRobot v2.1 UMI dir(s) -> ONE UMI ``.zarr.zip``.

Why this exists
---------------
``dataset/lerobot_umi_dataset.py`` reads the LeRobot dir directly, which means
every rank fully ffmpeg-decodes every mp4 into RAM at every job start:
~19 min and ~28.6 GiB *per rank* for the drawer datasets (x8 ranks per node).
The videos are h264 with GOP=250 at 1600x1300, so per-``__getitem__`` random
seeking is not a viable alternative (~200x slower than sequential decode).

Sequential decode is the one access pattern h264 is good at, so do it exactly
once, offline, and write a UMI-native ``.zarr.zip`` with per-frame JPEG-XL
chunks. Training then goes back to ``dataset.umi_dataset.UmiDataset`` with
``cache_dir`` set, which already implements FileLock + LMDB + read-only mmap:
one build per node, ~0 resident RAM, seconds to start. Nothing in
``common/sampler.py`` or ``dataset/umi_dataset.py`` needs to change -- the
sampler already keeps rgb as a lazy zarr handle (see sampler.py:89-90).

The ReplayBuffer keys / dtypes / episode ORDER produced here are identical to
what ``LeRobotUmiDataset`` builds in memory today (the field mapping and the
image transform are imported from that module rather than re-derived), so the
seeded train/val split over the pooled episodes is unchanged.

Usage
-----
    python Manip_Flow/scripts/lerobot_to_umi_zarr.py \
        --input /data/nas_ray/home/eason.er/datasets/drawer0805 \
        --input /data/nas_ray/home/eason.er/datasets/drawer \
        --output /data/nas_ray/home/eason.er/datasets/drawer_pooled_0807.zarr.zip

``--input`` order defines episode order in the pooled buffer: pass the SAME
order as the old ``task.dataset_path=[...]`` list.
"""

import argparse
import json
import os
import pathlib
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import zarr  # noqa: E402

from Manip_Flow.codecs.imagecodecs_numcodecs import JpegXl, register_codecs  # noqa: E402
from Manip_Flow.common.replay_buffer import ReplayBuffer, get_optimal_chunks  # noqa: E402
from Manip_Flow.dataset.lerobot_umi_dataset import (  # noqa: E402
    _GRIPPER_KEY,
    _ROBOT_TO_POSE_KEY,
    _ROBOT_TO_VIDEO_KEY,
    _make_image_transform,
    _quat_wxyz_to_rotvec,
)

register_codecs()

# lowdim keys written per robot, and their width. Must match what
# LeRobotUmiDataset._append_lerobot_root_to_buffer puts in each episode dict.
_LOWDIM_DIMS = {
    'eef_pos': 3,
    'eef_rot_axis_angle': 3,
    'gripper_width': 1,
}
_ROBOT_IDS = (0, 1)


@dataclass
class EpisodeSpec:
    root: str
    tag: str            # human-readable source tag for logs
    ep: int             # episode index within its root
    chunk: int          # LeRobot episode_chunk
    parquet: str
    videos: dict        # robot_id -> mp4 path
    n_frames: int       # authoritative length (parquet row count)
    start: int = -1     # global row offset in the pooled buffer


def scan_root(root: str, tag: str, max_episodes: Optional[int]) -> List[EpisodeSpec]:
    """Enumerate one LeRobot v2.1 dir, taking episode lengths from parquet metadata.

    Only the parquet footer is read (no column data, no video), so this is fast
    and lets us pre-allocate every zarr array at its exact final shape -- no
    resize / partial-chunk rewrite while filling.
    """
    import pyarrow.parquet as pq

    info = json.load(open(os.path.join(root, 'meta', 'info.json')))
    data_tmpl = info['data_path']
    video_tmpl = info['video_path']
    chunks_size = info.get('chunks_size', 1000)
    n_ep = info['total_episodes']
    if max_episodes is not None:
        n_ep = min(n_ep, max_episodes)

    specs = list()
    for ep in range(n_ep):
        chunk = ep // chunks_size
        parquet = os.path.join(root, data_tmpl.format(episode_chunk=chunk, episode_index=ep))
        videos = {
            rid: os.path.join(root, video_tmpl.format(
                episode_chunk=chunk, video_key=_ROBOT_TO_VIDEO_KEY[rid], episode_index=ep))
            for rid in _ROBOT_IDS
        }
        for path in [parquet] + list(videos.values()):
            if not os.path.isfile(path):
                raise FileNotFoundError(f'{tag} episode {ep}: missing {path}')
        n_frames = pq.ParquetFile(parquet).metadata.num_rows
        if n_frames <= 0:
            raise ValueError(f'{tag} episode {ep}: {parquet} has {n_frames} rows')
        specs.append(EpisodeSpec(
            root=root, tag=tag, ep=ep, chunk=chunk,
            parquet=parquet, videos=videos, n_frames=n_frames))
    return specs


def peek_source_res(mp4_path: str) -> Tuple[int, int]:
    """(H, W) of the first frame -- same probe LeRobotUmiDataset does."""
    import imageio

    reader = imageio.get_reader(mp4_path, 'ffmpeg')
    try:
        h, w = np.asarray(reader.get_next_data()).shape[:2]
    finally:
        reader.close()
    return int(h), int(w)


def write_episode_lowdim(spec: EpisodeSpec, arrays: dict) -> None:
    """Read one parquet and write its proprio rows into the pre-allocated arrays.

    Sequential / main-thread only: lowdim chunks span many frames (and thus
    episode boundaries), so concurrent writers could touch the same chunk.
    """
    df = pd.read_parquet(spec.parquet)
    if len(df) != spec.n_frames:
        raise ValueError(
            f'{spec.tag} episode {spec.ep}: parquet row count changed since scan '
            f'({len(df)} vs {spec.n_frames})')
    sl = slice(spec.start, spec.start + spec.n_frames)

    for rid in _ROBOT_IDS:
        pose = np.stack(df[_ROBOT_TO_POSE_KEY[rid]].values).astype(np.float32)  # (T,7)
        arrays[f'robot{rid}_eef_pos'][sl] = pose[:, 0:3].astype(np.float32)
        arrays[f'robot{rid}_eef_rot_axis_angle'][sl] = _quat_wxyz_to_rotvec(pose[:, 3:7])

    gripper = np.stack(df[_GRIPPER_KEY].values).astype(np.float32)  # (T,2) [left, right]
    arrays['robot0_gripper_width'][sl] = gripper[:, 0:1].astype(np.float32)
    arrays['robot1_gripper_width'][sl] = gripper[:, 1:2].astype(np.float32)


def write_episode_rgb(spec: EpisodeSpec, rid: int, arr, transform) -> int:
    """Stream-decode one mp4 straight into ``arr[spec.start : spec.start+T]``.

    One frame in flight at a time, and every write covers exactly one zarr chunk
    (chunks are ``(1, H, W, 3)``), so N of these run concurrently on distinct
    chunk files without a synchronizer.

    Length policy is identical to ``lerobot_umi_dataset._read_video_frames``:
    the parquet row count wins, a codec off-by-<=2 is clamped (extra frames
    dropped from the end / last frame repeated), anything larger is a hard error.
    """
    import imageio

    T = spec.n_frames
    out_base = spec.start
    n_decoded = 0
    last = None

    reader = imageio.get_reader(spec.videos[rid], 'ffmpeg')
    try:
        for fr in reader:
            if n_decoded < T:
                last = np.ascontiguousarray(transform(np.asarray(fr)))
                arr[out_base + n_decoded] = last
                n_decoded += 1
                continue
            # past the parquet length: count a couple, then bail out rather than
            # decode a possibly much longer video just to report the number.
            n_decoded += 1
            if n_decoded - T > 2:
                break
    finally:
        reader.close()

    diff = n_decoded - T
    if abs(diff) > 2:
        raise ValueError(
            f'{os.path.basename(spec.videos[rid])}: {n_decoded}{"+" if diff > 2 else ""} '
            f'video frames vs {T} proprio rows (diff {diff}); refusing to guess alignment.')
    if last is None:
        raise ValueError(f'{os.path.basename(spec.videos[rid])}: decoded 0 frames')
    if diff < 0:  # too few -> repeat the last frame, like the in-memory loader
        for i in range(n_decoded, T):
            arr[out_base + i] = last
    return T


def verify_output(output: str, n_steps: int, n_episodes: int, out_res: Tuple[int, int]) -> None:
    """Reopen the finished zip the way UmiDataset will and touch one rgb frame.

    Cheap, but it catches the two failure modes that would otherwise only show
    up minutes into a 16-rank job: a truncated/mis-sized array, and a JPEG-XL
    chunk that cannot be decoded (i.e. codec registration gone wrong).
    """
    ow, oh = out_res
    with zarr.ZipStore(output, mode='r') as store:
        root = zarr.group(store)
        ends = root['meta']['episode_ends'][:]
        assert len(ends) == n_episodes, f'{len(ends)} episodes, expected {n_episodes}'
        assert ends[-1] == n_steps, f'episode_ends[-1]={ends[-1]}, expected {n_steps}'
        assert np.all(np.diff(ends) > 0), 'episode_ends is not strictly increasing'
        for rid in _ROBOT_IDS:
            for cat, dim in _LOWDIM_DIMS.items():
                arr = root['data'][f'robot{rid}_{cat}']
                assert arr.shape == (n_steps, dim), f'{arr.path}: {arr.shape}'
                assert arr.dtype == np.float32, f'{arr.path}: {arr.dtype}'
            cam = root['data'][f'camera{rid}_rgb']
            assert cam.shape == (n_steps, oh, ow, 3), f'{cam.path}: {cam.shape}'
            assert cam.dtype == np.uint8, f'{cam.path}: {cam.dtype}'
            frame = cam[n_steps // 2]  # forces a real JPEG-XL decode
            assert frame.shape == (oh, ow, 3)
    print('[verify] ok: shapes, dtypes, episode_ends, JPEG-XL round-trip', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convert LeRobot v2.1 UMI dataset dir(s) into one UMI .zarr.zip')
    parser.add_argument('-i', '--input', action='append', required=True, metavar='LEROBOT_ROOT',
                        help='LeRobot dataset root (dir with data/ meta/ videos/). '
                             'Repeat to pool several; ORDER defines episode order.')
    parser.add_argument('-o', '--output', required=True, help='output .zarr.zip path')
    parser.add_argument('--out-res', nargs=2, type=int, default=(224, 224), metavar=('W', 'H'),
                        help='target rgb resolution, must match shape_meta (default 224 224)')
    parser.add_argument('--max-episodes', type=int, default=None,
                        help='cap episodes PER input root (smoke test)')
    parser.add_argument('--num-workers', type=int, default=min(8, (os.cpu_count() or 8)),
                        help='concurrent mp4 decode+encode workers (default %(default)s)')
    parser.add_argument('--jxl-level', type=int, default=99,
                        help='JpegXl level; >=100 or <0 selects lossless (default %(default)s)')
    parser.add_argument('--temp-dir', default=None,
                        help='working DirectoryStore location (default: <output>.tmp)')
    parser.add_argument('--keep-temp', action='store_true',
                        help='keep the working directory store after zipping')
    parser.add_argument('--overwrite', action='store_true', help='overwrite an existing output')
    args = parser.parse_args()

    # one thread per cv2 call: we already parallelize across episodes
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        pass

    roots = [os.path.abspath(os.path.expanduser(r)) for r in args.input]
    output = os.path.abspath(os.path.expanduser(args.output))
    if os.path.exists(output) and not args.overwrite:
        raise SystemExit(f'ERROR: {output} exists (pass --overwrite to replace)')
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temp_dir = args.temp_dir or (output + '.tmp')
    if os.path.exists(temp_dir):
        raise SystemExit(f'ERROR: working dir {temp_dir} exists; remove it first')
    ow, oh = args.out_res

    # ---- phase 1: scan (parquet footers only) -------------------------------
    t0 = time.monotonic()
    specs: List[EpisodeSpec] = list()
    for i, root in enumerate(roots):
        tag = f'[{i + 1}/{len(roots)} {os.path.basename(os.path.normpath(root))}]'
        rs = scan_root(root, tag, args.max_episodes)
        print(f'[scan]{tag} {len(rs)} episodes, {sum(s.n_frames for s in rs)} frames', flush=True)
        specs.extend(rs)
    if not specs:
        raise SystemExit('ERROR: no episodes found')

    cursor = 0
    for s in specs:
        s.start = cursor
        cursor += s.n_frames
    n_steps = cursor
    n_episodes = len(specs)
    episode_ends = np.cumsum([s.n_frames for s in specs]).astype(np.int64)
    src_bytes = sum(os.path.getsize(p) for s in specs for p in s.videos.values())
    print(f'[scan] pooled: {n_episodes} episodes, {n_steps} frames, '
          f'{len(_ROBOT_IDS)} cameras, {src_bytes / 2**30:.2f} GiB of mp4', flush=True)

    ih, iw = peek_source_res(specs[0].videos[0])
    transform = _make_image_transform((ih, iw), (ow, oh))
    print(f'[scan] source {iw}x{ih} -> center-crop-resize {ow}x{oh}', flush=True)

    # ---- phase 2: pre-allocate every array at its final shape ---------------
    # Working store is a DirectoryStore (not the zip): the zip format cannot
    # overwrite entries, and we want the pooled buffer on disk rather than in
    # RAM. It is byte-copied into the zip at the end, no recompression.
    img_compressor = JpegXl(level=args.jxl_level, numthreads=1)
    lowdim_compressor = ReplayBuffer.resolve_compressor('disk')
    store = zarr.DirectoryStore(temp_dir)
    replay_buffer = ReplayBuffer.create_empty_zarr(storage=store)
    arrays = dict()
    for rid in _ROBOT_IDS:
        for cat, dim in _LOWDIM_DIMS.items():
            key = f'robot{rid}_{cat}'
            arrays[key] = replay_buffer.data.zeros(
                name=key, shape=(n_steps, dim), dtype=np.float32,
                chunks=get_optimal_chunks((n_steps, dim), np.float32),
                compressor=lowdim_compressor)
        key = f'camera{rid}_rgb'
        arrays[key] = replay_buffer.data.zeros(
            name=key, shape=(n_steps, oh, ow, 3), dtype=np.uint8,
            chunks=(1, oh, ow, 3),  # one frame per chunk == one JPEG-XL image
            compressor=img_compressor)

    ok = False
    try:
        # ---- phase 3: proprio (sequential, cheap) --------------------------
        for i, s in enumerate(specs):
            write_episode_lowdim(s, arrays)
            if (i + 1) % 25 == 0 or i + 1 == n_episodes:
                print(f'[lowdim] {i + 1}/{n_episodes} episodes', flush=True)

        # ---- phase 4: rgb (parallel sequential-decode -> JPEG-XL) ----------
        tasks = [(s, rid) for s in specs for rid in _ROBOT_IDS]
        t_rgb = time.monotonic()
        done_frames = 0
        total_frames = n_steps * len(_ROBOT_IDS)
        with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {
                pool.submit(write_episode_rgb, s, rid, arrays[f'camera{rid}_rgb'], transform):
                    (s, rid)
                for s, rid in tasks
            }
            for n_done, fut in enumerate(as_completed(futures), start=1):
                s, rid = futures[fut]
                done_frames += fut.result()  # re-raises worker exceptions
                el = time.monotonic() - t_rgb
                fps = done_frames / max(el, 1e-6)
                eta = (total_frames - done_frames) / max(fps, 1e-6)
                print(f'[rgb] {n_done}/{len(tasks)} videos '
                      f'({done_frames}/{total_frames} frames, {fps:.0f} fps, '
                      f'eta {eta / 60:.1f} min) {s.tag} ep {s.ep} cam{rid}', flush=True)

        # ---- phase 5: meta + pack into the .zarr.zip ------------------------
        replay_buffer.update_meta({'episode_ends': episode_ends})
        print(f'[save] writing {output}', flush=True)
        if os.path.exists(output):
            os.remove(output)
        with zarr.ZipStore(output, mode='w') as zip_store:
            # chunks/compressors omitted -> resolved from the source arrays ->
            # save_to_store takes the zarr.copy_store byte-copy path (no re-encode)
            replay_buffer.save_to_store(store=zip_store)
        ok = True
    finally:
        if os.path.isdir(temp_dir) and not args.keep_temp and ok:
            shutil.rmtree(temp_dir, ignore_errors=True)
        elif os.path.isdir(temp_dir):
            print(f'[cleanup] working store kept at {temp_dir}', flush=True)

    verify_output(output, n_steps, n_episodes, (ow, oh))

    out_bytes = os.path.getsize(output)
    raw_bytes = n_steps * len(_ROBOT_IDS) * oh * ow * 3
    print(f'[done] {output}\n'
          f'       {n_episodes} episodes / {n_steps} frames in {(time.monotonic() - t0) / 60:.1f} min\n'
          f'       {out_bytes / 2**30:.2f} GiB zarr.zip '
          f'(mp4 source {src_bytes / 2**30:.2f} GiB, '
          f'uncompressed rgb would be {raw_bytes / 2**30:.1f} GiB)', flush=True)


if __name__ == '__main__':
    main()

"""Manip_Flow: upper-layer bimanual manipulation policy (flow matching).

UMI-compatible bimanual policy that replaces the Diffusion Policy DDPM/DDIM
head with conditional flow matching (rectified flow). Everything else — the
``umi_bimanual`` shape_meta, TimmObsEncoder, ConditionalUnet1D backbone,
UmiDataset and the training workspace — is VENDORED into this package
(``common/``, ``model/``, ``policy/``, ``dataset/``, ``workspace/``,
``env_runner/``, ``codecs/``, ``config/task/``), copied from the
``universal_manipulation_interface`` repo's ``diffusion_policy``/``umi``
packages with imports rewritten to ``pipeline.Manip_Flow.*``. Training and
inference no longer require that external repo on ``sys.path`` — only the
Motion_Prior repo root. Data is fed from LeRobot v2.1 dirs via
``dataset/lerobot_umi_dataset.py`` (no ``.zarr.zip``).

Interface contract with the lower level (Prior_Recon):
    predict_action -> (Ta, 20) RELATIVE bimanual action
        [r0: pos3 rot6d grip1 | r1: pos3 rot6d grip1], rel to last-obs EE pose
    -> DPKeypointProvider.dp_infer_fn (bridge/dp_base_anchor.py, P2)
    -> dp_action_to_keypoints (bridge/dp_adapter.py, P1)
    -> OnlinePrimitivePlanner.plan_segment(kp_window=...)  # 69-dim public API
"""

import os
from pathlib import Path
import subprocess

import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from Manip_Flow.policy.flow_timm_policy import FlowTimmPolicy

MANIP_FLOW_ROOT = Path(__file__).resolve().parents[1]


def test_humi_unet_config_preserves_lerobot_task_contract() -> None:
    # Given: the dedicated HuMI-style Manip_Flow training configuration.
    with initialize_config_dir(
        version_base=None,
        config_dir=str(MANIP_FLOW_ROOT / "config"),
    ):
        config = compose(config_name="train_flow_unet_humi_lerobot_umi_pnp")

    # When: Hydra resolves the configured dataset and policy targets.
    dataset_target = config.task.dataset._target_
    policy = config.policy

    # Then: LeRobot/action metadata stay task-owned while the policy uses HuMI's UNet.
    assert dataset_target == "Manip_Flow.dataset.lerobot_umi_dataset.LeRobotUmiDataset"
    assert list(config.shape_meta.action.shape) == [20]
    assert policy.backbone == "unet"
    assert policy.unet_time_log_scale == 10.0
    assert policy.time_embed_scale == 1.0
    assert OmegaConf.is_interpolation(config.task, "dataset_path")
    assert OmegaConf.is_interpolation(config.task.dataset, "dataset_path")
    assert config.logging.mode == "disabled"


def test_humi_unet_config_matches_30hz_lerobot_temporal_contract() -> None:
    # Given: the dedicated HuMI-style Manip_Flow training configuration.
    with initialize_config_dir(
        version_base=None,
        config_dir=str(MANIP_FLOW_ROOT / "config"),
    ):
        # When: Hydra resolves task interpolation into machine-consumed shape metadata.
        config = compose(config_name="train_flow_unet_humi_lerobot_umi_pnp")

    # Then: the compact DINOv3 policy uses the 10 Hz, 100-demo contract.
    assert config.policy.obs_encoder.model_name == "vit_base_patch16_dinov3.lvd1689m"
    assert config.policy.obs_encoder.pretrained is True
    assert config.policy.obs_encoder.frozen is False
    assert config.policy.obs_encoder.finetune_last_n_blocks == 1
    assert config.policy.obs_encoder.feature_aggregation == "attention_pool_2d"
    assert config.policy.obs_encoder.normalize_rgb is True
    assert config.policy.obs_encoder.share_rgb_model is False
    assert list(config.policy.down_dims) == [96, 192, 384]
    assert config.policy.num_inference_steps == 8
    assert config.training.freeze_encoder is False
    assert config.training.num_epochs == 100
    assert config.task.dataset.val_ratio == 0.10
    assert config.task.dataset_frequency == 30.0
    assert config.task.obs_down_sample_steps == 3
    assert config.task.dataset_frequency / config.task.obs_down_sample_steps == 10.0
    assert config.task.img_obs_horizon == 1
    assert config.task.low_dim_obs_horizon == 3
    assert config.shape_meta.action.horizon == 40
    assert config.shape_meta.action.down_sample_steps == 3
    assert config.policy.rtc_execution_horizon == 20
    assert config.policy.rtc_max_guidance_weight == 5.0
    assert config.policy.rtc_prefix_schedule == "exp"


def test_humi_unet_h100_job_uses_the_dinov3_config() -> None:
    # Given: the production multi-GPU job definition. (Was job_flow_umi_h100.yaml
    # until it was deleted in 5f4c70e, then the 12-epoch drawer job; both are
    # gone, so the 40-epoch drawer job is the live unet one.)
    job_config = OmegaConf.load(
        MANIP_FLOW_ROOT / "config" / "job_flow_umi_drawer_h100.yaml"
    )

    # When: its Hydra command-line arguments are inspected.
    args = list(job_config.tasks[0].args)
    config_name_index = args.index("--config-name") + 1

    # Then: production launches the unet DINOv3 policy with unshared towers.
    assert args[config_name_index] == "train_flow_unet_humi_lerobot_umi_pnp"
    assert "policy.obs_encoder.share_rgb_model=False" in args
    assert "dataloader.batch_size=16" in args
    assert "val_dataloader.batch_size=16" in args
    assert "logging.mode=offline" in args
    assert "wandb>=0.18.0" in job_config.pip_packages


# Which velocity model each submitted job actually builds. A stray --config-name
# would otherwise switch the backbone silently, and the two arms are not
# comparable: unet velocity is 10.80M against the DiT's 41.14M.
EXPECTED_JOB_BACKBONE = {
    "job_flow_umi_drawer_h100.yaml": "unet",
    "job_flow_umi_drawer_ditctx_h100_ep100.yaml": "dit",
    "job_flow_umi_shelf0730_h100.yaml": "unet",
    "job_flow_umi_walk_pnp_bottle_h100.yaml": "unet",
}


def test_every_job_builds_the_backbone_it_is_named_for() -> None:
    jobs = sorted((MANIP_FLOW_ROOT / "config").glob("job_flow_umi_*.yaml"))
    assert jobs, "no job configs found"
    # A new job must be listed above rather than defaulting to "probably unet".
    assert {path.name for path in jobs} == set(EXPECTED_JOB_BACKBONE)

    with initialize_config_dir(
        version_base=None, config_dir=str(MANIP_FLOW_ROOT / "config")
    ):
        for job_path in jobs:
            args = list(OmegaConf.load(job_path).tasks[0].args)
            config_name = args[args.index("--config-name") + 1]
            overrides = [
                arg for arg in args if "=" in arg and not arg.startswith("hydra.")
            ]
            config = compose(config_name=config_name, overrides=overrides)
            assert (
                config.policy.backbone == EXPECTED_JOB_BACKBONE[job_path.name]
            ), job_path.name


def test_dit_context_arm_pairs_tokens_with_two_image_frames() -> None:
    # The two halves of the fix are only useful together, and each is a silent
    # no-op if the other is missing: cross-attention over a single frame still has
    # no temporal information, and a second frame pooled into one vector is still
    # averaged away.
    with initialize_config_dir(
        version_base=None, config_dir=str(MANIP_FLOW_ROOT / "config")
    ):
        config = compose(config_name="train_flow_dit_ctx_humi_lerobot_umi_pnp")

    assert config.policy.backbone == "dit"
    assert config.policy.obs_encoder.context_tokens is True
    assert config.task.img_obs_horizon == 2
    # The pooled vector must SURVIVE: it still drives adaLN and the CFG null
    # branch, so the tokens are additive rather than a replacement.
    assert config.policy.obs_encoder.feature_aggregation == "attention_pool_2d"
    # FlowDiT1D's own default (10000) assumes time_embed_scale=1000; paired with
    # this config's 1.0 it would leave most frequency channels at ~0.
    assert config.policy.time_embed_scale == 1.0
    assert config.policy.dit_time_log_scale == 10.0
    # Same data contract as the unet arm, so the comparison means something.
    assert config.shape_meta.action.horizon == 40
    assert config.task.low_dim_obs_horizon == 3
    assert config.task.dataset_frequency / config.task.obs_down_sample_steps == 10.0
    assert config.policy.obs_encoder.finetune_last_n_blocks == 1
    assert config.policy.obs_encoder.share_rgb_model is False


def test_submit_script_resolves_the_job_config(tmp_path: Path) -> None:
    fake_submitter = tmp_path / "md_ai_kit"
    fake_submitter.write_text(
        "#!/usr/bin/env bash\n"
        "[[ \"$1\" == \"submit\" ]]\n"
        "[[ -f \"$2\" ]]\n"
        "grep -q train_flow_unet_humi_lerobot_umi_pnp \"$2\"\n"
    )
    fake_submitter.chmod(0o755)
    env = os.environ.copy()
    env["GH_PAT"] = "test-placeholder-token"
    env["PATH"] = f"{tmp_path}:{env['PATH']}"

    result = subprocess.run(
        ["bash", str(MANIP_FLOW_ROOT / "scripts" / "submit_flow_umi_pnp.sh")],
        cwd=MANIP_FLOW_ROOT.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


class _FakeEncoder(nn.Module):
    def output_shape(self) -> torch.Size:
        return torch.Size([16])


def test_humi_unet_config_routes_time_scale_to_unet() -> None:
    # Given: the action contract and HuMI time-embedding parameters.
    shape_meta = {"action": {"shape": [20], "horizon": 24}}

    # When: the policy constructs its configured UNet.
    policy = FlowTimmPolicy(
        shape_meta=shape_meta,
        obs_encoder=_FakeEncoder(),
        backbone="unet",
        down_dims=(32, 64, 128),
        diffusion_step_embed_dim=32,
        time_embed_scale=1.0,
        unet_time_log_scale=10.0,
    )

    # Then: the UNet's machine-consumed positional embedding uses HuMI's scale.
    assert policy.model.diffusion_step_encoder[0].log_scale == 10.0


def test_dataset_path_override_reaches_lerobot_dataset() -> None:
    # Given: a deployment-local dataset override distinct from the environment.
    with initialize_config_dir(
        version_base=None,
        config_dir=str(MANIP_FLOW_ROOT / "config"),
    ):
        # When: the standard task-level Hydra override is composed.
        config = compose(
            config_name="train_flow_unet_humi_lerobot_umi_pnp",
            overrides=["task.dataset_path=/tmp/cli-dataset"],
        )

    # Then: the LeRobot dataset receives that override rather than the env value.
    assert config.task.dataset_path == "/tmp/cli-dataset"
    assert config.task.dataset.dataset_path == "/tmp/cli-dataset"

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

    # Then: DINOv2 uses clean integer sampling on the 30 Hz LeRobot timeline.
    assert config.policy.obs_encoder.model_name == "vit_base_patch14_dinov2.lvd142m"
    assert config.policy.obs_encoder.pretrained is True
    assert config.policy.obs_encoder.frozen is False
    assert config.policy.obs_encoder.share_rgb_model is True
    assert config.training.freeze_encoder is False
    assert config.training.num_epochs == 200
    assert config.task.dataset_frequency == 30.0
    assert config.task.obs_down_sample_steps == 2
    assert config.task.dataset_frequency / config.task.obs_down_sample_steps == 15.0
    assert config.task.img_obs_horizon == 1
    assert config.task.low_dim_obs_horizon == 3
    assert config.shape_meta.action.horizon == 36
    assert config.shape_meta.action.down_sample_steps == 2
    assert config.policy.rtc_execution_horizon == 12
    assert config.policy.rtc_max_guidance_weight == 5.0
    assert config.policy.rtc_prefix_schedule == "exp"


def test_humi_unet_job_uses_the_shared_dino_config() -> None:
    # Given: the production multi-GPU job definition.
    job_config = OmegaConf.load(MANIP_FLOW_ROOT / "config" / "job_flow_umi_pnp_baidu4090.yaml")

    # When: its Hydra command-line arguments are inspected.
    args = list(job_config.tasks[0].args)
    config_name_index = args.index("--config-name") + 1

    # Then: production launches the HuMI-aligned shared-DINO policy for 200 epochs.
    assert args[config_name_index] == "train_flow_unet_humi_lerobot_umi_pnp"
    assert "training.num_epochs=200" in args
    assert "wandb==0.15.8" in job_config.pip_packages


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
    shape_meta = {"action": {"shape": [20], "horizon": 36}}

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

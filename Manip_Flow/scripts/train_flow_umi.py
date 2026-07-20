"""Train the flow-matching UMI bimanual policy.

Thin launcher around the vendored ``TrainDiffusionUnetImageWorkspace`` (which is
policy-agnostic: it hydra-instantiates ``cfg.policy`` and only calls
``policy(batch)`` / ``policy.predict_action``). It wires up:

  * sys.path: this repo root, so the ``pipeline.Manip_Flow...`` hydra targets
    resolve. The ``universal_manipulation_interface`` repo is NO LONGER needed:
    the ~30 ``diffusion_policy`` modules + ``umi`` helpers this stack uses were
    vendored into ``pipeline/Manip_Flow`` (common/model/policy/workspace/...).
  * ``task: umi_bimanual`` resolves from the LOCAL ``config/task/`` dir (also
    vendored), so no external hydra.searchpath injection is required.

Usage (on the training box, from anywhere):
    python pipeline/Manip_Flow/scripts/train_flow_umi.py \
        --config-name train_flow_lerobot_umi_pnp \
        task.dataset_path=/path/to/lerobot_dataset_root \
        training.debug=True    # smoke first

Same overrides as UMI's train.py apply (dataloader.batch_size=..., etc.).
"""

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hydra  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

# allows arbitrary python code execution in configs using the ${eval:''} resolver
# (umi_bimanual.yaml uses it for latency_steps)
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).resolve().parent.parent / "config"),
    config_name="train_flow_unet_umi_bimanual_workspace",
)
def main(cfg):
    # resolve immediately so all the ${now:} resolvers use the same time
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()

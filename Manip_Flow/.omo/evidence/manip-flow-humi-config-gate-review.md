# Manip_Flow HuMI Config Gate Re-review

recommendation: REJECT

blockers:

- violatedCriterion: C6-old-config-behavior
  evidencePointer: `Manip_Flow/tests/test_timm_obs_encoder_transforms.py:37-61,139-149`; reproduced command `PYTHONPATH=/home/eason_er/nyx/Motion_Prior_Manipualtion/pipeline/UMI-Diffusion conda run --no-capture-output -n motion_prior pytest -q Manip_Flow/tests`
  observation: The strengthened regression suite expects legacy `imagenet_norm=True` to preserve the old preprocessing behavior and a new `normalize_rgb=True` option to enable normalization, but current `TimmObsEncoder.__init__` has no `normalize_rgb` parameter. Four of seven tests fail with `TypeError`.

- violatedCriterion: C4-new-HuMI-preprocessing
  evidencePointer: `Manip_Flow/config/train_flow_unet_humi_lerobot_umi_pnp.yaml:35-57`; `Manip_Flow/model/vision/timm_obs_encoder.py:64-75,183-201`; same full-suite transcript
  observation: The new config still selects only `imagenet_norm: True`, while the current strengthened test contract selects `normalize_rgb=True` for actual normalization. Production and configuration have not been updated together, so the intended new-only normalization/legacy-preservation split is not constructible.

## originalIntent

Add a dedicated Manip_Flow configuration that reproduces HuMI's active flow-matching ConditionalUnet1D + DINOv2 ViT-B/14 + CLS architecture, while retaining Manip's LeRobot dataset adapter, 20-D action, horizon 12/downsample 3 contract, and behavior of existing configurations.

## desiredOutcome

The new Hydra config composes and constructs an encoder/policy that accepts the declared 224x224 images, applies DINOv2/ImageNet preprocessing in train and eval, emits CLS features, and drives a 20-D/12-step HuMI-style flow UNet. Existing configs and ResNet/DiT/UNet defaults continue to behave as before.

## userOutcomeReview

REJECT. The earlier 224-vs-518 DINOv2 defect is fixed: a real configured DINOv2 encoder was constructed with `patch_embed.img_size == (224, 224)` and `output_shape()` completed with output `(1, 3148)`. Hydra also resolves action `(20,)`, horizon `12`, and downsample `3`. However, the current worktree is internally inconsistent at the normalization compatibility seam. The current production implementation changes every legacy config using `imagenet_norm: True` by applying normalization, while the newly strengthened tests explicitly require the legacy flag to remain behavior-preserving and introduce `normalize_rgb` as the opt-in. Production does not accept that option, the new config does not set it, and the full test suite fails.

## requirementEvidence

- C1-new-config: PASS. `config/train_flow_unet_humi_lerobot_umi_pnp.yaml` exists and composes.
- C2-LeRobot-contract: PASS. Hydra resolves `LeRobotUmiDataset`, action shape `[20]`, horizon `12`, and observation downsample `3`.
- C3-HuMI-UNet-time-semantics: PASS. Config selects UNet, unit flow-time scale, and UNet sinusoidal log scale `10`; constructor routing test passes.
- C4-DINOv2-CLS-and-preprocessing: FAIL at preprocessing. Real DINOv2 accepts 224 and `feature_aggregation: null` selects CLS, but the new/legacy normalization contract is not implementable in the current code/config pair.
- C5-224-image-runtime: PASS. Direct real-backbone `output_shape()` completed; model image size was `(224, 224)` and ViT grid `(16, 16)`.
- C6-old-config-behavior: FAIL. Legacy `imagenet_norm=True` now changes evaluation inputs; the explicit compatibility regression cannot even construct because `normalize_rgb` is absent.
- C7-no-overreach: NOTE. The image-size fix is scoped and necessary. The normalization change is small but violates C6 in its present form.

## reproducedChecks

- Focused suite before concurrent test strengthening: `6 passed`.
- Current full `Manip_Flow/tests`: `3 passed, 4 failed`; every failure is `TypeError: __init__() got an unexpected keyword argument 'normalize_rgb'`.
- Real configured DINOv2, `pretrained=False`, shared backbone: PASS; `output_shape=(1,3148)`, model input `(224,224)`, CLS aggregation, eval Normalize present.
- `git diff --check`: PASS.

## slopAndProgrammingReview

Direct remove-ai-slops pass found no deletion-only, requested-removal, tautological, or implementation-mirroring algorithm tests. The image-size routing test checks a machine-consumed constructor argument and the eval-normalization test checks observable encoder output. The compatibility test is useful rather than excessive because old-config behavior is an explicit criterion. No needless production extraction or parsing was added.

Direct programming pass found the current failure is a contract/integration mismatch, not a style preference: tests and config/constructor API disagree. The configurable UNet scale plumbing remains minimal and backward-compatible by default. Existing module size/type-style issues predate this bounded change and are notes, not blockers.

The code-review report at `/home/eason_er/nyx/Motion_Prior_Manipualtion/.omo/evidence/manip-flow-humi-code-review.md` explicitly covers both required skill perspectives and overfit/slop categories, but it predates the current encoder production diff and strengthened tests. Its approval therefore does not verify this worktree state.

## checkedArtifactPaths

- `/home/eason_er/nyx/Motion_Prior_Manipualtion/pipeline/UMI-Diffusion/Manip_Flow/config/train_flow_unet_humi_lerobot_umi_pnp.yaml`
- `/home/eason_er/nyx/Motion_Prior_Manipualtion/pipeline/UMI-Diffusion/Manip_Flow/model/diffusion/positional_embedding.py`
- `/home/eason_er/nyx/Motion_Prior_Manipualtion/pipeline/UMI-Diffusion/Manip_Flow/model/diffusion/conditional_unet1d.py`
- `/home/eason_er/nyx/Motion_Prior_Manipualtion/pipeline/UMI-Diffusion/Manip_Flow/model/vision/timm_obs_encoder.py`
- `/home/eason_er/nyx/Motion_Prior_Manipualtion/pipeline/UMI-Diffusion/Manip_Flow/policy/flow_timm_policy.py`
- `/home/eason_er/nyx/Motion_Prior_Manipualtion/pipeline/UMI-Diffusion/Manip_Flow/tests/test_humi_unet_lerobot_config.py`
- `/home/eason_er/nyx/Motion_Prior_Manipualtion/pipeline/UMI-Diffusion/Manip_Flow/tests/test_timm_obs_encoder_transforms.py`
- `/home/eason_er/nyx/Motion_Prior_Manipualtion/HuMI/humi_high_level_policy/train.sh`
- `/home/eason_er/nyx/Motion_Prior_Manipualtion/HuMI/humi_high_level_policy/diffusion_policy/config/train_flow_matching_unet_timm_umi_workspace.yaml`
- `/home/eason_er/nyx/Motion_Prior_Manipualtion/HuMI/humi_high_level_policy/diffusion_policy/policy/flow_matching_unet_timm_policy.py`
- `/home/eason_er/nyx/Motion_Prior_Manipualtion/.omo/evidence/manip-flow-humi-code-review.md`
- `/home/eason_er/nyx/Motion_Prior_Manipualtion/.omo/evidence/manual_qa/humi-unet-lerobot-config-manual-qa.md`

## exactEvidenceGaps

- No full pretrained-weight training run was reproduced; this is not needed to establish the current constructor/test blocker.
- The current manual-QA and code-review reports predate the latest normalization compatibility test changes.
- The configured dataset path was not opened; Hydra-resolved metadata and constructor behavior were sufficient for the stated contract checks.

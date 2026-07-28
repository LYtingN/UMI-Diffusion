# Manip_Flow final manual QA

QA executed against the current on-disk worktree at `/home/eason_er/nyx/Motion_Prior_Manipualtion/pipeline/UMI-Diffusion`, using `/home/eason_er/miniconda3/envs/motion_prior/bin/python` (torch 2.4.1+cu121, timm 1.0.28). No product files were edited.

## manualQa

### surfaceEvidence

| scenario id | criterion reference | surface | exact invocation | verdict | artifactRefs |
|---|---|---|---|---|---|
| S1 | Full Manip_Flow regression suite | pytest CLI | `set -o pipefail; /home/eason_er/miniconda3/envs/motion_prior/bin/python -m pytest -q Manip_Flow/tests 2>&1 \| tee Manip_Flow/.omo/evidence/final-manual-qa/full-manip-flow-tests-current.txt` | PASS (9 passed) | A1 |
| S2 | Hydra compose resolves configured UMI-PNP dataset and CLI override | Python/Hydra CLI with `UMI_PNP_DATASET_PATH=/home/eason_er/nyx/Motion_Prior_Manipualtion/dataset/umi_diffusion/umi-pnp-table`; compose base config and `overrides=['task.dataset_path=/tmp/cli-dataset']`; assert nested route, action `(20,)`, horizon `12`, 10 inference steps, UNet log scale `10.0`, DINOv2 model, `normalize_rgb=True`, and `logging.mode=disabled` | PASS | A2 |
| S3 | Real non-pretrained configured DINOv2 + policy + conditional sample | Python CLI; compose config, register Hydra `eval` resolver, instantiate `TimmObsEncoder` with configured `vit_base_patch14_dinov2.lvd142m` and override `pretrained=False`, construct configured UNet `FlowTimmPolicy`, encode real-shaped synthetic observations, call `conditional_sample` with default configured 10 steps | PASS; encoder output `(1,3148)`, sample `(1,12,20)`, finite | A3 |
| S4 | Explicit RGB normalization and legacy compatibility | `/home/eason_er/miniconda3/envs/motion_prior/bin/python -m pytest -q Manip_Flow/tests/test_timm_obs_encoder_transforms.py::test_legacy_imagenet_norm_flag_preserves_existing_preprocessing Manip_Flow/tests/test_timm_obs_encoder_transforms.py::test_imagenet_normalization_runs_during_eval` | PASS (2 passed) | A4 |
| S5 | Real non-ViT encoder construction | Python CLI; instantiate `TimmObsEncoder(model_name='convnext_base', pretrained=False, global_pool='', feature_aggregation='avg', downsample_ratio=32)` on `[3,224,224]` input and run `output_shape()` plus forward | PASS; output `(1,1024)`, finite | A8 |

### adversarialCases

| scenario id | criterion reference | adversarial class | expected behavior | verdict | artifactRefs |
|---|---|---|---|---|---|
| ADV1 | Dataset path environment contract | missing required environment | Hydra must reject composition with a clear missing `UMI_PNP_DATASET_PATH` error | PASS; `InterpolationResolutionError` names missing variable | A5 |
| ADV2 | UNet horizon contract | invalid action horizon | Constructing UNet policy with horizon `10` (not divisible by 4 for 3 levels) must fail with actionable `ValueError` | PASS | A6 |
| ADV3 | Conditional inpainting semantics | all entries conditioned | `conditional_sample` must preserve fully masked condition exactly through all configured steps | PASS; output `(1,12,20)`, max error `0.0` | A7 |

### artifactRefs

| id | kind | description | path |
|---|---|---|---|
| A1 | terminal transcript | Current full Manip_Flow pytest run (`9 passed`) | `/home/eason_er/nyx/Motion_Prior_Manipualtion/pipeline/UMI-Diffusion/Manip_Flow/.omo/evidence/final-manual-qa/full-manip-flow-tests-current.txt` |
| A2 | terminal transcript | Hydra compose assertions with dataset env and privacy-default logging mode | `/home/eason_er/nyx/Motion_Prior_Manipualtion/pipeline/UMI-Diffusion/Manip_Flow/.omo/evidence/final-manual-qa/hydra-compose-current.txt` |
| A3 | terminal transcript | Real DINOv2 (`pretrained=False`) encoder/policy construction and 10-step conditional sample | `/home/eason_er/nyx/Motion_Prior_Manipualtion/pipeline/UMI-Diffusion/Manip_Flow/.omo/evidence/final-manual-qa/real-dinov2-policy-sample.txt` |
| A4 | terminal transcript | Explicit `normalize_rgb` and legacy `imagenet_norm` regression tests | `/home/eason_er/nyx/Motion_Prior_Manipualtion/pipeline/UMI-Diffusion/Manip_Flow/.omo/evidence/final-manual-qa/normalization-compatibility.txt` |
| A5 | terminal transcript | Missing `UMI_PNP_DATASET_PATH` adversarial composition | `/home/eason_er/nyx/Motion_Prior_Manipualtion/pipeline/UMI-Diffusion/Manip_Flow/.omo/evidence/final-manual-qa/hydra-compose-missing-env.txt` |
| A6 | terminal transcript | Invalid UNet horizon rejection | `/home/eason_er/nyx/Motion_Prior_Manipualtion/pipeline/UMI-Diffusion/Manip_Flow/.omo/evidence/final-manual-qa/unet-invalid-horizon.txt` |
| A7 | terminal transcript | Fully conditioned conditional-sample exactness | `/home/eason_er/nyx/Motion_Prior_Manipualtion/pipeline/UMI-Diffusion/Manip_Flow/.omo/evidence/final-manual-qa/conditional-mask-adversarial.txt` |
| A8 | terminal transcript | Real ConvNeXt construction and forward smoke | `/home/eason_er/nyx/Motion_Prior_Manipualtion/pipeline/UMI-Diffusion/Manip_Flow/.omo/evidence/final-manual-qa/real-convnext-construction.txt` |

## Overall verdict

PASS. All requested current-state scenarios passed, including the privacy-default (`logging.mode=disabled`), 10-step configuration, UNet positional log scale `10`, legacy normalization behavior, real non-pretrained DINOv2 construction, and finite `(B,12,20)` conditional output.

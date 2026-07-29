# Shared-DINO goal gate review

- recommendation: APPROVE
- confidence: HIGH
- reviewed HEAD: `01dc0173b1efd00c9195ac7d31af0200e212bd64`
- reviewed state: current unstaged diff in four files, including the corrected submit script

## Original intent

Convert Manip_Flow's HuMI-aligned LeRobot training configuration to use one
shared DINOv2 instance for both cameras, train both DINO and UNet, preserve the
LeRobot observation/action contract, produce exactly 172,392,596 total and
trainable parameters, and make the production 16-GPU job actually launch that
configuration. Also inspect the HuMI training perimeter for deterministic
failures.

## Desired outcome

A locally runnable production submission entry whose 16-GPU job definition
resolves the HuMI LeRobot UNet configuration, constructs one shared trainable
DINOv2 plus a trainable UNet with 172,392,596 total/trainable parameters, and
retains the existing two-camera LeRobot structure and 20-dimensional action
contract. External submission, commit, and push are outside the user's
authorization.

## User outcome review

The local model/configuration outcome is correct and was reproduced directly:

- Hydra config resolves two RGB keys and action shape `[20]`.
- A real policy instantiated with the configured architecture (pretrained
  download disabled only for the local constructor probe) reported
  `total 172392596`, `trainable 172392596`, camera keys
  `camera0_rgb,camera1_rgb`, and exactly one unique encoder object.
- `frozen: False`, `freeze_encoder: False`, and the workspace optimizer includes
  all `requires_grad` encoder parameters plus all UNet parameters.
- The complete Manip_Flow test directory passes: `12 passed in 2.45s`.
- The production definition requests `num_gpus: 16`, selects
  `train_flow_unet_humi_lerobot_umi_pnp`, and now pins
  `wandb==0.15.8`.
- The real shell submission entry was executed with a controlled fake
  `md_ai_kit`; it resolved the repository-local job file, passed it to
  `md_ai_kit submit`, and that job file contained the shared-DINO config name.
- The previous deterministic submit-path defect is corrected:
  `REPO_ROOT` now resolves two levels above the script and `TEMPLATE` resolves
  to `Manip_Flow/config/job_flow_umi_pnp_baidu4090.yaml`.

## Blockers

None.

## Criterion results

- `C1_SHARED_DINOV2_FOR_TWO_CAMERAS`: PASS. Direct instantiation showed two
  camera keys and one unique encoder object.
- `C2_DINO_AND_UNET_TRAINABLE`: PASS. Direct count showed all 172,392,596
  parameters trainable; config and optimizer path agree.
- `C3_EXACT_PARAMETER_COUNT_172392596`: PASS. Reproduced from a real model
  constructor, not inferred from YAML.
- `C4_PRESERVE_LEROBOT_STRUCTURE_AND_ACTION_DIMENSION`: PASS. Dataset target is
  `LeRobotUmiDataset`, two RGB keys remain, and action shape remains `[20]`.
- `C5_LOCAL_16GPU_SUBMISSION_ENTRY_ROUTES_SHARED_DINO_CONFIG`: PASS. The job
  requests 16 GPUs, selects the shared-DINO config, and the actual shell entry
  was executed against a fake submitter that verified the passed file exists
  and contains that config name.
- `C6_CHECK_DETERMINISTIC_HUMI_TRAINING_PERIMETER_BUGS`: PASS for the concrete
  failures found. The job installs the exact `wandb==0.15.8` dependency, and
  the submit script's deterministically incorrect repository/config path was
  corrected and covered by execution.

## Slop/overfit and programming pass

Direct pass:

- No deletion-only or requested-removal test was added.
- The configuration test checks machine-consumed Hydra/job fields. The submit
  regression executes the real shell script and substitutes only the external
  platform binary, making it a valid boundary test rather than a tautology.
- No production parsing, normalization, extraction, or abstraction was added.
- The YAML-only production changes preserve the existing LeRobot/action
  contract and add no source-code maintenance burden.

The code-review reports explicitly state that they consulted both
`omo:remove-ai-slops` and `omo:programming` and covers tautological,
implementation-mirroring, deletion-only, unnecessary parsing/normalization,
and abstraction concerns. The newest report predates the submit-script addition
in its declared scope, so this gate directly inspected that small shell/test
diff and found no overfit or maintenance blocker.

## Checked artifacts

- `Manip_Flow/config/train_flow_unet_humi_lerobot_umi_pnp.yaml`
- `Manip_Flow/config/job_flow_umi_pnp_baidu4090.yaml`
- `Manip_Flow/tests/test_humi_unet_lerobot_config.py`
- `Manip_Flow/scripts/submit_flow_umi_pnp.sh`
- `.omo/evidence/final_code_review-code-review.md`
- `.omo/evidence/shared_dino-code-review.md`
- current `git diff`, `HEAD`, and `origin/main`
- direct model-construction/count output
- targeted pytest output

## Exact evidence gaps

None within the authorized local-implementation scope. The work remains
uncommitted/unpushed, and no external job was submitted; these are delivery
notes rather than failed criteria because the user did not authorize those
actions.

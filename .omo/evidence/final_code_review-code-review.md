# Final code-quality re-review

## Result

- `codeQualityStatus`: WATCH
- `recommendation`: APPROVE
- Scope reviewed: live on-disk changes in `Manip_Flow` plus the untracked HuMI config/tests.

## Findings

### CRITICAL

None.

### HIGH

None. The previous ConvNeXt blocker is fixed: `img_size` is now ViT-only at `Manip_Flow/model/vision/timm_obs_encoder.py:102-117`; both the targeted regression at `Manip_Flow/tests/test_timm_obs_encoder_transforms.py:120-146` and a real installed-timm constructor probe pass.

### MEDIUM

1. **Changed production module remains over the consulted skills' size ceiling.**
   `Manip_Flow/model/vision/timm_obs_encoder.py` is 332 pure lines (303 at `HEAD`). This predates the bounded change and is not a demonstrated regression, but both consulted skill perspectives treat it as a maintainability trigger. Track a focused responsibility-based extraction or documented exception separately.

2. **Normalization coverage does not explicitly pin the training path.**
   `Manip_Flow/model/vision/timm_obs_encoder.py:189-203,353-357` applies normalization in both branches, while `Manip_Flow/tests/test_timm_obs_encoder_transforms.py:149-165` asserts the evaluation branch only. The current code is correct by inspection and the new configuration is covered; a narrow training-mode ordering test would reduce future regression risk.

### LOW

1. **Untracked `.debug-journal.md` is scope noise and contains unlinked success claims.** Remove it from the deliverable or replace it with evidence-backed material. It was not used as evidence for this review.

## Skill-perspective check

Ran: `omo:remove-ai-slops` and `omo:programming` were explicitly loaded and applied.

- No deletion-only, requested-removal, tautological, brittle prompt, or implementation-constant-mirroring tests were found. The constructor, configuration-routing, and normalization assertions exercise machine-consumed behavior.
- No needless production parsing/normalization, untyped escape hatch, or needless abstraction was added. The module-size and training-coverage concerns above remain as non-blocking maintenance items.

## Validation

- PASS: `PYTHONPATH="$PWD" conda run --no-capture-output -n motion_prior pytest -q Manip_Flow/tests` — 9 passed.
- PASS: real installed-timm construction of `convnext_base` without `img_size` and DINOv2 ViT-B/14 with `img_size=224`.
- PASS: dataset override is routed from `task.dataset_path` into `task.dataset.dataset_path` by the dedicated configuration test.
- PASS: `git diff --check`.

## Blockers

None.

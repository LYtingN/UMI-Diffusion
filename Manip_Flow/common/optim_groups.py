"""AdamW parameter groups for the flow policy.

The pretrained vision backbone wants a reduced lr; everything else -- the
velocity model, the randomly-initialized aggregation head, and any module the
policy hangs off itself (``cond_proj``, ``null_cond``) -- wants the base lr.

The reason this is a function and not four lines inline: the previous inline
version enumerated ``policy.model`` and ``policy.obs_encoder`` only, so a new
trainable module on the policy landed in NO group and was silently never
updated. ``build_param_groups`` asserts the groups partition every trainable
tensor, which turns that class of mistake into a startup failure.

Kept out of the workspace module so it imports without ``wandb``/``accelerate``.
"""

from __future__ import annotations

from typing import Any, Dict, List

BACKBONE_PREFIX = 'key_model_map.'


def build_param_groups(
    policy: Any, backbone_lr: float
) -> List[Dict[str, Any]]:
    """Groups for ``torch.optim.AdamW``, in a fixed order.

    0: velocity model, 1: pretrained vision backbone @ ``backbone_lr``,
    2: aggregation head, 3 (only when non-empty): everything else on the policy.
    Groups without an ``lr`` key inherit the optimizer's base lr.
    """
    velocity_params = [
        p for p in policy.model.parameters() if p.requires_grad
    ]

    backbone_params = []
    head_params = []
    for name, param in policy.obs_encoder.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(BACKBONE_PREFIX):
            backbone_params.append(param)
        else:
            head_params.append(param)

    # Frozen encoder tensors go in `claimed` too, so they are not mistaken for
    # unclaimed trainable ones below.
    claimed = {id(p) for p in velocity_params}
    claimed.update(id(p) for p in policy.obs_encoder.parameters())
    other_params = [
        p for p in policy.parameters()
        if p.requires_grad and id(p) not in claimed
    ]

    groups: List[Dict[str, Any]] = [
        {'params': velocity_params},
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params},
    ]
    if other_params:
        groups.append({'params': other_params})

    # The groups must PARTITION the trainable tensors. Too few and something
    # never gets an update; too many means one tensor is in two groups, so AdamW
    # steps it twice per iteration -- at two different learning rates.
    n_grouped = sum(len(group['params']) for group in groups)
    n_trainable = sum(1 for p in policy.parameters() if p.requires_grad)
    if n_grouped != n_trainable:
        problem = (
            'some tensor is in two groups and would be updated twice per step'
            if n_grouped > n_trainable
            else 'some parameters would never be updated'
        )
        raise AssertionError(
            f'optimizer groups cover {n_grouped} trainable tensors but the '
            f'policy has {n_trainable}; {problem}'
        )
    return groups

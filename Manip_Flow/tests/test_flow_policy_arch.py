"""The 2026-08-08 architecture/training changes, each pinned to its own claim.

These changes were made together, so a regression in any one of them would
otherwise surface only as "the run is worse than last week":

  cond bottleneck        -- fewer FiLM parameters, same temporal capacity
  horizon loss weights   -- de-emphasize the chunk tail deploy never executes
  cond dropout + CFG     -- an unconditional branch to guide away from
  optimizer param groups -- every trainable tensor lands in exactly one group
  context cross-attention -- the DiT reads observation TOKENS, not just a pooled
                            vector, and the unconditional branch is blind to
                            both pathways

Instantiating the real TimmObsEncoder would download DINOv3, so these use a stub
encoder with the same surface (``output_shape``, ``forward_features``,
``context_shape``, and the ``key_model_map`` naming build_param_groups keys off).
The parts that need the real encoder live in test_timm_obs_encoder_transforms.py
and test_timm_obs_encoder_context.py.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from Manip_Flow.common.optim_groups import build_param_groups  # noqa: E402
from Manip_Flow.model.common.normalizer import (  # noqa: E402
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from Manip_Flow.policy import rtc_flow  # noqa: E402
from Manip_Flow.policy.flow_timm_policy import FlowTimmPolicy  # noqa: E402

HORIZON = 8
ACTION_DIM = 20
FEATURE_DIM = 64
CONTEXT_TOKENS = 5


class _StubEncoder(nn.Module):
    def __init__(self, context_tokens: bool = False):
        super().__init__()
        # 'key_model_map.' is the prefix build_param_groups routes to the
        # reduced-lr group, so the stub has to use the real name.
        self.key_model_map = nn.ModuleDict({"state": nn.Linear(3, FEATURE_DIM)})
        # Stands in for attention_pool_2d: randomly initialized, so it must NOT
        # end up in the reduced-lr group.
        self.pool_head = nn.Linear(FEATURE_DIM, FEATURE_DIM)
        self.context_tokens = bool(context_tokens)
        self.context_dim = FEATURE_DIM if self.context_tokens else 0
        if self.context_tokens:
            self.token_head = nn.Linear(FEATURE_DIM, CONTEXT_TOKENS * FEATURE_DIM)

    def output_shape(self):
        return (FEATURE_DIM,)

    def context_shape(self):
        if not self.context_tokens:
            return None
        return (CONTEXT_TOKENS, FEATURE_DIM)

    def forward_features(self, obs_dict):
        state = obs_dict["state"]
        feature = self.key_model_map["state"](state)
        pooled = self.pool_head(feature).reshape(state.shape[0], -1)
        context = None
        if self.context_tokens:
            context = self.token_head(feature).reshape(
                state.shape[0], CONTEXT_TOKENS, FEATURE_DIM
            )
        return pooled, context

    def forward(self, obs_dict):
        return self.forward_features(obs_dict)[0]


def _policy(context: bool = False, **kwargs) -> FlowTimmPolicy:
    """A tiny FlowTimmPolicy with an identity normalizer already installed."""
    shape_meta = {
        "action": {"shape": (ACTION_DIM,), "horizon": HORIZON},
        "obs": {"state": {"shape": (3,), "type": "low_dim", "horizon": 1}},
    }
    defaults = dict(
        backbone="unet",
        down_dims=(32, 64),
        diffusion_step_embed_dim=16,
        num_inference_steps=2,
    )
    if context:
        # Context tokens only mean anything to a backbone that cross-attends.
        defaults.update(backbone="dit", dit_d_model=32, dit_depth=1, dit_n_heads=4)
    defaults.update(kwargs)
    policy = FlowTimmPolicy(
        shape_meta=shape_meta,
        obs_encoder=_StubEncoder(context_tokens=context),
        **defaults,
    )
    # Identity rather than fitted, so two policies built in one test share the
    # exact same normalization and only the change under test differs.
    normalizer = LinearNormalizer()
    for key in ("state", "action"):
        normalizer[key] = SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(normalizer)
    return policy


def _batch(batch_size: int = 4) -> dict:
    return {
        "obs": {"state": torch.randn(batch_size, 1, 3)},
        "action": torch.randn(batch_size, HORIZON, ACTION_DIM),
    }


# ---------------------------------------------------------------- loss weights

def test_horizon_weight_is_flat_then_linear_and_averages_to_one() -> None:
    # Given: weight 1.0 through token 20 of 40, tapering to 0.25 at the end.
    weight = FlowTimmPolicy.build_horizon_loss_weight(40, 20, 0.25)

    # Then: the plateau is flat, the tail decreases, and the ratio is exactly 4x.
    assert torch.all(weight[:20] == weight[0])
    assert torch.all(weight[20:].diff() < 0.0)
    torch.testing.assert_close(weight[0] / weight[-1], torch.tensor(4.0))
    # Renormalized to mean 1.0 so the loss keeps its magnitude. This is what
    # makes val_loss comparable in SCALE (not in meaning) to the uniform runs.
    torch.testing.assert_close(weight.mean(), torch.tensor(1.0))


def test_horizon_weight_is_all_ones_at_the_legacy_defaults() -> None:
    # Given: what an old config supplies (no full_steps, no taper).
    # Then: the weighting is identity, so old runs keep their old loss.
    torch.testing.assert_close(
        FlowTimmPolicy.build_horizon_loss_weight(40, 0, 1.0), torch.ones(40)
    )
    torch.testing.assert_close(
        FlowTimmPolicy.build_horizon_loss_weight(40, 40, 0.25), torch.ones(40)
    )


def test_horizon_weight_clamps_full_steps_past_the_horizon() -> None:
    # Given: full_steps larger than the horizon (e.g. horizon cut in config).
    # Then: no taper, rather than an out-of-range slice.
    torch.testing.assert_close(
        FlowTimmPolicy.build_horizon_loss_weight(8, 99, 0.25), torch.ones(8)
    )


def test_horizon_weight_is_not_persisted_in_the_state_dict() -> None:
    # It is derived from config, so a checkpoint trained before the weighting
    # must still load with strict=True.
    assert "horizon_loss_weight" not in _policy().state_dict()


def test_horizon_weight_actually_reaches_the_loss() -> None:
    # Given: two policies with identical weights, differing only in the taper.
    torch.manual_seed(0)
    flat = _policy()
    torch.manual_seed(0)
    tapered = _policy(loss_horizon_full_steps=2, loss_horizon_tail_weight=0.1)
    tapered.load_state_dict(flat.state_dict())
    batch = _batch()

    torch.manual_seed(1)
    flat_loss = flat.compute_loss(batch)
    torch.manual_seed(1)
    tapered_loss = tapered.compute_loss(batch)

    # Then: the same batch scores differently -- the buffer is not decorative.
    assert not torch.allclose(flat_loss, tapered_loss)


# ------------------------------------------------------------ cond bottleneck

def test_cond_bottleneck_shrinks_film_but_not_temporal_convolution() -> None:
    # Given: the same unet reached through the raw feature vs a 16-dim bottleneck.
    raw = _policy(cond_bottleneck_dim=0)
    squeezed = _policy(cond_bottleneck_dim=16)

    assert raw.global_cond_dim == FEATURE_DIM
    assert squeezed.global_cond_dim == 16

    def split(policy):
        film = conv = 0
        for name, param in policy.model.named_parameters():
            if "cond_encoder" in name:
                film += param.numel()
            else:
                conv += param.numel()
        return film, conv

    raw_film, raw_conv = split(raw)
    squeezed_film, squeezed_conv = split(squeezed)

    # Then: FiLM projection shrinks; every temporal convolution is untouched.
    assert squeezed_film < raw_film
    assert squeezed_conv == raw_conv


def test_cond_bottleneck_off_leaves_the_conditioning_path_an_identity() -> None:
    # The legacy default, which old configs and checkpoints rely on.
    assert isinstance(_policy(cond_bottleneck_dim=0).cond_proj, nn.Identity)


# --------------------------------------------------------- cond dropout / CFG

def test_cond_dropout_replaces_whole_rows_with_the_null_embedding() -> None:
    # Given: a policy whose null embedding is a recognizable constant.
    policy = _policy(cond_dropout_prob=1.0)
    with torch.no_grad():
        policy.null_cond.fill_(7.0)

    dropped, context = policy.drop_conditioning(torch.zeros(5, policy.global_cond_dim))

    # Then: at p=1.0 every row is the null embedding.
    torch.testing.assert_close(dropped, torch.full_like(dropped, 7.0))
    assert context is None  # nothing to drop when the encoder emits no tokens


def test_cond_dropout_is_all_or_nothing_per_sample() -> None:
    # Given: a middling dropout probability over many rows.
    torch.manual_seed(0)
    policy = _policy(cond_dropout_prob=0.5)
    with torch.no_grad():
        policy.null_cond.fill_(7.0)

    dropped, _ = policy.drop_conditioning(torch.ones(256, policy.global_cond_dim))

    # Then: each row is entirely dropped or entirely kept -- never a per-channel
    # mix, which would leak the observation into the unconditional branch.
    is_null = (dropped == 7.0).all(dim=-1)
    is_kept = (dropped == 1.0).all(dim=-1)
    assert torch.all(is_null | is_kept)
    assert is_null.any() and is_kept.any()


def test_no_null_embedding_exists_when_dropout_is_off() -> None:
    # Nothing to learn an unconditional branch from, so nothing is allocated and
    # the state_dict stays byte-compatible with the pre-CFG checkpoints.
    policy = _policy(cond_dropout_prob=0.0)
    assert policy.null_cond is None
    assert policy.null_conditioning(4) is None
    assert "null_cond" not in policy.state_dict()
    # And dropout is a no-op rather than a crash.
    global_cond = torch.ones(3, policy.global_cond_dim)
    assert policy.drop_conditioning(global_cond)[0] is global_cond


def _sample(policy, *, guidance_scale, with_uncond, global_cond):
    uncond = None
    if with_uncond:
        uncond = policy.null_conditioning(global_cond.shape[0])
    data = torch.zeros(global_cond.shape[0], HORIZON, ACTION_DIM)
    return rtc_flow.flow_euler_sample(
        model=policy.model,
        condition_data=data,
        condition_mask=torch.zeros_like(data, dtype=torch.bool),
        global_cond=global_cond,
        generator=torch.Generator().manual_seed(3),
        rtc_action_prefix=None,
        rtc_inference_delay=0,
        uncond_global_cond=uncond,
        guidance_scale=guidance_scale,
        config=rtc_flow.FlowSamplingConfig(
            inference_steps=2,
            time_embed_scale=1.0,
            action_horizon=HORIZON,
            execution_horizon=4,
            max_guidance_weight=5.0,
            prefix_schedule="exp",
        ),
    )


def test_guidance_scale_one_is_identical_to_no_guidance_at_all() -> None:
    # Given: one set of weights, one noise seed, and an unconditional branch
    # that IS available.
    torch.manual_seed(0)
    policy = _policy(cond_dropout_prob=0.1)
    with torch.no_grad():
        policy.null_cond.normal_()
    global_cond, _ = policy.encode_obs({"state": torch.randn(2, 1, 3)})

    # Then: scale 1.0 reproduces the unguided sample exactly, so turning CFG on
    # in config cannot change a deployed checkpoint until the scale is raised.
    torch.testing.assert_close(
        _sample(policy, guidance_scale=1.0, with_uncond=True,
                global_cond=global_cond),
        _sample(policy, guidance_scale=1.0, with_uncond=False,
                global_cond=global_cond),
    )


def test_guidance_scale_above_one_changes_the_sample() -> None:
    torch.manual_seed(0)
    policy = _policy(cond_dropout_prob=0.1)
    with torch.no_grad():
        policy.null_cond.normal_()
    global_cond, _ = policy.encode_obs({"state": torch.randn(2, 1, 3)})

    # Guarded explicitly because CFG extrapolation of an all-zero velocity is
    # still zero, so a backbone whose output head is zero-initialized shows no
    # guidance delta at init -- a real effect that looks exactly like a bug.
    assert not torch.allclose(
        _sample(policy, guidance_scale=1.0, with_uncond=True,
                global_cond=global_cond),
        _sample(policy, guidance_scale=3.0, with_uncond=True,
                global_cond=global_cond),
    )


def test_guidance_without_an_unconditional_branch_is_rejected() -> None:
    # Given: guidance requested on a checkpoint trained without cond dropout.
    policy = _policy()
    with pytest.raises(rtc_flow.RTCConfigError, match="cond_dropout_prob"):
        _sample(
            policy,
            guidance_scale=2.0,
            with_uncond=False,
            global_cond=torch.zeros(1, policy.global_cond_dim),
        )


# ------------------------------------------------------- context cross-attention

def test_context_tokens_reach_the_dit_and_change_the_prediction() -> None:
    # Given: a DiT that cross-attends, with the zero-init output gate opened so
    # the pathway is actually exercised (at init the whole block is identity).
    torch.manual_seed(0)
    policy = _policy(context=True)
    block = policy.model.blocks[0]
    assert block.use_cross_attn
    with torch.no_grad():
        block.adaLN[-1].bias.normal_(std=0.5)
        policy.model.final_proj.weight.normal_(std=0.1)

    sample = torch.randn(2, HORIZON, ACTION_DIM)
    time = torch.zeros(2)
    global_cond = torch.randn(2, policy.global_cond_dim)
    context = torch.randn(2, CONTEXT_TOKENS, policy.context_dim)

    baseline = policy.model(sample, time, global_cond=global_cond, context=context)
    perturbed = policy.model(
        sample, time, global_cond=global_cond, context=context + 1.0
    )

    # Then: changing ONLY the tokens changes the velocity. This is the pathway a
    # pooled global_cond cannot provide, so if it were silently ignored the whole
    # arm would be a no-op that still costs 2x the step.
    assert not torch.allclose(baseline, perturbed)


def test_cross_attention_starts_as_a_no_op() -> None:
    # Given: a context DiT at init, where adaLN-zero gates every sublayer shut.
    torch.manual_seed(0)
    policy = _policy(context=True)
    with torch.no_grad():
        # Open the final projection only, so a difference would have to come from
        # the blocks rather than from a zeroed output layer.
        policy.model.final_proj.weight.normal_(std=0.1)

    sample = torch.randn(2, HORIZON, ACTION_DIM)
    args = dict(global_cond=torch.randn(2, policy.global_cond_dim))
    with_context = policy.model(
        sample, torch.zeros(2), context=torch.randn(2, CONTEXT_TOKENS, FEATURE_DIM),
        **args,
    )
    other_context = policy.model(
        sample, torch.zeros(2), context=torch.randn(2, CONTEXT_TOKENS, FEATURE_DIM),
        **args,
    )

    # Then: the context is inert at init, so enabling it cannot destabilize the
    # first steps -- it is learned in through the gate.
    torch.testing.assert_close(with_context, other_context)


def test_dit_dropout_reaches_all_three_sites_and_is_off_in_eval() -> None:
    # dit_dropout is the only regularizer inside the velocity model, so a config
    # knob that silently failed to land would be invisible: training would just
    # overfit as if it were 0. It has to reach the MLP AND both attentions --
    # attention dropout on the context keys is what stops a single patch token
    # from becoming a shortcut.
    policy = _policy(context=True, dit_dropout=0.1)
    block = policy.model.blocks[0]
    assert block.attn.dropout == 0.1
    assert block.cross_attn.dropout == 0.1
    assert isinstance(block.mlp[2], nn.Dropout) and block.mlp[2].p == 0.1

    # Both final_proj and the adaLN gates are zero-init, so a forward at init is
    # exactly 0 and comparing two of them proves NOTHING about dropout. Open both.
    torch.manual_seed(0)
    with torch.no_grad():
        policy.model.final_proj.weight.normal_(std=0.1)
        for block in policy.model.blocks:
            block.adaLN[-1].weight.normal_(std=0.02)
            block.adaLN[-1].bias.normal_(std=0.5)

    sample = torch.randn(2, HORIZON, ACTION_DIM)
    args = dict(
        global_cond=torch.randn(2, policy.global_cond_dim),
        context=torch.randn(2, CONTEXT_TOKENS, FEATURE_DIM),
    )

    def twice() -> tuple:
        with torch.no_grad():
            return (
                policy.model(sample, torch.zeros(2), **args),
                policy.model(sample, torch.zeros(2), **args),
            )

    policy.train()
    first, second = twice()
    assert not torch.equal(first, second), "dropout is not active in train mode"

    # And eval must be exact: val_draw_std_pos_last_m measures how far apart
    # independent SAMPLES land, and dropout noise would inflate it into
    # meaninglessness -- it is the metric the whole context arm is judged on.
    policy.eval()
    first, second = twice()
    assert torch.equal(first, second)


def test_context_is_rejected_by_a_backbone_that_cannot_use_it() -> None:
    # ConditionalUnet1D.forward swallows **kwargs, so a context tensor would be
    # silently discarded: you would pay for 2x the ViT forwards and train without
    # the pathway you paid for.
    with pytest.raises(rtc_flow.FlowPolicyConfigError, match="backbone='unet'"):
        _policy(context=True, backbone="unet", down_dims=(32, 64))

    # And symmetrically, a context-free DiT refuses tokens rather than ignoring them.
    policy = _policy()
    dit = _policy(context=True).model
    with pytest.raises(ValueError, match="context_dim"):
        _policy(backbone="dit", dit_d_model=32, dit_depth=1, dit_n_heads=4).model(
            torch.randn(2, HORIZON, ACTION_DIM),
            torch.zeros(2),
            global_cond=torch.randn(2, policy.global_cond_dim),
            context=torch.randn(2, CONTEXT_TOKENS, FEATURE_DIM),
        )
    del dit


def test_cond_dropout_blinds_both_pathways_on_the_same_rows() -> None:
    # Given: dropout at a middling rate over many rows.
    torch.manual_seed(0)
    policy = _policy(context=True, cond_dropout_prob=0.5)
    with torch.no_grad():
        policy.null_cond.fill_(7.0)
        policy.null_context.fill_(-3.0)

    global_cond = torch.ones(256, policy.global_cond_dim)
    context = torch.ones(256, CONTEXT_TOKENS, policy.context_dim)
    dropped_cond, dropped_context = policy.drop_conditioning(global_cond, context)

    cond_is_null = (dropped_cond == 7.0).all(dim=-1)
    context_is_null = (dropped_context == -3.0).flatten(1).all(dim=-1)

    # Then: exactly the same rows are blinded in both. A row with a null pooled
    # vector but the real patch tokens still sees the observation, so the
    # "unconditional" branch would not be unconditional and CFG would extrapolate
    # along a meaningless direction.
    assert torch.equal(cond_is_null, context_is_null)
    assert cond_is_null.any() and (~cond_is_null).any()


def test_guidance_with_context_needs_a_blinded_context() -> None:
    # Given: guidance requested while cross-attending, but no null_context (i.e.
    # a checkpoint trained without cond dropout).
    policy = _policy(context=True)
    assert policy.null_context is None
    data = torch.zeros(2, HORIZON, ACTION_DIM)

    with pytest.raises(rtc_flow.RTCConfigError, match="uncond_context"):
        rtc_flow.flow_euler_sample(
            model=policy.model,
            condition_data=data,
            condition_mask=torch.zeros_like(data, dtype=torch.bool),
            global_cond=torch.randn(2, policy.global_cond_dim),
            context=torch.randn(2, CONTEXT_TOKENS, policy.context_dim),
            generator=None,
            rtc_action_prefix=None,
            rtc_inference_delay=0,
            uncond_global_cond=torch.randn(2, policy.global_cond_dim),
            uncond_context=None,
            guidance_scale=2.0,
            config=rtc_flow.FlowSamplingConfig(
                inference_steps=2,
                time_embed_scale=1.0,
                action_horizon=HORIZON,
                execution_horizon=4,
                max_guidance_weight=5.0,
                prefix_schedule="exp",
            ),
        )


def test_context_survives_the_n_samples_repeat() -> None:
    # Given: several flow samples per obs-encoder forward, which repeats the
    # conditioning. A context left un-repeated would be a batch-size mismatch, or
    # worse, silently broadcast the wrong observation onto the wrong action.
    policy = _policy(context=True, train_flow_n_samples=4, cond_dropout_prob=0.1)
    loss = policy.compute_loss(_batch(3))
    loss.backward()

    assert torch.isfinite(loss)
    # The context pathway is on the graph, not decoratively attached.
    assert policy.obs_encoder.token_head.weight.grad is not None
    assert policy.model.context_proj[1].weight.grad is not None


def test_context_policy_samples_and_composes_with_rtc() -> None:
    torch.manual_seed(0)
    policy = _policy(context=True, cond_dropout_prob=0.1, cfg_scale=3.0)
    policy.eval()
    obs = {"state": torch.randn(2, 1, 3)}
    with torch.no_grad():
        plain = policy.predict_action(obs)["action"]
        guided = policy.predict_action(
            obs, rtc_action_prefix=torch.randn(2, 4, ACTION_DIM), rtc_inference_delay=1
        )["action"]
    assert plain.shape == (2, HORIZON, ACTION_DIM)
    assert torch.isfinite(plain).all() and torch.isfinite(guided).all()


# ----------------------------------------------------------- optimizer groups

def test_param_groups_cover_every_trainable_tensor_exactly_once() -> None:
    # Given: modules hanging off the policy itself (cond_proj, null_cond) that
    # the old inline grouping -- which enumerated model/obs_encoder only --
    # silently never passed to the optimizer.
    policy = _policy(cond_bottleneck_dim=16, cond_dropout_prob=0.1)
    groups = build_param_groups(policy, backbone_lr=1e-5)

    grouped = [p for group in groups for p in group["params"]]
    ids = {id(p) for p in grouped}
    assert len(ids) == len(grouped)  # no tensor in two groups
    assert ids == {id(p) for p in policy.parameters() if p.requires_grad}
    assert id(policy.null_cond) in ids
    assert id(policy.cond_proj[0].weight) in ids


def test_only_the_vision_backbone_gets_the_reduced_lr() -> None:
    policy = _policy(cond_bottleneck_dim=16)
    groups = build_param_groups(policy, backbone_lr=1e-5)

    # Then: exactly one group carries an explicit lr, and it is the pretrained
    # tower. A randomly-initialized head at 0.1x lr would barely train.
    assert [group.get("lr") for group in groups].count(1e-5) == 1
    assert groups[1]["lr"] == 1e-5
    backbone = {id(p) for p in groups[1]["params"]}
    assert backbone == {
        id(policy.obs_encoder.key_model_map["state"].weight),
        id(policy.obs_encoder.key_model_map["state"].bias),
    }
    # The randomly-initialized pool head trains at the BASE lr, not 0.1x.
    assert id(policy.obs_encoder.pool_head.weight) in {
        id(p) for p in groups[2]["params"]
    }


def test_a_future_policy_level_module_is_swept_into_the_other_group() -> None:
    # Given: a trainable parameter on the policy that predates no group -- i.e.
    # the next cond_proj/null_cond someone adds. The old inline grouping
    # enumerated model/obs_encoder only, so this used to be dropped silently.
    policy = _policy()
    policy.surprise = nn.Parameter(torch.zeros(2))

    groups = build_param_groups(policy, backbone_lr=1e-5)

    # Then: it is grouped without anyone touching build_param_groups, at the
    # base lr, not at the backbone's 0.1x.
    assert id(policy.surprise) in {id(p) for p in groups[-1]["params"]}
    assert "lr" not in groups[-1]


def test_param_groups_raise_when_a_tensor_would_be_counted_twice() -> None:
    # Given: one tensor reachable through both policy.model and
    # policy.obs_encoder (weight tying). AdamW would then see it in two groups
    # and apply two updates per step, at two different learning rates.
    policy = _policy()
    shared = policy.obs_encoder.key_model_map["state"].weight
    policy.model.tied = shared

    with pytest.raises(AssertionError, match="updated twice per step"):
        build_param_groups(policy, backbone_lr=1e-5)


def test_frozen_encoder_blocks_are_not_counted_as_unclaimed() -> None:
    # finetune_last_n_blocks freezes most of the ViT, so the encoder carries
    # requires_grad=False tensors. They must not be mistaken for unclaimed
    # trainable ones (which would raise) nor handed to the optimizer.
    policy = _policy()
    frozen = policy.obs_encoder.key_model_map["state"].weight
    frozen.requires_grad_(False)

    groups = build_param_groups(policy, backbone_lr=1e-5)
    assert id(frozen) not in {id(p) for group in groups for p in group["params"]}

"""TimmObsEncoder's patch-token context sequence.

The pooled ``attention_pool_2d`` vector averages every patch token of every frame
into one 768-vector, which is why two visually similar frames from different task
phases end up with near-identical conditioning. ``context_tokens=True`` also
returns the tokens themselves, for a backbone that cross-attends to them.

The real DINOv3 weights are a download, so these mock ``timm.create_model`` with
a ViT-shaped stub. The token COUNT and the per-(camera, frame) stamping are
arithmetic on the encoder's side, which is exactly what these pin down.
"""

from unittest.mock import patch

import timm
import torch
import torch.nn as nn

from Manip_Flow.model.vision.timm_obs_encoder import TimmObsEncoder

# 64 is the smallest embed dim AttentionPool2d tolerates: it derives its head
# count as feature_dim // 64, so anything smaller gives zero heads.
EMBED_DIM = 64
GRID = 4  # 8x8 image / patch 2 -> 4x4 = 16 patch tokens
N_PATCH = GRID * GRID


class _PatchEmbed:
    patch_size = (2, 2)


class _VitStub(nn.Module):
    """Patch tokens that encode which image they came from, after 1 CLS prefix."""

    num_prefix_tokens = 1
    num_features = EMBED_DIM
    embed_dim = EMBED_DIM
    patch_embed = _PatchEmbed()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch = value.shape[0]
        # Per-image constant so a wrong (camera, frame) ordering is visible.
        per_image = value.mean(dim=(1, 2, 3)).view(batch, 1, 1)
        tokens = per_image.expand(batch, 1 + N_PATCH, EMBED_DIM).clone()
        # Mark the CLS prefix, so slicing it into the spatial map would show up.
        tokens[:, 0, :] = -99.0
        return tokens


def _encoder(*, img_horizon: int, cameras: int, context_tokens: bool):
    shape_meta = {
        "obs": {
            f"camera{index}_rgb": {
                "shape": [3, 8, 8],
                "horizon": img_horizon,
                "type": "rgb",
            }
            for index in range(cameras)
        }
    }
    shape_meta["obs"]["state"] = {"shape": [2], "horizon": 3, "type": "low_dim"}
    with patch.object(timm, "create_model", side_effect=lambda **_: _VitStub()):
        return TimmObsEncoder(
            shape_meta=shape_meta,
            model_name="vit_base_patch16_dinov3.lvd1689m",
            pretrained=False,
            frozen=False,
            global_pool="",
            transforms=None,
            feature_aggregation="attention_pool_2d",
            downsample_ratio=32,
            context_tokens=context_tokens,
        )


def _obs(*, batch: int, img_horizon: int, cameras: int):
    obs = {
        f"camera{index}_rgb": torch.rand(batch, img_horizon, 3, 8, 8)
        for index in range(cameras)
    }
    obs["state"] = torch.randn(batch, 3, 2)
    return obs


def test_context_is_every_patch_of_every_camera_and_frame() -> None:
    # Given: two cameras, two frames -- the img_obs_horizon=2 configuration.
    encoder = _encoder(img_horizon=2, cameras=2, context_tokens=True)

    _, context = encoder.forward_features(
        _obs(batch=3, img_horizon=2, cameras=2)
    )

    # Then: 2 cameras x 2 frames x 16 patches, at the backbone's embed dim.
    assert context.shape == (3, 2 * 2 * N_PATCH, EMBED_DIM)
    # And context_shape() -- which the policy sizes its cross-attention from --
    # agrees with a real forward.
    assert encoder.context_shape() == (2 * 2 * N_PATCH, EMBED_DIM)


def test_the_cls_prefix_is_not_in_the_context() -> None:
    # DINOv3 prepends 1 CLS + 4 register tokens; slicing them into the spatial
    # map would silently feed non-spatial tokens to the cross-attention.
    encoder = _encoder(img_horizon=1, cameras=1, context_tokens=True)
    _, context = encoder.forward_features(_obs(batch=2, img_horizon=1, cameras=1))
    assert context.shape[1] == N_PATCH
    assert not (context == -99.0).any()


def test_each_camera_and_frame_is_stamped_differently() -> None:
    # Given: an encoder fed the IDENTICAL image in both frames of both cameras.
    encoder = _encoder(img_horizon=2, cameras=2, context_tokens=True)
    frame = torch.rand(1, 1, 3, 8, 8)
    obs = {
        "camera0_rgb": frame.repeat(1, 2, 1, 1, 1),
        "camera1_rgb": frame.repeat(1, 2, 1, 1, 1),
        "state": torch.zeros(1, 3, 2),
    }

    _, context = encoder.forward_features(obs)
    groups = context.reshape(1, 2, 2, N_PATCH, EMBED_DIM)  # cam, frame, patch, C

    # Then: the four (camera, frame) groups are still distinguishable. Without
    # this the cross-attention sees an unordered bag of patches and cannot tell
    # "approaching" from "retreating" -- which is the entire reason for feeding
    # more than one frame.
    flat = [groups[0, cam, frame_index] for cam in range(2) for frame_index in range(2)]
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            assert not torch.allclose(flat[i], flat[j]), (i, j)


def test_more_frames_than_shape_meta_declared_is_rejected() -> None:
    # The stamp is sized from shape_meta's horizon. Feeding more frames than that
    # slices a single stamp row, which BROADCASTS onto every frame -- identical
    # frames become indistinguishable again and nothing raises. That is the exact
    # failure this embedding exists to prevent, so it has to be loud.
    encoder = _encoder(img_horizon=1, cameras=1, context_tokens=True)
    frame = torch.rand(1, 1, 3, 8, 8)
    obs = {
        "camera0_rgb": frame.repeat(1, 2, 1, 1, 1),
        "state": torch.zeros(1, 3, 2),
    }

    try:
        encoder.forward_features(obs)
    except AssertionError as error:
        assert "context_source_embed" in str(error)
    else:
        raise AssertionError("2 frames against a horizon-1 stamp was accepted")


def test_context_is_none_and_costs_nothing_when_disabled() -> None:
    # The default path must allocate no embedding and return no tokens, so
    # existing configs and checkpoints are untouched.
    encoder = _encoder(img_horizon=1, cameras=2, context_tokens=False)

    feature, context = encoder.forward_features(
        _obs(batch=2, img_horizon=1, cameras=2)
    )

    assert context is None
    assert encoder.context_shape() is None
    assert encoder.context_dim == 0
    assert not hasattr(encoder, "context_source_embed")
    # forward() keeps its old single-tensor return for every other caller.
    torch.testing.assert_close(
        encoder(_obs(batch=2, img_horizon=1, cameras=2)).shape, feature.shape
    )


def test_pooled_feature_is_unchanged_by_enabling_context() -> None:
    # The tokens are ADDITIVE: the pooled vector still drives adaLN and the CFG
    # null branch, and it must come out of the same backbone forward.
    obs = _obs(batch=2, img_horizon=2, cameras=2)

    torch.manual_seed(0)
    plain = _encoder(img_horizon=2, cameras=2, context_tokens=False)
    torch.manual_seed(0)
    with_context = _encoder(img_horizon=2, cameras=2, context_tokens=True)
    with_context.load_state_dict(plain.state_dict(), strict=False)

    plain.eval()
    with_context.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            plain(obs), with_context.forward_features(obs)[0]
        )

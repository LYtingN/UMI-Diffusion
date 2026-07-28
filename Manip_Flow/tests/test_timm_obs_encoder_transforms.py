from unittest.mock import patch

import timm
import torch
import torch.nn as nn

from Manip_Flow.model.vision.timm_obs_encoder import TimmObsEncoder


class _AddOne(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1


class _FeatureMap(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch_mean = value.mean(dim=(1, 2, 3), keepdim=True)
        return batch_mean.expand(-1, 512, 1, 1)


class _PatchEmbed:
    patch_size = (2, 2)


class _VitFeatureMap(nn.Module):
    num_prefix_tokens = 1
    num_features = 8
    embed_dim = 8
    patch_embed = _PatchEmbed()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch = value.shape[0]
        return torch.zeros(batch, 17, self.num_features)


def _make_encoder(
    *,
    imagenet_norm: bool = False,
    normalize_rgb: bool = False,
) -> TimmObsEncoder:
    shape_meta = {
        "obs": {
            "camera0_rgb": {
                "shape": [3, 8, 8],
                "horizon": 1,
                "type": "rgb",
            }
        }
    }
    fake_backbone = nn.Sequential(_FeatureMap(), nn.Identity(), nn.Identity())
    with patch.object(timm, "create_model", return_value=fake_backbone):
        return TimmObsEncoder(
            shape_meta=shape_meta,
            model_name="resnet18.a1_in1k",
            pretrained=False,
            frozen=False,
            global_pool="",
            transforms=[_AddOne()],
            imagenet_norm=imagenet_norm,
            normalize_rgb=normalize_rgb,
            feature_aggregation="avg",
            downsample_ratio=32,
        )


def test_image_augmentation_runs_during_training() -> None:
    # Given
    encoder = _make_encoder()
    encoder.train()
    obs = {"camera0_rgb": torch.zeros(1, 1, 3, 8, 8)}

    # When
    features = encoder(obs)

    # Then
    torch.testing.assert_close(features, torch.ones_like(features))


def test_image_augmentation_is_disabled_during_eval() -> None:
    # Given
    encoder = _make_encoder()
    encoder.eval()
    obs = {"camera0_rgb": torch.zeros(1, 1, 3, 8, 8)}

    # When
    features = encoder(obs)

    # Then
    torch.testing.assert_close(features, torch.zeros_like(features))


def test_vit_uses_declared_observation_image_size() -> None:
    # Given
    shape_meta = {
        "obs": {
            "camera0_rgb": {
                "shape": [3, 8, 8],
                "horizon": 1,
                "type": "rgb",
            }
        }
    }

    # When
    with patch.object(timm, "create_model", return_value=_VitFeatureMap()) as create:
        TimmObsEncoder(
            shape_meta=shape_meta,
            model_name="vit_base_patch14_dinov2.lvd142m",
            pretrained=False,
            frozen=False,
            global_pool="",
            transforms=None,
            feature_aggregation=None,
        )

    # Then
    assert create.call_args.kwargs["img_size"] == 8


def test_convnext_does_not_receive_vit_image_size_argument() -> None:
    # Given
    shape_meta = {
        "obs": {
            "camera0_rgb": {
                "shape": [3, 8, 8],
                "horizon": 1,
                "type": "rgb",
            }
        }
    }
    fake_backbone = nn.Sequential(*[nn.Identity() for _ in range(4)])

    # When
    with patch.object(timm, "create_model", return_value=fake_backbone) as create:
        TimmObsEncoder(
            shape_meta=shape_meta,
            model_name="convnext_base",
            pretrained=False,
            frozen=False,
            global_pool="",
            transforms=None,
            feature_aggregation="avg",
        )

    # Then
    assert "img_size" not in create.call_args.kwargs


def test_imagenet_normalization_runs_during_eval() -> None:
    # Given
    encoder = _make_encoder(normalize_rgb=True)
    encoder.eval()
    obs = {"camera0_rgb": torch.zeros(1, 1, 3, 8, 8)}
    expected_value = torch.tensor(
        [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]
    ).mean()

    # When
    features = encoder(obs)

    # Then
    torch.testing.assert_close(
        features,
        torch.full_like(features, expected_value),
    )


def test_legacy_imagenet_norm_flag_preserves_existing_preprocessing() -> None:
    # Given
    encoder = _make_encoder(imagenet_norm=True)
    encoder.eval()
    obs = {"camera0_rgb": torch.zeros(1, 1, 3, 8, 8)}

    # When
    features = encoder(obs)

    # Then
    torch.testing.assert_close(features, torch.zeros_like(features))

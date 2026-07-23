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


def _make_encoder() -> TimmObsEncoder:
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

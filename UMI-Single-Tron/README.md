# UMI-Single-Tron

自包含的 UMI 单臂 DiT 扩散策略训练代码,从 `umi-diffusion-training` 迁移而来,
仅保留 `train_diffusion_dit_timm_umi_workspace` 训练配置所需的完整依赖闭包,
**不依赖原始仓库**。

## 安装

```bash
pip install -r requirements.txt
# torch/torchvision 请按你的 CUDA 版本安装(原环境为 pytorch 2.1.0 + cu121)
```

## 训练

```bash
python train.py --config-name=train_diffusion_dit_timm_umi_workspace
```

数据集路径在 `diffusion_policy/config/task/umi_single_arm_dit.yaml` 的
`dataset_path`(默认 `data/umi-pick-cube/dataset.npz`),按需修改。

## 结构

```
train.py                                    # Hydra 入口
diffusion_policy/
  config/
    train_diffusion_dit_timm_umi_workspace.yaml
    task/umi_single_arm_dit.yaml
  workspace/    # 训练循环 (accelerate + EMA)
  policy/       # DiffusionDiTTimmPolicy
  model/        # DiT backbone / TimmObsEncoder(DINOv2) / normalizer / ...
  dataset/      # UmiDataset (zarr replay buffer)
  common/ codecs/ env_runner/
umi/common/     # pose_util (位姿表示)
```

> `env_runner` 为训练时的空壳 rollout(返回空 dict),仅为满足 workspace 接口保留。

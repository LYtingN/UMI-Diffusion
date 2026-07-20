# Manip_Flow — 上层双臂操作策略(flow matching, DiT)

在 Prior_Recon 运动先验之上的上层操作策略。以 UMI 的 bimanual Diffusion
Policy 为参考设计,把 DDPM/DDIM 生成头替换为 **conditional flow matching
(rectified flow)**,backbone 默认 **DiT-1D**;其余组件(shape_meta、
TimmObsEncoder、UmiDataset、训练 workspace)全部复用 UMI 仓库,不复制代码。

## 为什么不用 UMI 的 diffusion 头

- UMI 的训练 workspace 对 policy 完全泛型(只调 `policy(batch)` /
  `policy.predict_action`),换头代价极小,不影响数据/编码器/训练设施。
- 下层 Prior_Recon 本身就是 flow matching,上下层统一生成框架。
- 推理:4–8 步 Euler 即可(UMI DDIM 16 步),对 online planner 每段一次的
  `dp_infer_fn` 调用更友好。

## 为什么 backbone 用 DiT 而不是 UNet

注意:**DiT 只是网络结构名**(*Diffusion Transformer* 论文的 adaLN
transformer 骨干),与生成范式无关——本包的训练目标是速度回归
`v = x₁-x₀`、采样是 Euler ODE 积分,是纯 flow matching,没有 noise
scheduler / ε 预测。选 DiT 的理由:

1. 下层 Masked_Flow 就是 transformer + flow,风格统一;现代 flow 策略栈
   (π0、RDT)也是 DiT 路线。
2. horizon 自由:ConditionalUnet1D 要求 horizon % 4 == 0(两级下采样的
   skip 拼接),DiT 无整除约束。默认 horizon 12 恰好两者都行,但 18/50
   等变体只有 DiT 能跑。
3. adaLN-zero 初始为恒等映射,对速度场回归是良性起点。
4. `backbone: unet` 保留(与 UMI 参考 A/B 对比;默认 horizon 12 可直接用)。

## 帧预算合同(与 delta87+lookahead 规划器对接的核心)

规划器(hist 2 + 4×8 = seg_len **34**,lookahead **16**,30 fps):

```
kp_window_len = 34 + 16 = 50 帧 @ 30fps ≈ 1.667 s
chunk 换算帧数 = floor((Ta-1)/action_fps × 30) + 1,
provider 窗口帧数 = hist + chunk 换算帧数 - 1,
action_fps = 采集fps / obs_down_sample_steps
```

**滚动闭环(本栈最初的设计)**:每循环生成一个 chunk → 先验消费为 kp
窗口 → 执行 `stride` 帧新轨迹 → replan。lookahead 预览只需对**被执行的
primitive** 完整;后面 primitive 的预览被 `look_valid` 截断——这正是先验
训练时随机截断预览所覆盖的分布。窗口需求公式:

```
窗口帧数 ≥ max(seg_len 34, stride + hist 2 + lookahead 16)
```

- stride=16(执行 prim 0-1,两者预览 16/16 全满):provider **34 帧即可**;
  默认 DP 重采样 34 帧,与 2 帧真实 history 去重合并后实际得到 35 帧。
  prim 2/3 预览截断且输出被丢弃;自回归是前向的,不污染已生成帧;
  下一段 history carry 取帧 [16,18](prim 1 干净输出)。
- stride=32(整段执行完才 replan):50 帧。
- chunk **< 34 帧永远是硬错误**:`_build_s_ee_from_segment` 对短窗口零
  填充,EE 条件被钉到全零垃圾姿态(delta87 EE 列硬钉放大)。UMI 默认
  配置(16 action @ 20Hz → 23 raw / 24 provider 帧)踩中。

**约束在时长,不在 token 数**:dp_adapter 会 slerp/lerp 重采样到 30fps,
低频少 token 一样能覆盖窗口——UMI 自己就是这么干的(downsample 3 →
~20Hz)。30fps 数据下的选择:

| down_sample | 动作频率 | horizon | 时长 | 30fps 帧数 | 说明 |
|---|---|---|---|---|---|
| 3 | 10 Hz | **12** | 1.10s | 34(+2 history 去重后 35) | **默认**:执行16、replan;被执行 prim 预览全满 |
| 3 | 10 Hz | 17 | 1.60s | 49(+history 后 50) | 4 个 prim 预览全满;stride 可到 32;17%4≠0 → 仅 DiT |

注:horizon 12 能被 4 整除,UNet backbone 在默认配置下也可用;DiT 仍为
默认(与下层 transformer+flow 统一、horizon 不受整除限制)。

配套的 bridge 侧加固(本次修改了 Prior_Recon 的 bridge):
- `dp_adapter.pad_keypoints_hold_last`:hold-last 补帧工具(手保持不动,
  物理上有意义,区别于规划器的零填充)。
- `DPKeypointProvider(min_window_len=planner.seg_len)`:chunk 不足 seg_len
  默认**报错**(`allow_short_window=True` 降级为 hold-pad + 警告);
  lookahead 区域缺帧不补(交给 look_valid);`last_n_valid` 上报真实帧数;
  `kp_window(start, window_len)` 参数名已澄清(收到的是 kp_window_len=50)。
- `FlowPolicyInference.assert_planner_budget(dp_fps, seg_len, kp_window_len,
  replan_stride=16, history_len=2)`:
  P5 集成时先自检。

## 与 Prior_Recon 的接口链(69 维公共接口不动)

```
FlowTimmPolicy.predict_action ──(12,20) RELATIVE action @10Hz──▶
  inference.FlowPolicyInference.make_dp_infer_fn ──dp_infer_fn(start,window_len)──▶
    bridge/dp_base_anchor.DPKeypointProvider (P2: base=执行态FK; 帧预算校验)──▶
      bridge/dp_adapter.dp_action_to_keypoints (P1: world=base@rel; 重采样30fps)──▶
        (35,2,7) 世界系 keypoints ──▶ plan_segment(kp_window=...) # 执行16帧后replan
        (35,2)   gripper 宽度    ──▶ 独立 gripper 控制器 (P7)
```

关键不变量(见 `inference.py` docstring):
- 本模块返回**裸的 relative action**,不做 world 变换(dp_adapter 负责,
  base 由 P2 从执行态 qpos FK)。不要过 UMI 的 `get_real_umi_action`。
- 喂给策略的 obs EE 位姿必须与 P2 同一 FK 链(HandPoseFK raw wrist 世界系),
  保证 "相对最后一帧 obs" ≡ "相对 P2 base"。
- provider 输入必须是以最后一帧 obs 时刻结束的连续 2 帧执行态 qpos;它把
  真实 history 放在窗口前缀,并丢弃 action[0] 中重复的当前帧。
- 双手同一世界系,`tx_robot1_robot0 = I`。

## 文件

- `policy/flow_timm_policy.py` — flow matching policy(API 镜像
  DiffusionUnetTimmPolicy;`backbone: dit|unet`;flow 版 prefix inpainting)
- `model/flow_dit_1d.py` — DiT-1D 速度场 backbone(adaLN-zero,与
  ConditionalUnet1D 同调用约定)
- `config/train_flow_unet_umi_bimanual_workspace.yaml` — 训练配置
  (task 复用 UMI `umi_bimanual`,覆盖 horizon 12 / down_sample 3)
- `scripts/train_flow_umi.py` — 训练入口(复用 UMI workspace,不改 UMI 代码)
- `inference.py` — checkpoint → `dp_infer_fn` 的 P5 胶水 + 帧预算自检
- `scripts/smoke_flow_policy.py` — 冒烟(训练机上跑;`--ckpt` 加真实规划器
  两段流式全链路)

## 用法(训练机)

```bash
# 冒烟:训练/推理/inpaint/帧预算/adapter/provider 校验
python pipeline/Manip_Flow/scripts/smoke_flow_policy.py
# 加真实 delta87+lookahead checkpoint 的端到端两段规划
python pipeline/Manip_Flow/scripts/smoke_flow_policy.py --ckpt /path/to/ckpt.pt

# 训练(UMI zarr 数据,先 debug 跑通)
python pipeline/Manip_Flow/scripts/train_flow_umi.py \
    task.dataset_path=/path/to/bimanual.zarr.zip training.debug=True
```

依赖:UMI 仓库位于 `<repo>/universal_manipulation_interface`(训练/推理侧
import;Prior_Recon 的 bridge 不 import,合同边界不变)。

## 调参起点

- `num_inference_steps: 8`(可降到 4 试实时性)
- `time_sample: uniform`;中段监督不足可切 `logit_normal`
- `dit_d_model 512 / depth 6 / heads 8`(≈28M,与 UMI UNet 同量级)
- 数据采集 fps 若非 30 或 stride 非 16:保持 provider 窗口帧数 ≥
  max(34, stride+18),调 horizon 与 down_sample;
  DPKeypointProvider 的 `dp_fps` = 采集fps/down_sample

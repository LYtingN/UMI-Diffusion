"""LeRobot RTC soft-prefix guidance adapted to forward-time rectified flow.

The guidance and EXP/LINEAR schedules follow Hugging Face LeRobot's
``RTCProcessor`` at commit 64b23178d5348609c266250d3e1f511eba4c33ff. This
policy integrates noise-to-data from t=0 to t=1, so its equivalent clean-action
estimate is ``x_t + (1 - t) v_t``. No action channel is hard-latched by RTC.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import torch


class RTCConfigError(ValueError):
    pass


class FlowPolicyConfigError(ValueError):
    pass


class FlowSamplingConfig(NamedTuple):
    inference_steps: int
    time_embed_scale: float
    action_horizon: int
    execution_horizon: int
    max_guidance_weight: float
    prefix_schedule: str


def rtc_prefix_weights(
    *,
    inference_delay: int,
    execution_horizon: int,
    total_horizon: int,
    schedule: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if total_horizon < 1 or execution_horizon < 1 or inference_delay < 0:
        raise RTCConfigError(
            "RTC horizons must be positive and inference_delay non-negative"
        )
    end = min(execution_horizon, total_horizon)
    start = min(inference_delay, end)
    transition_size = end - start
    if transition_size > 0:
        transition = torch.linspace(
            1.0,
            0.0,
            transition_size + 2,
            device=device,
            dtype=dtype,
        )[1:-1]
        if schedule == "exp":
            transition = transition * torch.expm1(transition) / (torch.e - 1.0)
        elif schedule != "linear":
            raise RTCConfigError(
                f"RTC prefix schedule must be 'linear' or 'exp', got {schedule!r}"
            )
    else:
        transition = torch.empty(0, device=device, dtype=dtype)
    return torch.cat(
        (
            torch.ones(start, device=device, dtype=dtype),
            transition,
            torch.zeros(total_horizon - end, device=device, dtype=dtype),
        )
    )


def sample_flow_time(
    count: int,
    device: torch.device,
    dtype: torch.dtype,
    schedule: str,
    logit_normal_mean: float,
    logit_normal_std: float,
) -> torch.Tensor:
    if schedule == "logit_normal":
        normal = torch.randn(count, device=device, dtype=dtype)
        return torch.sigmoid(
            normal * logit_normal_std + logit_normal_mean
        )
    return torch.rand(count, device=device, dtype=dtype)


def rtc_guided_velocity(
    *,
    state: torch.Tensor,
    time: torch.Tensor,
    velocity_fn: Callable[[torch.Tensor], torch.Tensor],
    prefix: torch.Tensor,
    weights: torch.Tensor,
    max_guidance_weight: float,
) -> torch.Tensor:
    with torch.enable_grad():
        differentiable_state = state.detach().requires_grad_(True)
        velocity = velocity_fn(differentiable_state)
        estimate = differentiable_state + (1.0 - time) * velocity
        error = (prefix - estimate) * weights
        correction = torch.autograd.grad(
            estimate,
            differentiable_state,
            error.detach(),
        )[0]
    numerator = (1.0 - time).square() + time.square()
    denominator = time * (1.0 - time)
    raw_weight = numerator / denominator
    guidance_weight = torch.nan_to_num(
        raw_weight,
        posinf=max_guidance_weight,
    ).clamp(max=max_guidance_weight)
    return (velocity + guidance_weight * correction).detach()


def flow_euler_sample(
    *,
    model: torch.nn.Module,
    condition_data: torch.Tensor,
    condition_mask: torch.Tensor,
    global_cond: torch.Tensor | None,
    generator: torch.Generator | None,
    rtc_action_prefix: torch.Tensor | None,
    rtc_inference_delay: int,
    config: FlowSamplingConfig,
) -> torch.Tensor:
    initial = torch.randn(
        size=condition_data.shape,
        dtype=condition_data.dtype,
        device=condition_data.device,
        generator=generator,
    )
    state = initial
    times = torch.linspace(
        0.0,
        1.0,
        config.inference_steps + 1,
        dtype=condition_data.dtype,
        device=condition_data.device,
    )
    weights = None
    guided_prefix = rtc_action_prefix
    if rtc_action_prefix is not None:
        prefix_horizon = min(
            rtc_action_prefix.shape[1],
            config.action_horizon,
        )
        guided_prefix = torch.zeros_like(condition_data)
        guided_prefix[:, :prefix_horizon] = rtc_action_prefix[:, :prefix_horizon]
        weights = rtc_prefix_weights(
            inference_delay=rtc_inference_delay,
            execution_horizon=min(
                config.execution_horizon,
                prefix_horizon,
            ),
            total_horizon=config.action_horizon,
            schedule=config.prefix_schedule,
            device=condition_data.device,
            dtype=condition_data.dtype,
        ).view(1, config.action_horizon, 1)
    for index in range(config.inference_steps):
        time = times[index]
        pinned = (1.0 - time) * initial + time * condition_data
        state = torch.where(condition_mask, pinned, state)
        time_batch = (
            time.expand(state.shape[0]) * config.time_embed_scale
        )

        def velocity_fn(value: torch.Tensor) -> torch.Tensor:
            return model(
                value,
                time_batch,
                local_cond=None,
                global_cond=global_cond,
            )

        velocity = (
            velocity_fn(state)
            if guided_prefix is None
            else rtc_guided_velocity(
                state=state,
                time=time,
                velocity_fn=velocity_fn,
                prefix=guided_prefix,
                weights=weights,
                max_guidance_weight=config.max_guidance_weight,
            )
        )
        state = state + velocity * (times[index + 1] - time)
    return torch.where(condition_mask, condition_data, state)

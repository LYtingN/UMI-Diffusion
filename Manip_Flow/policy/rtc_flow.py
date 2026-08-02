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
    latched_channels: tuple[int, ...]


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


def rtc_action_prefix_weights(
    *,
    inference_delay: int,
    execution_horizon: int,
    prefix_horizon: int,
    total_horizon: int,
    action_dim: int,
    latched_channels: tuple[int, ...],
    schedule: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not 0 <= prefix_horizon <= total_horizon or action_dim < 1:
        raise RTCConfigError("RTC prefix horizon or action dimension is invalid")
    if any(channel < 0 or channel >= action_dim for channel in latched_channels):
        raise RTCConfigError("RTC latched channel is outside the action dimension")
    pose_weights = rtc_prefix_weights(
        inference_delay=inference_delay,
        execution_horizon=execution_horizon,
        total_horizon=total_horizon,
        schedule=schedule,
        device=device,
        dtype=dtype,
    )
    weights = pose_weights[:, None].repeat(1, action_dim)
    weights[:prefix_horizon, list(latched_channels)] = 1.0
    return weights


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
        weights = rtc_action_prefix_weights(
            inference_delay=rtc_inference_delay,
            execution_horizon=min(
                config.execution_horizon,
                prefix_horizon,
            ),
            prefix_horizon=prefix_horizon,
            total_horizon=config.action_horizon,
            action_dim=condition_data.shape[-1],
            latched_channels=config.latched_channels,
            schedule=config.prefix_schedule,
            device=condition_data.device,
            dtype=condition_data.dtype,
        ).unsqueeze(0)
        latch_mask = torch.zeros_like(condition_mask)
        latch_mask[:, :prefix_horizon, list(config.latched_channels)] = True
        condition_data = torch.where(latch_mask, guided_prefix, condition_data)
        condition_mask = condition_mask | latch_mask
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

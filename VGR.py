# VGR.py
import torch

def update_vgr(curvature, state, beta, eps, step):
    """
    Variance-Gated Reliability (VGR) module with bias correction.

    Maintains exponential moving averages of curvature statistics and computes
    a reliability gate based on curvature variance. The gate value indicates
    the trustworthiness of the curvature estimate, with higher variance
    leading to lower reliability.

    `state` holds the VGR sub-state for exactly one parameter (the caller's
    per-parameter optimizer state), keyed by field name only - not by the
    parameter object. Keying by the parameter tensor itself breaks
    optimizer state_dict/load_state_dict round-trips, since a resumed run
    constructs fresh Parameter objects with different identity.

    Args:
        curvature: curvature tensor for this parameter
        state: dict for storing this parameter's VGR state (modified in-place)
        beta: EMA decay coefficient
        eps: epsilon for numerical stability
        step: current optimization step (for bias correction)

    Returns:
        updated state dict with 'curv_ema', 'curv_sq_ema', 'gate'
    """
    curv = curvature

    # Compute bias correction factor for EMA
    bias_correction = 1.0 - beta ** step

    # Initialize state for this parameter if not exists
    if 'curv_ema' not in state:
        state['curv_ema'] = torch.zeros_like(curv)
        state['curv_sq_ema'] = torch.zeros_like(curv)
        state['gate'] = torch.ones_like(curv)

    # Update EMA of curvature mean
    state['curv_ema'].mul_(beta).add_(curv, alpha=1 - beta)

    # Update EMA of curvature squared
    state['curv_sq_ema'].mul_(beta).add_(curv * curv, alpha=1 - beta)

    # Apply bias correction
    curv_ema_corrected = state['curv_ema'] / bias_correction
    curv_sq_ema_corrected = state['curv_sq_ema'] / bias_correction

    # Compute variance using bias-corrected values
    variance = curv_sq_ema_corrected - curv_ema_corrected ** 2

    # Clamp variance to be non-negative for numerical stability
    variance = torch.clamp(variance, min=0.0)

    # Compute reliability gate
    # Higher variance -> lower gate -> less reliable
    state['gate'] = 1.0 / (1.0 + variance + eps)

    return state

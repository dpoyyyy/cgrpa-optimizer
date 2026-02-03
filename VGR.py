# VGR.py
import torch

def update_vgr(curvature, state, beta, eps, step):
    """
    Variance-Gated Reliability (VGR) module with bias correction.
    
    Maintains exponential moving averages of curvature statistics and computes
    a reliability gate based on curvature variance. The gate value indicates
    the trustworthiness of the curvature estimate, with higher variance
    leading to lower reliability.
    
    Args:
        curvature: dict mapping parameters to curvature tensors
        state: dict for storing VGR state (modified in-place)
        beta: EMA decay coefficient
        eps: epsilon for numerical stability
        step: current optimization step (for bias correction)
        
    Returns:
        updated state dict containing gates for each parameter
    """
    # Compute bias correction factor for EMA
    bias_correction = 1.0 - beta ** step
    
    # Process each parameter's curvature
    for param, curv in curvature.items():
        # Initialize state for this parameter if not exists
        if param not in state:
            state[param] = {}
            state[param]['curv_ema'] = torch.zeros_like(curv)
            state[param]['curv_sq_ema'] = torch.zeros_like(curv)
            state[param]['gate'] = torch.ones_like(curv)
        
        # Get state for this parameter
        param_state = state[param]
        
        # Update EMA of curvature mean
        param_state['curv_ema'].mul_(beta).add_(curv, alpha=1 - beta)
        
        # Update EMA of curvature squared
        param_state['curv_sq_ema'].mul_(beta).add_(curv * curv, alpha=1 - beta)
        
        # Apply bias correction
        curv_ema_corrected = param_state['curv_ema'] / bias_correction
        curv_sq_ema_corrected = param_state['curv_sq_ema'] / bias_correction
        
        # Compute variance using bias-corrected values
        variance = curv_sq_ema_corrected - curv_ema_corrected ** 2
        
        # Clamp variance to be non-negative for numerical stability
        variance = torch.clamp(variance, min=0.0)
        
        # Compute reliability gate
        # Higher variance -> lower gate -> less reliable
        param_state['gate'] = 1.0 / (1.0 + variance + eps)
    
    return state

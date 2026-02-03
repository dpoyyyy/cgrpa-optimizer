# RPA_Core.py
import torch

def rpa_update(param, direction, curv_mean, state, alpha, eps):
    """
    Regularized Projected Anchor (RPA) update with coupled anchor dynamics.
    
    Maintains an anchor point that moves in a curvature-coupled manner,
    then geometrically interpolates the parameter toward the anchor.
    The coupling coefficient ensures the anchor and parameter remain
    bounded and prevents divergence in high-curvature regions.
    
    Args:
        param: parameter tensor to update
        direction: modified gradient direction tensor
        curv_mean: mean curvature tensor
        state: dict for storing anchor (modified in-place)
        alpha: anchor step size (base learning rate)
        eps: epsilon for numerical stability
        
    Returns:
        tuple of (updated param tensor, updated state dict)
    """
    # Lazily initialize anchor as a clone of current parameter
    if 'anchor' not in state:
        state['anchor'] = param.clone()
    
    # Update anchor by taking a step in the negative direction
    anchor_new = state['anchor'] - alpha * direction
    
    # Store updated anchor
    state['anchor'] = anchor_new
    
    # Compute interpolation coefficient beta based on curvature
    # Higher curvature -> smaller beta -> stay closer to current param
    # Lower curvature -> larger beta -> move more toward anchor
    
    # Clamp curv_mean to prevent division issues
    curv_mean_safe = torch.clamp(curv_mean, min=-100.0, max=100.0)
    
    # Ensure denominator is always positive and bounded
    denominator = torch.clamp(1.0 + curv_mean_safe, min=eps, max=1e6)
    beta = 1.0 / denominator
    
    # Clamp beta to valid interpolation range [0, 1]
    beta = torch.clamp(beta, min=0.0, max=1.0)
    
    # Geometric interpolation between current param and anchor
    param_new = (1.0 - beta) * param + beta * anchor_new
    
    # Sanitize output to handle NaN and Inf values
    param_new = torch.nan_to_num(param_new, nan=param.mean().item(), posinf=1e6, neginf=-1e6)
    
    return param_new, state

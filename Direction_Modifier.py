# Direction_Modifier.py
import torch

def modify_direction(grad, curv_mean, gate, eps, tau_max=10.0):
    """
    Modify gradient direction using variance-gated interpolation with
    trust region bounded Newton-like scaling.
    
    Args:
        grad: gradient tensor
        curv_mean: mean curvature tensor (same shape as grad)
        gate: reliability gate tensor (same shape as grad)
        eps: epsilon for numerical stability
        tau_max: maximum trust region scaling factor (default: 10.0)
        
    Returns:
        modified direction tensor (same shape as inputs)
    """
    # Safe curvature magnitude (handles negative curvature)
    kappa_t = torch.abs(curv_mean) + eps
    
    # Clamp kappa to prevent division by zero or overflow
    kappa_t = torch.clamp(kappa_t, min=eps, max=1e6)
    
    # Trust region bounded inverse curvature scaling
    tau_t = torch.minimum(1.0 / kappa_t, torch.tensor(tau_max, dtype=kappa_t.dtype, device=kappa_t.device))
    
    # Clamp tau to valid range
    tau_t = torch.clamp(tau_t, min=0.0, max=tau_max)
    
    # Adaptive direction scaling factor
    # When gate = 0 (unreliable): lambda_t = 1 (pure SGD)
    # When gate = 1 (reliable): lambda_t = tau_t (Newton-scaled)
    lambda_t = (1.0 - gate) + gate * tau_t
    
    # Clamp lambda to prevent explosive updates
    lambda_t = torch.clamp(lambda_t, min=0.0, max=tau_max)
    
    # Final direction: d_t = lambda_t * grad
    modified_dir = lambda_t * grad
    
    # Sanitize output to handle NaN and Inf values
    modified_dir = torch.nan_to_num(modified_dir, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return modified_dir

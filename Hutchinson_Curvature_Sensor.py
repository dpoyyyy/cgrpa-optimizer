# Hutchinson_Curvature_Sensor.py
import torch

def hutchinson_diagonal_estimator(loss, params):
    """
    Hutchinson diagonal Hessian estimator using Rademacher random vectors.
    
    This function estimates the diagonal of the Hessian matrix using a
    single-sample Monte Carlo approach with Rademacher (+1/-1) vectors.
    
    Args:
        loss: scalar tensor (loss value)
        params: iterable of torch.nn.Parameter
        
    Returns:
        dict mapping each parameter to its estimated diagonal curvature tensor
    """
    # Filter params to only those that require grad
    params = [p for p in params if p.requires_grad]
    
    if len(params) == 0:
        return {}
    
    # Compute first-order gradients with graph retention for second derivatives
    grads = torch.autograd.grad(
        loss, 
        params, 
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )
    
    # Filter out None gradients
    valid_params = []
    valid_grads = []
    for p, g in zip(params, grads):
        if g is not None:
            valid_params.append(p)
            valid_grads.append(g)
    
    if len(valid_params) == 0:
        return {}
    
    # Generate Rademacher random vectors (+1 or -1) for each parameter
    rademacher_vectors = []
    for param in valid_params:
        v = torch.randint_like(param, high=2, dtype=param.dtype, device=param.device) * 2 - 1
        rademacher_vectors.append(v)
    
    # Compute dot product between gradients and Rademacher vectors
    grad_v_product = sum(
        (g * v).sum() 
        for g, v in zip(valid_grads, rademacher_vectors)
    )
    
    # Compute Hessian-vector product via automatic differentiation
    hvp = torch.autograd.grad(
        grad_v_product,
        valid_params,
        retain_graph=False,
        allow_unused=True
    )
    
    # Estimate diagonal as elementwise product v ⊙ Hv
    diagonal_estimates = {}
    for param, v, hv in zip(valid_params, rademacher_vectors, hvp):
        if hv is not None:
            diagonal_estimates[param] = (v * hv).detach()
        else:
            diagonal_estimates[param] = torch.zeros_like(param)
    
    return diagonal_estimates

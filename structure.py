# structure.py
import logging
import torch
from torch.optim.optimizer import Optimizer
import VGR, RPA_Core, Direction_Modifier, Hutchinson_Curvature_Sensor

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.WARNING)


class CGRPAOptimizer(Optimizer):
    """
    CGRPA Optimizer - Curvature-Gated Recursive Parameter Anchoring.

    A second-order optimizer that combines Hutchinson diagonal curvature estimation,
    Variance-Gated Reliability (VGR), adaptive direction modification, and
    Regularized Projected Anchor (RPA) updates for stable optimization.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients (beta_vgr, beta_unused) for VGR EMA (default: (0.9, 0.999))
        eps: term added to denominator for numerical stability (default: 1e-8)
        tau_max: maximum trust region scaling factor (default: 5.0)
        weight_decay: weight decay coefficient (default: 0.0)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 tau_max=5.0, weight_decay=0.0):
        """Initialize optimizer with hyperparameters."""
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(lr=lr, betas=betas, eps=eps, tau_max=tau_max,
                        weight_decay=weight_decay)
        super(CGRPAOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
                    Required for computing second-order curvature information.

        Returns:
            loss value (if closure is provided)
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Collect all parameters with gradients
        all_params = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    all_params.append(p)

        # Estimate curvature using Hutchinson diagonal estimator.
        # HVP failure must never silently degrade into zero curvature -
        # that lets tau_max saturate and the update explode. Track the
        # failure explicitly so the update loop can force a safe SGD path.
        hvp_failed = False
        curv_estimates = {}
        with torch.enable_grad():
            if len(all_params) > 0 and loss is not None:
                try:
                    curv_estimates = Hutchinson_Curvature_Sensor.hutchinson_diagonal_estimator(
                        loss, all_params
                    )
                except Exception as e:
                    hvp_failed = True
                    curv_estimates = {}
                    logger.warning(
                        "HVP Computation failed: %s. Falling back to First-Order SGD.", e
                    )

        # Update parameters
        with torch.no_grad():
            for group in self.param_groups:
                beta_vgr = group['betas'][0]
                lr = group['lr']
                eps = group['eps']
                tau_max = group['tau_max']
                weight_decay = group['weight_decay']

                for p in group['params']:
                    if p.grad is None:
                        continue

                    grad = p.grad.clone()

                    # Skip update if gradient contains NaN or Inf
                    if torch.isnan(grad).any() or torch.isinf(grad).any():
                        continue

                    # Normalize gradient if norm exceeds 1.0
                    grad_norm = torch.norm(grad)
                    if grad_norm > 1.0:
                        grad = grad / (grad_norm + eps)

                    # Initialize or retrieve parameter state
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['vgr_state'] = {}
                        state['rpa_state'] = {}
                    state['step'] += 1

                    # Apply weight decay
                    if weight_decay > 0.0:
                        grad = grad.add(p, alpha=weight_decay)

                    # Safe fallback path: HVP failed for this step, or this
                    # parameter has no curvature estimate. Do NOT route
                    # through VGR/Direction Modifier/RPA - go straight to
                    # first-order SGD and move to the next parameter.
                    if hvp_failed or p not in curv_estimates:
                        p.data.sub_(grad, alpha=lr)
                        continue

                    curvature = curv_estimates[p].clone()

                    # Sanitize curvature values
                    curvature = torch.nan_to_num(curvature, nan=0.0, posinf=50.0, neginf=-50.0)
                    curvature = torch.clamp(curvature, min=-50.0, max=50.0)

                    # Update VGR state
                    state['vgr_state'] = VGR.update_vgr(
                        curvature=curvature,
                        state=state['vgr_state'],
                        beta=beta_vgr,
                        eps=eps,
                        step=state['step']
                    )

                    curv_mean = state['vgr_state']['curv_ema']
                    gate = state['vgr_state']['gate']

                    # Clamp curvature mean and gate to safe ranges
                    curv_mean = torch.clamp(curv_mean, min=-50.0, max=50.0)
                    gate = torch.clamp(gate, min=0.0, max=1.0)

                    # Compute modified direction
                    direction = Direction_Modifier.modify_direction(
                        grad=grad,
                        curv_mean=curv_mean,
                        gate=gate,
                        eps=eps,
                        tau_max=tau_max
                    )

                    # Fallback to gradient descent if direction is invalid
                    if torch.isnan(direction).any() or torch.isinf(direction).any():
                        direction = grad

                    # Limit direction magnitude
                    direction = torch.clamp(direction, min=-1.0, max=1.0)

                    # Apply RPA update
                    new_param, state['rpa_state'] = RPA_Core.rpa_update(
                        param=p,
                        direction=direction,
                        curv_mean=curv_mean,
                        state=state['rpa_state'],
                        alpha=lr,
                        eps=eps
                    )

                    # Verify output and apply fallback if necessary
                    if torch.isnan(new_param).any() or torch.isinf(new_param).any():
                        # Fallback to SGD step
                        new_param = p - lr * grad
                        state['rpa_state']['anchor'] = p.clone()
                    else:
                        # Check for parameter explosion
                        param_magnitude = torch.norm(new_param)
                        if param_magnitude > 1e6:
                            new_param = p - lr * grad
                            state['rpa_state']['anchor'] = p.clone()

                    # Apply update if valid
                    if not (torch.isnan(new_param).any() or torch.isinf(new_param).any()):
                        p.data.copy_(new_param)

        return loss

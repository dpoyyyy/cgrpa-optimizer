# __init__.py
"""
CGRPA Optimizer - Curvature-Gated Recursive Parameter Anchoring

A second-order PyTorch optimizer combining Hutchinson curvature estimation,
variance-based reliability gating, and recursive anchor dynamics.
"""

from .structure import CGRPAOptimizer

__version__ = "0.1.0"
__all__ = ["CGRPAOptimizer"]

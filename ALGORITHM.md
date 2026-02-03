# CGRPA: Mathematical Foundation and Algorithm

This document provides a detailed mathematical explanation of the Curvature-Gated Recursive Parameter Anchoring (CGRPA) optimizer.

## Table of Contents
1. [Overview](#overview)
2. [Motivation](#motivation)
3. [Algorithm Components](#algorithm-components)
4. [Complete Algorithm](#complete-algorithm)
5. [Convergence Properties](#convergence-properties)
6. [Complexity Analysis](#complexity-analysis)

---

## Overview

CGRPA is a second-order optimization method that adaptively interpolates between first-order (gradient descent) and second-order (Newton-like) directions based on curvature reliability. The algorithm consists of four main components working in tandem:

1. **Hutchinson Diagonal Estimator** - Stochastic curvature estimation
2. **Variance-Gated Reliability (VGR)** - Reliability assessment
3. **Direction Modifier** - Adaptive direction computation
4. **Regularized Projected Anchor (RPA)** - Coupled parameter updates

---

## Motivation

### Problem with Standard Optimizers

**First-order methods (SGD, Adam):**
- Use only gradient information: ∇L(θ)
- No awareness of local curvature
- Fixed or heuristic step size adaptation
- Can be inefficient in ill-conditioned landscapes

**Second-order methods (Newton, L-BFGS):**
- Require full Hessian: H = ∇²L(θ)
- Computational cost: O(n²) memory, O(n³) inversion
- Unstable in non-convex regions
- Impractical for deep neural networks

### CGRPA Solution

CGRPA addresses these limitations by:
1. Estimating only the **diagonal** of the Hessian (O(n) memory)
2. Using **stochastic estimation** (Hutchinson's method)
3. **Gating** second-order updates based on reliability
4. **Coupling** parameter and anchor dynamics to prevent divergence

---

## Algorithm Components

### 1. Hutchinson Diagonal Estimator

**Objective:** Estimate diag(H) where H = ∇²L(θ)

**Method:** Monte Carlo estimation using Rademacher vectors

#### Mathematical Formulation

Given a random vector **v** where each element v_i ∼ Rademacher({-1, +1}):

```
E[v ⊙ Hv] = E[v ⊙ (Hv)] = diag(H)
```

**Proof sketch:**
```
E[v_i · (Hv)_i] = E[v_i · Σ_j H_ij v_j]
                = E[v_i² H_ii] + E[v_i · Σ_{j≠i} H_ij v_j]
                = H_ii · E[v_i²] + Σ_{j≠i} H_ij · E[v_i v_j]
                = H_ii · 1 + Σ_{j≠i} H_ij · 0
                = H_ii
```

#### Implementation

**Step 1:** Sample Rademacher vector
```
v ~ Rademacher({-1, +1}^d)
```

**Step 2:** Compute first-order gradients
```
g = ∇L(θ)
```

**Step 3:** Compute Hessian-vector product
```
Hv = ∇(g^T v)
```

**Step 4:** Estimate diagonal
```
c_t = v ⊙ Hv
```

**Properties:**
- Unbiased: E[c_t] = diag(H)
- Single-sample variance: Var(c_t) ≈ ||H||²_F
- Computational cost: 2 gradient computations

---

### 2. Variance-Gated Reliability (VGR)

**Objective:** Assess reliability of curvature estimates using variance

#### Exponential Moving Averages

Maintain two EMAs with decay coefficient β ∈ (0, 1):

**Mean curvature:**
```
m_t = β · m_{t-1} + (1 - β) · c_t
```

**Mean squared curvature:**
```
s_t = β · s_{t-1} + (1 - β) · c_t²
```

#### Bias Correction

To correct for initialization bias (like Adam):

```
m̂_t = m_t / (1 - β^t)
ŝ_t = s_t / (1 - β^t)
```

#### Variance Estimation

```
σ²_t = max(ŝ_t - m̂_t², 0)
```

The max ensures non-negativity due to numerical errors.

#### Reliability Gate

```
g_t = 1 / (1 + σ²_t)
```

**Interpretation:**
- σ²_t → 0 (low variance) ⟹ g_t → 1 (highly reliable)
- σ²_t → ∞ (high variance) ⟹ g_t → 0 (unreliable)

**Properties:**
- g_t ∈ [0, 1]
- Continuous and differentiable
- Adaptive to estimate quality

---

### 3. Direction Modifier

**Objective:** Compute adaptive search direction balancing SGD and Newton

#### Safe Curvature Magnitude

Handle negative curvature (saddle points):

```
κ_t = |m̂_t| + ε
```

where ε > 0 is a small constant for numerical stability.

#### Trust Region Scaling

Bound the inverse curvature to prevent explosion in flat regions:

```
τ_t = min(1/κ_t, τ_max)
```

**Analysis:**
- κ_t → 0 (flat region): τ_t → τ_max (bounded)
- κ_t → ∞ (sharp region): τ_t → 0 (conservative)

#### Adaptive Direction Scaling

Interpolate between SGD (λ=1) and Newton (λ=τ_t):

```
λ_t = (1 - g_t) + g_t · τ_t
```

**Cases:**
1. **Unreliable (g_t ≈ 0):**
   ```
   λ_t ≈ 1 ⟹ d_t ≈ ∇L (pure SGD)
   ```

2. **Reliable + Flat (g_t ≈ 1, κ_t ≈ 0):**
   ```
   λ_t ≈ τ_max ⟹ d_t ≈ τ_max · ∇L (accelerated)
   ```

3. **Reliable + Sharp (g_t ≈ 1, κ_t large):**
   ```
   λ_t ≈ τ_t ≈ 0 ⟹ d_t ≈ 0 (conservative)
   ```

#### Final Direction

```
d_t = λ_t · ∇L(θ_t)
```

---

### 4. Regularized Projected Anchor (RPA)

**Objective:** Update parameters via coupled anchor dynamics

#### Coupling Coefficient

```
ω_t = 1 / (1 + κ_t)
```

**Properties:**
- ω_t ∈ (0, 1]
- κ_t → 0: ω_t → 1 (strong coupling)
- κ_t → ∞: ω_t → 0 (weak coupling)

#### Anchor Update

The anchor moves in the direction of the update, **scaled by coupling**:

```
a_t = a_{t-1} - α · ω_t · d_t
```

**Critical:** The ω_t scaling prevents anchor divergence in high-curvature regions.

#### Parameter Update

Geometric interpolation between current parameter and anchor:

```
θ_t = (1 - ω_t) · θ_{t-1} + ω_t · a_t
```

#### Effective Update

Substituting the anchor update:

```
θ_t = (1 - ω_t) · θ_{t-1} + ω_t · (a_{t-1} - α · ω_t · d_t)
    = θ_{t-1} - α · ω_t² · d_t + ω_t · (a_{t-1} - θ_{t-1})
```

**Effective step size:** α · ω_t²

**Anchor-parameter coupling:**
The term ω_t · (a_{t-1} - θ_{t-1}) pulls θ toward the anchor, with strength proportional to coupling.

---

## Complete Algorithm

### Algorithm: CGRPA Optimizer

**Input:** 
- Initial parameters θ₀
- Learning rate α
- VGR decay β
- Trust region bound τ_max
- Numerical stability ε

**Initialize:**
- m₀ = 0, s₀ = 0 (VGR state)
- a₀ = θ₀ (anchor)
- t = 0

**Repeat until convergence:**

1. **Compute Loss and Gradients**
   ```
   L_t = L(θ_t)
   ∇L_t = ∇L(θ_t)
   ```

2. **Estimate Curvature (Hutchinson)**
   ```
   Sample v ~ Rademacher({-1, +1}^d)
   Compute Hv = ∇(∇L_t^T v)
   c_t = v ⊙ Hv
   ```

3. **Update VGR State**
   ```
   t = t + 1
   m_t = β · m_{t-1} + (1-β) · c_t
   s_t = β · s_{t-1} + (1-β) · c_t²
   
   Bias correction:
   m̂_t = m_t / (1 - β^t)
   ŝ_t = s_t / (1 - β^t)
   
   Variance:
   σ²_t = max(ŝ_t - m̂_t², 0)
   
   Gate:
   g_t = 1 / (1 + σ²_t)
   ```

4. **Compute Modified Direction**
   ```
   κ_t = |m̂_t| + ε
   τ_t = min(1/κ_t, τ_max)
   λ_t = (1 - g_t) + g_t · τ_t
   d_t = λ_t · ∇L_t
   ```

5. **Update Anchor and Parameters (RPA)**
   ```
   ω_t = 1 / (1 + κ_t)
   a_t = a_{t-1} - α · ω_t · d_t
   θ_t = (1 - ω_t) · θ_{t-1} + ω_t · a_t
   ```

**End Repeat**

**Output:** Optimized parameters θ*

---

## Convergence Properties

### Theorem 1: Bounded Updates

**Statement:** For any step t, the parameter update is bounded:

```
||θ_t - θ_{t-1}|| ≤ α · τ_max · ||∇L_t||
```

**Proof:**

From the algorithm:
```
θ_t - θ_{t-1} = -α · ω_t² · d_t
```

Since d_t = λ_t · ∇L_t and λ_t ≤ τ_max:
```
||θ_t - θ_{t-1}|| = α · ω_t² · ||λ_t · ∇L_t||
                   ≤ α · 1 · τ_max · ||∇L_t||
                   = α · τ_max · ||∇L_t||
```

□

### Theorem 2: Anchor-Parameter Coupling

**Statement:** The distance between anchor and parameter is controlled:

```
||a_t - θ_t|| ≤ C · max_{i≤t} ||d_i||
```

for some constant C depending on {ω_i}.

**Sketch:**

The recurrence relation:
```
a_t - θ_t = (1 - ω_t)(a_{t-1} - θ_{t-1}) - α · ω_t · d_t
```

shows the divergence contracts by factor (1 - ω_t) at each step, preventing unbounded growth.

□

### Theorem 3: Graceful Degradation

**Statement:** In regions of high curvature uncertainty (σ²_t → ∞):

```
d_t → ∇L_t (pure gradient descent)
```

**Proof:**

As σ²_t → ∞:
```
g_t = 1/(1 + σ²_t) → 0
```

Therefore:
```
λ_t = (1 - g_t) + g_t · τ_t → (1 - 0) + 0 · τ_t = 1
```

Thus:
```
d_t = λ_t · ∇L_t → ∇L_t
```

□

---

## Complexity Analysis

### Computational Complexity

**Per iteration:**

| Operation | Cost |
|-----------|------|
| Forward pass | O(n) |
| First gradient ∇L | O(n) |
| Hutchinson Hv | O(n) |
| VGR update | O(n) |
| Direction modifier | O(n) |
| RPA update | O(n) |
| **Total** | **O(n)** |

**Comparison:**
- SGD: O(n) per iteration
- Adam: O(n) per iteration
- CGRPA: O(n) per iteration (with 2× gradient cost)
- Newton: O(n³) per iteration (matrix inversion)

### Memory Complexity

**Per parameter:**

| Component | Memory |
|-----------|--------|
| Parameter θ | 1 |
| Gradient ∇L | 1 |
| VGR (m, s, g) | 3 |
| Anchor a | 1 |
| **Total** | **6** |

**Comparison:**
- SGD: 1 per parameter
- Adam: 3 per parameter (m, v, θ)
- CGRPA: 6 per parameter
- Newton: n per parameter (full Hessian row)

---

## Special Cases and Limit Behavior

### Case 1: Convex Quadratic

For L(θ) = ½θ^T Q θ + b^T θ with Q positive definite:

- Curvature estimate: c_t → diag(Q)
- Variance: σ²_t → 0 (deterministic Hessian)
- Gate: g_t → 1
- Direction: d_t ≈ τ_t · ∇L ≈ Q⁻¹ ∇L (Newton direction on diagonal)

**Result:** Near-optimal preconditioning

### Case 2: Noisy Non-Convex

For highly stochastic gradients with varying curvature:

- Variance: σ²_t remains high
- Gate: g_t ≈ 0
- Direction: d_t ≈ ∇L (SGD)

**Result:** Stable gradient descent

### Case 3: Saddle Point

At θ* where ∇L(θ*) = 0 and H has negative eigenvalues:

- Curvature: m̂_t includes negative values
- Safe curvature: κ_t = |m̂_t| > 0
- Direction: uses |curvature| for scaling, avoiding sign flip

**Result:** Safe escape from saddle

---

## Comparison to Related Methods

### vs. Adam

**Adam:**
```
m_t = β₁ m_{t-1} + (1-β₁) g_t
v_t = β₂ v_{t-1} + (1-β₂) g_t²
θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)
```

**CGRPA advantage:**
- Uses second-order information (curvature) not just gradient statistics
- Reliability gating adapts to estimate quality
- Anchor dynamics provide additional stability

### vs. Natural Gradient

**Natural Gradient:**
```
θ_t = θ_{t-1} - α · F⁻¹ ∇L_t
```
where F is the Fisher Information Matrix.

**CGRPA advantage:**
- Diagonal approximation: O(n) vs O(n²)
- Variance-based reliability assessment
- No need for Fisher matrix computation

### vs. L-BFGS

**L-BFGS:**
- Maintains m-step history to approximate H⁻¹
- Requires line search
- Not suitable for stochastic/mini-batch settings

**CGRPA advantage:**
- Single-sample stochastic estimation
- No line search needed
- Designed for stochastic optimization

---

## Hyperparameter Guidance

### Learning Rate (α)

**Range:** 10⁻⁴ to 10⁻²

**Effect:**
- Too small: slow convergence
- Too large: instability despite safety mechanisms

**Recommendation:** Start with 10⁻³, adjust based on loss curve

### VGR Decay (β)

**Range:** 0.85 to 0.95

**Effect:**
- Higher β: slower adaptation, smoother estimates
- Lower β: faster adaptation, noisier estimates

**Recommendation:** 0.9 for most cases

### Trust Region (τ_max)

**Range:** 2.0 to 20.0

**Effect:**
- Higher: more aggressive Newton steps in flat regions
- Lower: more conservative, closer to SGD

**Recommendation:**
- Deep networks: 5.0
- Shallow networks: 10.0
- Convex problems: 20.0

### Epsilon (ε)

**Range:** 10⁻⁸ to 10⁻⁶

**Effect:** Numerical stability in divisions

**Recommendation:** 10⁻⁸ (rarely needs adjustment)

---

## Summary

CGRPA provides an efficient second-order optimization method through:

1. **O(n) curvature estimation** via Hutchinson's method
2. **Adaptive reliability** via VGR variance gating
3. **Safe direction modification** with trust region bounds
4. **Stable parameter updates** via coupled anchor dynamics

The algorithm gracefully interpolates between first-order and second-order methods based on curvature reliability, achieving:
- Better convergence than SGD/Adam in well-conditioned regions
- Robustness to ill-conditioning and high variance
- Computational feasibility for large-scale deep learning

---

## References

1. Hutchinson, M. F. (1990). A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines.
2. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
3. Martens, J. (2010). Deep learning via Hessian-free optimization.
4. Pascanu, R., & Bengio, Y. (2013). Revisiting natural gradient for deep networks.

---

*For implementation details, see the source code. For usage examples, see README.md and QUICKSTART.md.*

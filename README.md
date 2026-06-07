# optimal-transport-rs

**Optimal transport theory for agent systems — Sinkhorn algorithm, Wasserstein distances, barycenters, and JKO gradient flows.**

This crate implements the core algorithms of optimal transport in pure Rust: compute the Sinkhorn entropic regularized transport plan between two distributions, measure Wasserstein-1 and Wasserstein-2 distances (with efficient closed-form solutions for 1D), find weighted barycenters in Wasserstein space, and simulate JKO (Jordan-Kinderlehrer-Otto) gradient flows that evolve distributions over time. With 32 tests across all modules, it provides the geometric toolkit for comparing, interpolating, and evolving probability distributions.

## Why This Matters

Optimal transport is the geometry of moving mass. For AGI systems, it answers fundamental questions: *How different are two beliefs? How do I interpolate between distributions? How does a population evolve toward equilibrium?* The Wasserstein distance respects the metric structure of the underlying space — unlike KL divergence, it doesn't collapse when distributions have disjoint support. Barycenters give you "average" distributions in geometrically meaningful ways. JKO flows model how distributions evolve under potentials, which is the continuous analog of how agent populations respond to incentives. This is the mathematical language for *change with structure*.

## Quick Start

```toml
# Cargo.toml
[dependencies]
optimal-transport-rs = "0.1.0"
```

```rust
use optimal_transport_rs::{Distribution, cost_matrix, euclidean};
use optimal_transport_rs::sinkhorn::sinkhorn;
use optimal_transport_rs::wasserstein::{wasserstein_1d_1, wasserstein_2_squared};
use optimal_transport_rs::barycenter::barycenter;
use optimal_transport_rs::jko_flow::{JKOFlow, PotentialFunctional};

// Create two distributions
let source = Distribution::new(
    vec![0.3, 0.4, 0.3],
    vec![vec![0.0], vec![1.0], vec![2.0]],
);
let target = Distribution::new(
    vec![0.2, 0.5, 0.3],
    vec![vec![0.5], vec![1.5], vec![2.5]],
);

// Sinkhorn transport plan
let result = sinkhorn(&source, &target, 0.1, 100, 1e-6);
println!("Transport cost: {:.4}", result.cost);
println!("Converged: {}", result.converged);

// Wasserstein distances (fast 1D closed-form)
let w1 = wasserstein_1d_1(&source, &target);
let w2sq = wasserstein_2_squared(&source, &target);
println!("W1: {:.4}, W2²: {:.4}", w1, w2sq);

// Barycenter of three distributions
let d1 = Distribution::uniform(vec![vec![0.0], vec![1.0]]);
let d2 = Distribution::uniform(vec![vec![2.0], vec![3.0]]);
let d3 = Distribution::uniform(vec![vec![4.0], vec![5.0]]);
let bary = barycenter(&[d1, d2, d3], &[0.5, 0.3, 0.2], 4, 100, 1e-6);
println!("Barycenter mean: {:.2}", bary.mean()[0]);

// JKO gradient flow toward a quadratic potential
let potential = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
let flow = JKOFlow::new(PotentialFunctional::new(potential), 0.1, 50, 1e-4);
let initial = Distribution::uniform(vec![vec![-2.0], vec![0.0], vec![2.0]]);
let result = flow.run(&initial, 20);
println!("Final distribution mean: {:.4}", result.final_distribution.mean()[0]);
```

## Architecture

| Module | Purpose |
|---|---|
| `sinkhorn` | Entropic regularized OT via Sinkhorn-Knopp iteration |
| `wasserstein` | W1 and W2 distance computation (1D closed-form + multi-dim approximation) |
| `barycenter` | Weighted Fréchet mean in Wasserstein space |
| `jko_flow` | JKO gradient flow for distribution evolution under a functional |

## API Tour

### Core Types (`lib`)

- **`Distribution { weights, points }`** — Discrete probability distribution
  - `::new(weights, points)` — Creates and normalizes weights
  - `::uniform(points)` — Equal-weight distribution
  - `.len()`, `.is_empty()`, `.is_valid()` — Introspection
  - `.mean() → Vec<f64>` — Weighted centroid
- **`cost_matrix(a, b) → Vec<Vec<f64>>`** — Squared Euclidean cost matrix
- **`euclidean(a, b) → f64`**, **`sq_euclidean(a, b) → f64`** — Distance utilities

### Sinkhorn Algorithm (`sinkhorn`)

- **`sinkhorn(source, target, reg, max_iter, tol) → SinkhornResult`**
  - `.plan` — Optimal transport matrix T[i][j]
  - `.cost` — Regularized transport cost
  - `.iterations` — Steps used
  - `.converged` — Whether tolerance was reached

### Wasserstein Distances (`wasserstein`)

- **`wasserstein_1d_1(source, target) → f64`** — Exact W1 for 1D via CDF formula
- **`wasserstein_2_squared(source, target) → f64`** — W2² with 1D quantile matching + multi-dim Sinkhorn fallback

### Barycenter (`barycenter`)

- **`barycenter(distributions, weights, n_support, max_iter, tol) → Distribution`** — Free-support barycenter
  - Uses quantile averaging for 1D, support-point averaging for multi-dim
  - Returns a distribution with `n_support` equally-weighted points

### JKO Gradient Flow (`jko_flow`)

- **`PotentialFunctional { potential, eps }`** — V(x) driving functional
  - Implements `Functional` trait: `.evaluate()`, `.gradient()`
- **`JKOFlow`** — Configure and run gradient flow
  - `::new(functional, step_size, max_iter, tol)`
  - `.run(initial, num_steps) → JKOResult`
  - Result includes trajectory of distributions, functional values, and step costs

## Performance

- Sinkhorn: O(n × m × iterations) — converges in 50-500 iterations for reg ≥ 0.01
- W1 (1D): O(n log n) for sorting, then O(n)
- W2² (1D): O(n × quantile_samples) — typically 1000 samples
- Barycenter: O(k × distributions × support) per iteration
- JKO flow: O(steps × sinkhorn_per_step)
- No external dependencies — pure Rust, no BLAS/LAPACK required

## Ecosystem

Part of the **SuperInstance** family:

- [`wasserstein-agents-rs`](https://github.com/SuperInstance/wasserstein-agents-rs) — Agent systems built on Wasserstein geometry
- [`witness-topology-rs`](https://github.com/SuperInstance/witness-topology-rs) — Topological features from point clouds
- [`renormalization-group-rs`](https://github.com/SuperInstance/renormalization-group-rs) — Multi-scale distribution analysis
- [`sheaf-coherence-rs`](https://github.com/SuperInstance/sheaf-coherence-rs) — Coherence across local data
- [`spectral-prosody-rs`](https://github.com/SuperInstance/spectral-prosody-rs) — Spectral feature extraction

## Ideas for Improvement

- **Log-domain Sinkhorn** — Numerical stability for small regularization
- **Multi-scale Sinkhorn** — ε-scaling for faster convergence
- **Unbalanced OT** — Partial transport for distributions with different total mass
- **Sliced Wasserstein** — O(n log n) approximation for high-dimensional W2
- **GPU Sinkhorn** — Matrix operations map naturally to GPU
- **Neural OT** — Learn the transport map with parameterized networks

## License

MIT

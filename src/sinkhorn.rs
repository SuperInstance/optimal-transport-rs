//! Sinkhorn algorithm for entropic regularized optimal transport.
//!
//! Computes an approximate optimal transport plan by adding an entropic
//! regularization term to the classic linear program.

use crate::{cost_matrix, Distribution};

/// Result of the Sinkhorn algorithm.
#[derive(Debug, Clone)]
pub struct SinkhornResult {
    /// The optimal transport plan (matrix T where T[i][j] is mass moved from i to j).
    pub plan: Vec<Vec<f64>>,
    /// The regularized transport cost.
    pub cost: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
}

/// Run the Sinkhorn-Knopp algorithm for entropic regularized optimal transport.
///
/// # Arguments
/// * `source` - Source distribution
/// * `target` - Target distribution
/// * `reg` - Entropic regularization parameter (higher = more blurred transport)
/// * `max_iter` - Maximum number of iterations
/// * `tol` - Convergence tolerance
pub fn sinkhorn(
    source: &Distribution,
    target: &Distribution,
    reg: f64,
    max_iter: usize,
    tol: f64,
) -> SinkhornResult {
    assert!(reg > 0.0, "regularization must be positive");

    let cost = cost_matrix(&source.points, &target.points);
    let n = source.len();
    let m = target.len();

    // K = exp(-C / reg)
    let mut k: Vec<Vec<f64>> = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            k[i][j] = (-cost[i][j] / reg).exp();
        }
    }

    let mut u = vec![1.0 / n as f64; n];
    let mut v = vec![1.0 / m as f64; m];

    let mut converged = false;
    let mut iter = 0;

    for it in 0..max_iter {
        iter = it + 1;

        // Update u: u = a ./ (K @ v)
        let mut new_u = vec![0.0; n];
        for i in 0..n {
            let sum: f64 = (0..m).map(|j| k[i][j] * v[j]).sum();
            new_u[i] = source.weights[i] / sum.max(1e-300);
        }

        // Update v: v = b ./ (K^T @ u)
        let mut new_v = vec![0.0; m];
        for j in 0..m {
            let sum: f64 = (0..n).map(|i| k[i][j] * new_u[i]).sum();
            new_v[j] = target.weights[j] / sum.max(1e-300);
        }

        // Check convergence
        let diff: f64 = new_u.iter().zip(&u).map(|(a, b)| (a - b).abs()).sum::<f64>()
            + new_v.iter().zip(&v).map(|(a, b)| (a - b).abs()).sum::<f64>();

        u = new_u;
        v = new_v;

        if diff < tol {
            converged = true;
            break;
        }
    }

    // Compute transport plan: T = diag(u) @ K @ diag(v)
    let mut plan = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            plan[i][j] = u[i] * k[i][j] * v[j];
        }
    }

    // Compute cost
    let total_cost: f64 = (0..n)
        .flat_map(|i| (0..m).map(move |j| (i, j)))
        .map(|(i, j)| plan[i][j] * cost[i][j])
        .sum();

    SinkhornResult {
        plan,
        cost: total_cost,
        iterations: iter,
        converged,
    }
}

/// Run Sinkhorn with default parameters (reg=0.1, max_iter=1000, tol=1e-6).
pub fn sinkhorn_default(source: &Distribution, target: &Distribution) -> SinkhornResult {
    sinkhorn(source, target, 0.1, 1000, 1e-6)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Distribution;

    #[test]
    fn test_sinkhorn_identical_distributions() {
        let d = Distribution::uniform(vec![vec![0.0], vec![1.0]]);
        let result = sinkhorn_default(&d, &d);
        assert!(result.converged);
        assert!(result.cost < 1e-3);
    }

    #[test]
    fn test_sinkhorn_preserves_mass() {
        let a = Distribution::new(vec![0.5, 0.5], vec![vec![0.0], vec![1.0]]);
        let b = Distribution::new(vec![0.3, 0.7], vec![vec![0.5], vec![1.5]]);
        let result = sinkhorn_default(&a, &b);
        // Total mass should be ~1
        let total_mass: f64 = result.plan.iter().flat_map(|row| row.iter()).sum();
        assert!((total_mass - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_sinkhorn_converges() {
        let a = Distribution::new(vec![1.0], vec![vec![0.0]]);
        let b = Distribution::new(vec![1.0], vec![vec![5.0]]);
        let result = sinkhorn(&a, &b, 1.0, 1000, 1e-8);
        assert!(result.converged);
        // Cost should be 25 (sq distance from 0 to 5)
        assert!((result.cost - 25.0).abs() < 0.1);
    }

    #[test]
    fn test_sinkhorn_high_regularization() {
        let a = Distribution::uniform(vec![vec![0.0], vec![1.0]]);
        let b = Distribution::uniform(vec![vec![0.0], vec![1.0]]);
        let result = sinkhorn(&a, &b, 10.0, 1000, 1e-8);
        assert!(result.converged);
        // With high reg, plan should be nearly uniform
        assert!(result.plan[0][0] > 0.2);
    }

    #[test]
    fn test_sinkhorn_plan_nonnegative() {
        let a = Distribution::new(vec![0.4, 0.6], vec![vec![0.0], vec![2.0]]);
        let b = Distribution::new(vec![0.7, 0.3], vec![vec![1.0], vec![3.0]]);
        let result = sinkhorn_default(&a, &b);
        for row in &result.plan {
            for &val in row {
                assert!(val >= -1e-10);
            }
        }
    }

    #[test]
    fn test_sinkhorn_iterations_bounded() {
        let a = Distribution::uniform(vec![vec![0.0], vec![1.0]]);
        let b = Distribution::uniform(vec![vec![0.0], vec![1.0]]);
        let result = sinkhorn(&a, &b, 0.1, 50, 1e-8);
        assert!(result.iterations <= 50);
    }
}

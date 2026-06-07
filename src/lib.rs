//! # optimal-transport-rs
//!
//! Optimal transport theory for agent systems.
//!
//! Provides implementations of:
//! - **Sinkhorn algorithm** for entropic regularized optimal transport
//! - **Wasserstein distances** (W1 and W2) between discrete distributions
//! - **Barycenter** computation for weighted combinations of distributions
//! - **JKO gradient flow** for distribution evolution over time

pub mod sinkhorn;
pub mod wasserstein;
pub mod barycenter;
pub mod jko_flow;

/// A discrete probability distribution over a set of points.
#[derive(Debug, Clone)]
pub struct Distribution {
    /// Weights (must sum to 1.0)
    pub weights: Vec<f64>,
    /// Support points (positions)
    pub points: Vec<Vec<f64>>,
}

impl Distribution {
    /// Create a new distribution. Normalizes weights to sum to 1.
    pub fn new(weights: Vec<f64>, points: Vec<Vec<f64>>) -> Self {
        assert_eq!(weights.len(), points.len(), "weights and points must have same length");
        assert!(!weights.is_empty(), "distribution must not be empty");
        let sum: f64 = weights.iter().sum();
        assert!(sum > 0.0, "total weight must be positive");
        let normalized: Vec<f64> = weights.iter().map(|w| w / sum).collect();
        Distribution { weights: normalized, points }
    }

    /// Create a uniform distribution over n points.
    pub fn uniform(points: Vec<Vec<f64>>) -> Self {
        let n = points.len();
        assert!(n > 0, "need at least one point");
        let w = 1.0 / n as f64;
        Distribution { weights: vec![w; n], points }
    }

    /// Number of support points.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// True if distribution is empty.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Validate that weights sum to approximately 1.
    pub fn is_valid(&self) -> bool {
        let sum: f64 = self.weights.iter().sum();
        (sum - 1.0).abs() < 1e-6 && self.weights.iter().all(|&w| w >= 0.0)
    }

    /// Compute weighted mean of the distribution.
    pub fn mean(&self) -> Vec<f64> {
        if self.points.is_empty() {
            return vec![];
        }
        let dim = self.points[0].len();
        let mut result = vec![0.0; dim];
        for (i, point) in self.points.iter().enumerate() {
            for (d, val) in point.iter().enumerate() {
                result[d] += self.weights[i] * val;
            }
        }
        result
    }
}

/// Squared Euclidean distance between two points.
pub fn sq_euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Euclidean distance between two points.
pub fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    sq_euclidean(a, b).sqrt()
}

/// Compute the cost matrix between two sets of points using Euclidean distance squared.
pub fn cost_matrix(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter().map(|p| b.iter().map(|q| sq_euclidean(p, q)).collect()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distribution_creation() {
        let d = Distribution::new(vec![1.0, 2.0, 3.0], vec![vec![0.0], vec![1.0], vec![2.0]]);
        assert_eq!(d.len(), 3);
        assert!(d.is_valid());
        let sum: f64 = d.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_distribution_uniform() {
        let d = Distribution::uniform(vec![vec![0.0], vec![1.0], vec![2.0]]);
        assert_eq!(d.len(), 3);
        for w in &d.weights {
            assert!((w - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_distribution_mean() {
        let d = Distribution::new(vec![1.0, 1.0, 1.0], vec![vec![0.0], vec![3.0], vec![6.0]]);
        let m = d.mean();
        assert!((m[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_distance() {
        let d = euclidean(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_cost_matrix() {
        let a = vec![vec![0.0], vec![1.0]];
        let b = vec![vec![0.0], vec![2.0]];
        let c = cost_matrix(&a, &b);
        assert!((c[0][0] - 0.0).abs() < 1e-10);
        assert!((c[0][1] - 4.0).abs() < 1e-10);
        assert!((c[1][0] - 1.0).abs() < 1e-10);
        assert!((c[1][1] - 1.0).abs() < 1e-10);
    }
}

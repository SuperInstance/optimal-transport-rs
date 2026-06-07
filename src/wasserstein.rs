//! Wasserstein distance computations.
//!
//! Implements Wasserstein-1 (Earth Mover's Distance) and Wasserstein-2
//! distance between discrete distributions.

use crate::{cost_matrix, Distribution};

/// Compute the Wasserstein-1 distance between two 1D distributions.
///
/// Uses the efficient CDF-based formula: W1 = integral |F(x) - G(x)| dx
/// Both distributions must have sorted support points.
pub fn wasserstein_1d_1(source: &Distribution, target: &Distribution) -> f64 {
    assert!(!source.points.is_empty() && !target.points.is_empty());
    assert_eq!(source.points[0].len(), 1, "expected 1D distributions");
    assert_eq!(target.points[0].len(), 1, "expected 1D distributions");

    // Collect all unique positions with their CDF differences
    let mut events: Vec<(f64, f64)> = Vec::new(); // (position, delta)

    for (i, p) in source.points.iter().enumerate() {
        events.push((p[0], source.weights[i]));
    }
    for (i, p) in target.points.iter().enumerate() {
        events.push((p[0], -target.weights[i]));
    }
    events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut cdf_diff: f64 = 0.0;
    let mut distance: f64 = 0.0;
    let mut prev_x = events[0].0;

    for (x, delta) in events {
        distance += (x - prev_x) * cdf_diff.abs();
        cdf_diff += delta;
        prev_x = x;
    }

    distance
}

/// Compute the Wasserstein-2 distance squared between two distributions.
///
/// Uses the closed-form formula for 1D distributions, or a simple
/// discretized approximation for higher dimensions.
pub fn wasserstein_2_squared(source: &Distribution, target: &Distribution) -> f64 {
    if source.points[0].len() == 1 && target.points[0].len() == 1 {
        wasserstein_2d_1d(source, target)
    } else {
        // General case: use Sinkhorn approximation
        wasserstein_2_general(source, target)
    }
}

/// Compute W2^2 for 1D distributions using quantile matching.
fn wasserstein_2d_1d(source: &Distribution, target: &Distribution) -> f64 {
    let n = 1000;
    let mut total = 0.0;

    // Sort source and target by position
    let mut s_items: Vec<(f64, f64)> = source
        .points
        .iter()
        .zip(&source.weights)
        .map(|(p, w)| (p[0], *w))
        .collect();
    s_items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut t_items: Vec<(f64, f64)> = target
        .points
        .iter()
        .zip(&target.weights)
        .map(|(p, w)| (p[0], *w))
        .collect();
    t_items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let dx = 1.0 / n as f64;
    let mut s_idx = 0usize;
    let mut t_idx = 0usize;
    let mut s_cum = 0.0_f64;
    let mut t_cum = 0.0_f64;

    for k in 0..n {
        // Find quantile for source
        let target_cum = (k as f64 + 0.5) * dx;
        while s_cum + s_items[s_idx].1 < target_cum && s_idx < s_items.len() - 1 {
            s_cum += s_items[s_idx].1;
            s_idx += 1;
        }
        let s_val = s_items[s_idx].0;
        while t_cum + t_items[t_idx].1 < target_cum && t_idx < t_items.len() - 1 {
            t_cum += t_items[t_idx].1;
            t_idx += 1;
        }
        let t_val = t_items[t_idx].0;

        total += (s_val - t_val) * (s_val - t_val) * dx;
    }

    total
}

/// General Wasserstein-2^2 using the Sinkhorn approximation.
fn wasserstein_2_general(source: &Distribution, target: &Distribution) -> f64 {
    let cost = cost_matrix(&source.points, &target.points);
    let n = source.len();
    let m = target.len();

    // Simple iterative projection (Sinkhorn-like, single step for approximation)
    // For exact: use the sinkhorn module. Here we use a greedy matching.
    let mut cost_sum = 0.0;
    let mut remaining_target: Vec<f64> = target.weights.clone();

    for (i, cost_row) in cost.iter().enumerate().take(n) {
        let mut remaining_source = source.weights[i];
        let mut indices: Vec<usize> = (0..m).collect();
        indices.sort_by(|&a, &b| cost_row[a].partial_cmp(&cost_row[b]).unwrap());

        for j in indices {
            if remaining_source <= 1e-15 {
                break;
            }
            let transfer = remaining_source.min(remaining_target[j]);
            cost_sum += transfer * cost_row[j];
            remaining_source -= transfer;
            remaining_target[j] -= transfer;
        }
    }

    cost_sum
}

/// Compute Wasserstein-2 distance (not squared).
pub fn wasserstein_2(source: &Distribution, target: &Distribution) -> f64 {
    wasserstein_2_squared(source, target).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Distribution;

    #[test]
    fn test_w1_identical_distributions() {
        let d = Distribution::new(vec![1.0], vec![vec![0.0]]);
        let w = wasserstein_1d_1(&d, &d);
        assert!(w.abs() < 1e-10);
    }

    #[test]
    fn test_w1_delta_distributions() {
        let a = Distribution::new(vec![1.0], vec![vec![0.0]]);
        let b = Distribution::new(vec![1.0], vec![vec![3.0]]);
        let w = wasserstein_1d_1(&a, &b);
        assert!((w - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_w1_uniform_shift() {
        let a = Distribution::new(vec![0.5, 0.5], vec![vec![0.0], vec![1.0]]);
        let b = Distribution::new(vec![0.5, 0.5], vec![vec![2.0], vec![3.0]]);
        let w = wasserstein_1d_1(&a, &b);
        // Shift by 2: W1 should be 2
        assert!((w - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_w2_squared_identical() {
        let d = Distribution::new(vec![1.0], vec![vec![5.0]]);
        let w2 = wasserstein_2_squared(&d, &d);
        assert!(w2.abs() < 1e-10);
    }

    #[test]
    fn test_w2_squared_delta() {
        let a = Distribution::new(vec![1.0], vec![vec![0.0]]);
        let b = Distribution::new(vec![1.0], vec![vec![4.0]]);
        let w2 = wasserstein_2_squared(&a, &b);
        assert!((w2 - 16.0).abs() < 0.5);
    }

    #[test]
    fn test_w2_distance_sqrt() {
        let a = Distribution::new(vec![1.0], vec![vec![0.0]]);
        let b = Distribution::new(vec![1.0], vec![vec![4.0]]);
        let w = wasserstein_2(&a, &b);
        assert!((w - 4.0).abs() < 0.5);
    }

    #[test]
    fn test_w1_symmetry() {
        let a = Distribution::new(vec![0.3, 0.7], vec![vec![0.0], vec![2.0]]);
        let b = Distribution::new(vec![0.6, 0.4], vec![vec![1.0], vec![3.0]]);
        let w_ab = wasserstein_1d_1(&a, &b);
        let w_ba = wasserstein_1d_1(&b, &a);
        assert!((w_ab - w_ba).abs() < 1e-10);
    }

    #[test]
    fn test_w1_triangle_inequality() {
        let a = Distribution::new(vec![1.0], vec![vec![0.0]]);
        let b = Distribution::new(vec![1.0], vec![vec![2.0]]);
        let c = Distribution::new(vec![1.0], vec![vec![5.0]]);
        let w_ab = wasserstein_1d_1(&a, &b);
        let w_bc = wasserstein_1d_1(&b, &c);
        let w_ac = wasserstein_1d_1(&a, &c);
        assert!(w_ac <= w_ab + w_bc + 1e-10);
    }
}

//! Distribution barycenter computation.
//!
//! Computes the weighted Fréchet mean (barycenter) of distributions
//! in Wasserstein space.

use crate::{euclidean, Distribution};

/// Compute the free-support barycenter of distributions using a fixed-point iteration.
///
/// The barycenter minimizes sum_i weight_i * W2^2(barycenter, distributions[i]).
/// For 1D distributions, uses quantile averaging.
///
/// # Arguments
/// * `distributions` - Slice of distributions
/// * `weights` - Weight for each distribution (sums to 1)
/// * `n_support` - Number of support points in the barycenter
/// * `max_iter` - Maximum iterations
/// * `tol` - Convergence tolerance
pub fn barycenter(
    distributions: &[Distribution],
    weights: &[f64],
    n_support: usize,
    max_iter: usize,
    tol: f64,
) -> Distribution {
    assert_eq!(distributions.len(), weights.len(), "must have same number of distributions and weights");
    assert!(!distributions.is_empty(), "need at least one distribution");
    assert!(n_support > 0, "need at least one support point");

    let weight_sum: f64 = weights.iter().sum();
    assert!(weight_sum > 0.0, "weights must sum to positive value");

    // Check if all 1D
    let all_1d = distributions.iter().all(|d| d.points[0].len() == 1);
    if all_1d {
        barycenter_1d(distributions, weights, n_support, max_iter, tol)
    } else {
        // For multi-dim, use support-point averaging
        barycenter_multidim(distributions, weights, n_support, max_iter, tol)
    }
}

/// 1D barycenter using quantile averaging.
fn barycenter_1d(
    distributions: &[Distribution],
    weights: &[f64],
    n_support: usize,
    _max_iter: usize,
    _tol: f64,
) -> Distribution {
    // Average quantiles
    let w_sum: f64 = weights.iter().sum();
    let normalized_weights: Vec<f64> = weights.iter().map(|w| w / w_sum).collect();

    let mut points = Vec::with_capacity(n_support);
    let uniform_w = 1.0 / n_support as f64;

    for k in 0..n_support {
        let q = (k as f64 + 0.5) / n_support as f64; // quantile
        let mut val = 0.0;

        for (di, d) in distributions.iter().enumerate() {
            let mut items: Vec<(f64, f64)> = d
                .points
                .iter()
                .zip(&d.weights)
                .map(|(p, w)| (p[0], *w))
                .collect();
            items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let quantile_val = compute_quantile(&items, q);
            val += normalized_weights[di] * quantile_val;
        }
        points.push(vec![val]);
    }

    Distribution {
        weights: vec![uniform_w; n_support],
        points,
    }
}

/// Compute the quantile value for a sorted distribution.
fn compute_quantile(sorted_items: &[(f64, f64)], q: f64) -> f64 {
    let mut cum = 0.0;
    for (val, weight) in sorted_items {
        cum += weight;
        if cum >= q {
            return *val;
        }
    }
    sorted_items.last().map(|(v, _)| *v).unwrap_or(0.0)
}

/// Multi-dimensional barycenter (fixed-support approximation).
fn barycenter_multidim(
    distributions: &[Distribution],
    weights: &[f64],
    n_support: usize,
    max_iter: usize,
    tol: f64,
) -> Distribution {
    let dim = distributions[0].points[0].len();

    // Initialize barycenter support points as weighted average of all support points
    let total_points: usize = distributions.iter().map(|d| d.points.len()).sum();
    let step = (total_points as f64 / n_support as f64).max(1.0) as usize;

    let all_points: Vec<&[f64]> = distributions.iter().flat_map(|d| d.points.iter().map(|p| p.as_slice())).collect();

    let mut support: Vec<Vec<f64>> = (0..n_support)
        .map(|i| {
            let idx = (i * step).min(all_points.len() - 1);
            all_points[idx].to_vec()
        })
        .collect();

    // Truncate or pad to exact n_support
    support.truncate(n_support);
    while support.len() < n_support {
        support.push(vec![0.0; dim]);
    }

    let uniform_w = 1.0 / n_support as f64;
    let w_sum: f64 = weights.iter().sum();
    let norm_weights: Vec<f64> = weights.iter().map(|w| w / w_sum).collect();

    // Iteratively update support points
    for _ in 0..max_iter {
        let mut new_support = vec![vec![0.0; dim]; n_support];

        for (k, sp) in support.iter().enumerate() {
            for (di, d) in distributions.iter().enumerate() {
                // Find closest point in distribution to this support point
                let (closest, _) = d
                    .points
                    .iter()
                    .map(|p| (p, euclidean(sp, p)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap();

                for (ddim, val) in closest.iter().enumerate() {
                    new_support[k][ddim] += norm_weights[di] * val;
                }
            }
        }

        // Check convergence
        let change: f64 = support
            .iter()
            .zip(&new_support)
            .map(|(old, new)| euclidean(old, new))
            .sum();

        support = new_support;

        if change < tol {
            break;
        }
    }

    Distribution {
        weights: vec![uniform_w; n_support],
        points: support,
    }
}

/// Compute the weighted sum of W2^2 distances from a candidate to each distribution.
/// Useful for evaluating barycenter quality.
pub fn barycenter_objective(
    candidate: &Distribution,
    distributions: &[Distribution],
    weights: &[f64],
) -> f64 {
    let mut total = 0.0;
    for (i, d) in distributions.iter().enumerate() {
        // Approximate W2^2 using greedy matching
        let cost = crate::cost_matrix(&candidate.points, &d.points);
        let n = candidate.len();
        let m = d.len();
        let mut remaining_target: Vec<f64> = d.weights.clone();
        let mut cost_sum = 0.0;

        for (si, cost_row) in cost.iter().enumerate().take(n) {
            let mut remaining_source = candidate.weights[si];
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
        total += weights[i] * cost_sum;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Distribution;

    #[test]
    fn test_barycenter_single_distribution() {
        let d = Distribution::new(vec![0.5, 0.5], vec![vec![0.0], vec![2.0]]);
        let result = barycenter(&[d.clone()], &[1.0], 2, 100, 1e-6);
        assert_eq!(result.len(), 2);
        assert!(result.is_valid());
    }

    #[test]
    fn test_barycenter_two_identical() {
        let d = Distribution::new(vec![0.5, 0.5], vec![vec![0.0], vec![4.0]]);
        let result = barycenter(&[d.clone(), d.clone()], &[0.5, 0.5], 4, 100, 1e-6);
        assert!(result.is_valid());
        // Mean should be near 2.0
        let m = result.mean();
        assert!((m[0] - 2.0).abs() < 0.5);
    }

    #[test]
    fn test_barycenter_symmetric() {
        let a = Distribution::new(vec![1.0], vec![vec![-2.0]]);
        let b = Distribution::new(vec![1.0], vec![vec![2.0]]);
        let result = barycenter(&[a, b], &[0.5, 0.5], 1, 100, 1e-6);
        let m = result.mean();
        assert!(m[0].abs() < 0.5);
    }

    #[test]
    fn test_barycenter_weights() {
        let a = Distribution::new(vec![1.0], vec![vec![0.0]]);
        let b = Distribution::new(vec![1.0], vec![vec![10.0]]);
        let result = barycenter(&[a, b], &[0.9, 0.1], 1, 100, 1e-6);
        let m = result.mean();
        // Should be closer to 0 (weighted 0.9)
        assert!(m[0] < 3.0);
    }

    #[test]
    fn test_barycenter_objective() {
        let d = Distribution::new(vec![1.0], vec![vec![0.0]]);
        let obj = barycenter_objective(&d, &[d.clone()], &[1.0]);
        assert!(obj.abs() < 1e-10);
    }

    #[test]
    fn test_barycenter_n_support() {
        let d = Distribution::uniform(vec![vec![0.0], vec![1.0], vec![2.0]]);
        let result = barycenter(&[d], &[1.0], 10, 100, 1e-6);
        assert_eq!(result.len(), 10);
    }
}

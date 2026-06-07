//! JKO (Jordan-Kinderlehrer-Otto) gradient flow for distribution evolution.
//!
//! Implements the JKO scheme: at each step, the distribution moves in the
//! Wasserstein-2 direction that minimizes the sum of transport cost and
//! the driving functional.

use crate::{euclidean, Distribution};

/// A functional that drives the gradient flow.
pub trait Functional: Send + Sync {
    /// Evaluate the functional at a distribution.
    fn evaluate(&self, dist: &Distribution) -> f64;

    /// Compute the gradient of the functional at each support point.
    fn gradient(&self, dist: &Distribution) -> Vec<Vec<f64>>;
}

/// Potential energy functional: F(μ) = integral V(x) dμ(x)
/// where V is a user-provided potential function.
pub struct PotentialFunctional {
    /// The potential function V(x).
    pub potential: fn(&[f64]) -> f64,
    /// Numerical gradient step size.
    pub eps: f64,
}

impl PotentialFunctional {
    pub fn new(potential: fn(&[f64]) -> f64) -> Self {
        PotentialFunctional { potential, eps: 1e-5 }
    }

    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }
}

impl Functional for PotentialFunctional {
    fn evaluate(&self, dist: &Distribution) -> f64 {
        dist.points
            .iter()
            .zip(&dist.weights)
            .map(|(p, w)| w * (self.potential)(p))
            .sum()
    }

    fn gradient(&self, dist: &Distribution) -> Vec<Vec<f64>> {
        let dim = dist.points[0].len();
        dist.points
            .iter()
            .map(|p| {
                let mut grad = Vec::with_capacity(dim);
                for d in 0..dim {
                    let mut p_plus = p.to_vec();
                    let mut p_minus = p.to_vec();
                    p_plus[d] += self.eps;
                    p_minus[d] -= self.eps;
                    let g = ((self.potential)(&p_plus) - (self.potential)(&p_minus))
                        / (2.0 * self.eps);
                    grad.push(g);
                }
                grad
            })
            .collect()
    }
}

/// Result of a JKO gradient flow step.
#[derive(Debug, Clone)]
pub struct JKOStep {
    /// Distribution after this step.
    pub distribution: Distribution,
    /// Cost of the step (transport + functional change).
    pub step_cost: f64,
    /// Functional value after step.
    pub functional_value: f64,
}

/// Result of a complete JKO flow.
#[derive(Debug, Clone)]
pub struct JKOResult {
    /// All distributions along the flow (including initial).
    pub trajectory: Vec<Distribution>,
    /// Functional values at each step.
    pub functional_values: Vec<f64>,
    /// Step costs.
    pub step_costs: Vec<f64>,
}

/// Run a JKO gradient flow for a given number of steps.
///
/// At each step, the support points are moved in the direction of
/// -∇F (negative gradient of the functional), scaled by the time step.
/// This is a forward-Euler approximation of the JKO scheme.
///
/// # Arguments
/// * `initial` - Starting distribution
/// * `functional` - The driving functional
/// * `tau` - Time step size
/// * `n_steps` - Number of JKO steps
pub fn jko_flow<F: Functional>(
    initial: Distribution,
    functional: &F,
    tau: f64,
    n_steps: usize,
) -> JKOResult {
    let mut trajectory = vec![initial.clone()];
    let mut functional_values = vec![functional.evaluate(&initial)];
    let mut step_costs = Vec::new();

    let mut current = initial;

    for _ in 0..n_steps {
        let grad = functional.gradient(&current);

        // Move each support point in direction -tau * gradient
        let new_points: Vec<Vec<f64>> = current
            .points
            .iter()
            .zip(&grad)
            .map(|(p, g)| {
                p.iter()
                    .zip(g.iter())
                    .map(|(x, gx)| x - tau * gx)
                    .collect()
            })
            .collect();

        // Compute transport cost of the step
        let step_cost: f64 = current
            .points
            .iter()
            .zip(&new_points)
            .zip(&current.weights)
            .map(|((old, new), w)| w * euclidean(old, new).powi(2))
            .sum();

        let new_dist = Distribution {
            weights: current.weights.clone(),
            points: new_points,
        };

        let f_val = functional.evaluate(&new_dist);

        trajectory.push(new_dist.clone());
        functional_values.push(f_val);
        step_costs.push(step_cost);

        current = new_dist;
    }

    JKOResult {
        trajectory,
        functional_values,
        step_costs,
    }
}

/// Run a single JKO step and return the result.
pub fn jko_step<F: Functional>(
    current: &Distribution,
    functional: &F,
    tau: f64,
) -> JKOStep {
    let grad = functional.gradient(current);

    let new_points: Vec<Vec<f64>> = current
        .points
        .iter()
        .zip(&grad)
        .map(|(p, g)| {
            p.iter()
                .zip(g.iter())
                .map(|(x, gx)| x - tau * gx)
                .collect()
        })
        .collect();

    let step_cost: f64 = current
        .points
        .iter()
        .zip(&new_points)
        .zip(&current.weights)
        .map(|((old, new), w)| w * euclidean(old, new).powi(2))
        .sum();

    let new_dist = Distribution {
        weights: current.weights.clone(),
        points: new_points,
    };

    JKOStep {
        functional_value: functional.evaluate(&new_dist),
        distribution: new_dist,
        step_cost,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quadratic_potential(x: &[f64]) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    fn zero_potential(_x: &[f64]) -> f64 {
        0.0
    }

    #[test]
    fn test_potential_functional_evaluate() {
        let f = PotentialFunctional::new(quadratic_potential);
        let d = Distribution::new(vec![1.0], vec![vec![3.0]]);
        let val = f.evaluate(&d);
        assert!((val - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_potential_functional_gradient() {
        let f = PotentialFunctional::new(quadratic_potential).with_eps(1e-5);
        let d = Distribution::new(vec![1.0], vec![vec![2.0]]);
        let grad = f.gradient(&d);
        // d/dx(x^2) = 2x, so gradient at x=2 should be 4
        assert!((grad[0][0] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_jko_single_step() {
        let f = PotentialFunctional::new(quadratic_potential);
        let d = Distribution::new(vec![1.0], vec![vec![2.0]]);
        let step = jko_step(&d, &f, 0.1);
        // Should move toward origin (gradient of x^2 at x=2 is 4, so move by -0.4)
        assert!(step.distribution.points[0][0] < 2.0);
        assert!(step.functional_value < f.evaluate(&d));
    }

    #[test]
    fn test_jko_flow_converges_to_minimum() {
        let f = PotentialFunctional::new(quadratic_potential);
        let d = Distribution::new(vec![1.0], vec![vec![5.0]]);
        let result = jko_flow(d, &f, 0.1, 100);
        // After many steps, should be near origin
        let final_pos = result.trajectory.last().unwrap().points[0][0];
        assert!(final_pos.abs() < 0.5);
    }

    #[test]
    fn test_jko_flow_decreases_functional() {
        let f = PotentialFunctional::new(quadratic_potential);
        let d = Distribution::new(vec![0.5, 0.5], vec![vec![3.0], vec![-3.0]]);
        let result = jko_flow(d, &f, 0.05, 50);
        // Functional should decrease
        assert!(result.functional_values.last().unwrap() < &result.functional_values[0]);
    }

    #[test]
    fn test_jko_zero_potential_no_movement() {
        let f = PotentialFunctional::new(zero_potential);
        let d = Distribution::new(vec![1.0], vec![vec![5.0]]);
        let result = jko_flow(d, &f, 0.1, 10);
        // With zero potential, no movement
        let final_pos = result.trajectory.last().unwrap().points[0][0];
        assert!((final_pos - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_jko_trajectory_length() {
        let f = PotentialFunctional::new(quadratic_potential);
        let d = Distribution::new(vec![1.0], vec![vec![1.0]]);
        let result = jko_flow(d, &f, 0.1, 5);
        assert_eq!(result.trajectory.len(), 6); // initial + 5 steps
        assert_eq!(result.functional_values.len(), 6);
        assert_eq!(result.step_costs.len(), 5);
    }
}

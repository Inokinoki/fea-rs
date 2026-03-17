//! Advanced acceleration techniques for FEA solvers.
#![allow(non_snake_case)]
#![allow(unused_assignments)]
//!
//! This module provides:
//! - Riesz acceleration methods
//! - Multipoint extrapolation
//! - Vector sequence acceleration
//! - Convergence enhancement techniques

use nalgebra::DVector;

/// Riesz acceleration for gradient methods.
///
/// Uses Riesz representation to accelerate convergence
/// of gradient-based iterative methods.
pub struct RieszAcceleration {
    /// Memory depth for acceleration.
    depth: usize,
    /// Previous gradients.
    gradients: Vec<DVector<f64>>,
    /// Previous steps.
    steps: Vec<DVector<f64>>,
    /// Current step size estimate.
    step_size: f64,
}

impl RieszAcceleration {
    /// Creates new Riesz acceleration.
    pub fn new(depth: usize) -> Self {
        Self {
            depth,
            gradients: Vec::with_capacity(depth),
            steps: Vec::with_capacity(depth),
            step_size: 1.0,
        }
    }

    /// Updates accelerator with new gradient and returns accelerated step.
    pub fn update(&mut self, x: &DVector<f64>, gradient: &DVector<f64>) -> DVector<f64> {
        // Store history
        if self.gradients.len() >= self.depth {
            self.gradients.remove(0);
            self.steps.remove(0);
        }

        self.gradients.push(gradient.clone());
        self.steps.push(x.clone());

        if self.gradients.len() < 2 {
            return x - gradient.scale(self.step_size);
        }

        // Compute optimal step using Riesz representation
        let m = self.gradients.len();
        let grad_curr = &self.gradients[m - 1];
        let grad_prev = &self.gradients[m - 2];

        let y = grad_curr - grad_prev;
        let s = &self.steps[m - 1] - &self.steps[m - 2];

        let sy = s.dot(&y);
        let yy = y.dot(&y);

        if yy > 1e-15 && sy > 0.0 {
            self.step_size = (sy / yy).clamp(0.1, 10.0);
        }

        x - grad_curr.scale(self.step_size)
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.gradients.clear();
        self.steps.clear();
        self.step_size = 1.0;
    }
}

/// Multipoint extrapolation acceleration.
///
/// Uses multiple previous iterates to extrapolate the solution.
pub struct MultipointExtrapolation {
    /// Number of points for extrapolation.
    n_points: usize,
    /// History of iterates.
    history: Vec<DVector<f64>>,
    /// Weights for extrapolation.
    weights: Vec<f64>,
}

impl MultipointExtrapolation {
    /// Creates new multipoint extrapolation.
    pub fn new(n_points: usize) -> Self {
        Self {
            n_points,
            history: Vec::with_capacity(n_points),
            weights: Self::compute_weights(n_points),
        }
    }

    /// Computes extrapolation weights.
    fn compute_weights(n: usize) -> Vec<f64> {
        // Simple linear extrapolation weights
        // For n=3: w = [-1, 2, 0] gives x_new = 2*x_{n-1} - x_{n-2}
        let mut weights = vec![0.0; n];
        if n >= 2 {
            weights[n - 2] = -1.0;
            weights[n - 1] = 2.0;
        } else if n == 1 {
            weights[0] = 1.0;
        }
        weights
    }

    /// Updates with new iterate and returns extrapolated value.
    pub fn update(&mut self, x: &DVector<f64>) -> DVector<f64> {
        if self.history.len() >= self.n_points {
            self.history.remove(0);
        }
        self.history.push(x.clone());

        if self.history.len() < self.n_points {
            return x.clone();
        }

        // Compute extrapolated value
        let mut x_extrap = DVector::zeros(x.len());
        for (i, w) in self.weights.iter().enumerate() {
            if *w != 0.0 {
                x_extrap += self.history[i].scale(*w);
            }
        }

        x_extrap
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

/// Vector epsilon algorithm for sequence acceleration.
///
/// Wynn's epsilon algorithm generalized to vectors.
pub struct VectorEpsilonAlgorithm {
    /// History size.
    max_history: usize,
    /// Iterate history.
    history: Vec<DVector<f64>>,
    /// Epsilon table (flattened).
    epsilon: Vec<Vec<DVector<f64>>>,
}

impl VectorEpsilonAlgorithm {
    /// Creates new vector epsilon algorithm.
    pub fn new(max_history: usize) -> Self {
        Self {
            max_history,
            history: Vec::with_capacity(max_history),
            epsilon: Vec::new(),
        }
    }

    /// Updates with new iterate.
    pub fn update(&mut self, x: &DVector<f64>) -> Option<DVector<f64>> {
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(x.clone());

        if self.history.len() < 3 {
            return None;
        }

        // Build epsilon table
        let n = self.history.len();

        // epsilon[-1] = 0, epsilon[0] = x
        let mut eps_table: Vec<Vec<DVector<f64>>> = vec![vec![]; 2 * n - 1];

        for i in 0..n {
            eps_table[2 * i].push(self.history[i].clone());
        }

        // Wynn's recurrence (simplified for vectors)
        for k in 1..n {
            for i in 0..n - k {
                let idx = 2 * (i + k) - 1;
                let prev_idx = 2 * (i + k - 1);

                if idx < eps_table.len() && prev_idx < eps_table.len() {
                    // Check bounds before accessing
                    if eps_table[prev_idx].len() > i + 1 && eps_table[prev_idx].len() > i {
                        let diff = &eps_table[prev_idx][i] - &eps_table[prev_idx][i + 1];
                        let norm_sq = diff.dot(&diff);

                        if norm_sq > 1e-15 {
                            let eps_val = if 2 * (i + k - 2) < eps_table.len() && eps_table[2 * (i + k - 2)].len() > i {
                                eps_table[2 * (i + k - 2)][i].clone()
                            } else {
                                DVector::zeros(x.len())
                            };
                            if let Some(v) = eps_table.get_mut(idx) {
                                v.push(eps_val + diff.scale(1.0 / norm_sq));
                            }
                        } else {
                            if let Some(v) = eps_table.get_mut(idx) {
                                v.push(DVector::zeros(x.len()));
                            }
                        }
                    }
                }
            }
        }

        // Return most accelerated value
        eps_table.last().and_then(|v| v.first().cloned())
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.history.clear();
        self.epsilon.clear();
    }
}

/// Topological epsilon algorithm.
pub struct TopologicalEpsilon {
    depth: usize,
    x_history: Vec<DVector<f64>>,
    r_history: Vec<DVector<f64>>,
}

impl TopologicalEpsilon {
    /// Creates new topological epsilon.
    pub fn new(depth: usize) -> Self {
        Self {
            depth,
            x_history: Vec::with_capacity(depth),
            r_history: Vec::with_capacity(depth),
        }
    }

    /// Updates with new iterate and residual.
    pub fn update(&mut self, x: &DVector<f64>, r: &DVector<f64>) -> DVector<f64> {
        if self.x_history.len() >= self.depth {
            self.x_history.remove(0);
            self.r_history.remove(0);
        }

        self.x_history.push(x.clone());
        self.r_history.push(r.clone());

        if self.x_history.len() < 2 {
            return x.clone();
        }

        // Topological epsilon update
        let m = self.x_history.len();
        let mut x_new = x.clone();

        for i in 1..m {
            let dx = &self.x_history[i] - &self.x_history[i - 1];
            let dr = &self.r_history[i] - &self.r_history[i - 1];

            let dx_dot_dr = dx.dot(&dr);
            let dr_dot_dr = dr.dot(&dr);

            if dr_dot_dr > 1e-15 && dx_dot_dr.abs() > 1e-15 {
                let alpha = dx_dot_dr / dr_dot_dr;
                x_new += dx.scale(alpha);
            }
        }

        x_new / m as f64
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.x_history.clear();
        self.r_history.clear();
    }
}

/// Generalized vector extrapolation (GVE).
pub struct GeneralizedVectorExtrapolation {
    /// Extrapolation order.
    order: usize,
    /// History of iterates.
    x_history: Vec<DVector<f64>>,
    /// History of differences.
    d_history: Vec<DVector<f64>>,
}

impl GeneralizedVectorExtrapolation {
    /// Creates new GVE.
    pub fn new(order: usize) -> Self {
        Self {
            order,
            x_history: Vec::with_capacity(order),
            d_history: Vec::with_capacity(order),
        }
    }

    /// Updates and returns extrapolated value.
    pub fn update(&mut self, x: &DVector<f64>) -> DVector<f64> {
        if self.x_history.len() >= self.order {
            self.x_history.remove(0);
            self.d_history.remove(0);
        }

        // Compute difference
        if let Some(x_prev) = self.x_history.last() {
            self.d_history.push(x - x_prev);
        }
        self.x_history.push(x.clone());

        if self.x_history.len() < 2 {
            return x.clone();
        }

        // GVE extrapolation
        let m = self.x_history.len();
        let mut x_extrap = self.x_history[m - 1].clone();

        // Compute weighted sum of differences
        for i in 0..m - 1 {
            let weight = (i + 1) as f64 / m as f64;
            x_extrap += self.d_history[i].scale(weight * 0.1);
        }

        x_extrap
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.x_history.clear();
        self.d_history.clear();
    }
}

/// Steffensen-type acceleration with adaptive parameters.
pub struct AdaptiveSteffensen {
    /// Current damping parameter.
    damping: f64,
    /// Previous iterate.
    x_prev: Option<DVector<f64>>,
    /// Previous function value.
    f_prev: Option<DVector<f64>>,
    /// Success count for adaptation.
    success_count: usize,
    /// Failure count for adaptation.
    failure_count: usize,
}

impl AdaptiveSteffensen {
    /// Creates new adaptive Steffensen.
    pub fn new(initial_damping: f64) -> Self {
        Self {
            damping: initial_damping.clamp(0.01, 1.0),
            x_prev: None,
            f_prev: None,
            success_count: 0,
            failure_count: 0,
        }
    }

    /// Applies Steffensen iteration with adaptive damping.
    pub fn update<F>(&mut self, x: &DVector<f64>, f: F) -> DVector<f64>
    where
        F: Fn(&DVector<f64>) -> DVector<f64>,
    {
        let fx = f(x);

        if let (Some(xp), Some(fp)) = (&self.x_prev, &self.f_prev) {
            let dx = x - xp;
            let df = &fx - fp;

            let dx_norm = dx.dot(&dx);
            let df_norm = df.dot(&df);

            if df_norm > 1e-15 {
                let alpha = (dx.dot(&df) / df_norm).clamp(-1.0, 1.0);
                let x_new = x - fx.scale(self.damping * alpha);

                // Check if this improved convergence
                let f_new = f(&x_new);
                if f_new.dot(&f_new) < fx.dot(&fx) {
                    self.success_count += 1;
                    if self.success_count >= 3 {
                        self.damping = (self.damping * 1.1).min(1.0);
                        self.success_count = 0;
                    }
                    return x_new;
                } else {
                    self.failure_count += 1;
                    if self.failure_count >= 2 {
                        self.damping = (self.damping * 0.9).max(0.01);
                        self.failure_count = 0;
                    }
                }
            }
        }

        self.x_prev = Some(x.clone());
        self.f_prev = Some(fx.clone());
        x - fx.scale(self.damping)
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.x_prev = None;
        self.f_prev = None;
        self.success_count = 0;
        self.failure_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_riesz_acceleration() {
        let mut riesz = RieszAcceleration::new(5);
        let mut x = DVector::from_element(10, 1.0);

        for _ in 0..20 {
            let grad = x.scale(0.5); // Simple quadratic
            x = riesz.update(&x, &grad);
        }

        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_multipoint_extrapolation() {
        let mut mpe = MultipointExtrapolation::new(3);
        let x = DVector::from_element(10, 1.0);

        let _ = mpe.update(&x);
        let _ = mpe.update(&x.scale(0.9));
        let result = mpe.update(&x.scale(0.81));

        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_vector_epsilon() {
        let mut eps = VectorEpsilonAlgorithm::new(5);
        let x = DVector::from_element(10, 1.0);

        let _ = eps.update(&x);
        let _ = eps.update(&x.scale(0.5));
        let _ = eps.update(&x.scale(0.25));

        // May or may not return accelerated value
    }

    #[test]
    fn test_topological_epsilon() {
        let mut top_eps = TopologicalEpsilon::new(5);
        let x = DVector::from_element(10, 0.5);
        let r = DVector::from_element(10, 0.1);

        let x_new = top_eps.update(&x, &r);
        assert!(x_new.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_gve() {
        let mut gve = GeneralizedVectorExtrapolation::new(5);
        let x = DVector::from_element(10, 1.0);

        for i in 0..5 {
            let x_new = DVector::from_element(10, 1.0 / (i + 1) as f64);
            let _ = gve.update(&x_new);
        }

        assert!(gve.x_history.len() > 0);
    }

    #[test]
    fn test_adaptive_steffensen() {
        let mut steff = AdaptiveSteffensen::new(0.5);
        let mut x = DVector::from_element(10, 1.0);

        let f = |x: &DVector<f64>| x.scale(0.5);

        for _ in 0..20 {
            x = steff.update(&x, &f);
        }

        assert!(x.iter().all(|v| v.is_finite()));
    }
}

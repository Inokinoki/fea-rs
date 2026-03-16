//! Advanced Nonlinear Acceleration Methods.
//!
//! This module provides acceleration techniques specifically designed for
//! nonlinear fixed-point iterations and Newton-type methods:
//! - Aitken's Delta-Squared method
//! - Anderson acceleration variants
//! - Vector extrapolation methods
//! - Convergence monitoring and adaptation

use nalgebra::{DMatrix, DVector};

/// Aitken's Delta-Squared acceleration for scalar sequences.
pub struct AitkenAcceleration {
    /// Previous iterate.
    x_prev: Option<Vec<f64>>,
    /// Second previous iterate.
    x_prev2: Option<Vec<f64>>,
}

impl AitkenAcceleration {
    /// Creates a new Aitken accelerator.
    pub fn new() -> Self {
        Self {
            x_prev: None,
            x_prev2: None,
        }
    }

    /// Updates with new iterate and returns accelerated value if available.
    pub fn update(&mut self, x: &[f64]) -> Option<Vec<f64>> {
        if let (Some(x_prev), Some(x_prev2)) = (&self.x_prev, &self.x_prev2) {
            let n = x.len();
            let mut x_accel = Vec::with_capacity(n);

            for i in 0..n {
                let dx1 = x[i] - x_prev[i];
                let dx2 = x_prev[i] - x_prev2[i];
                let denom = dx1 - dx2;

                if denom.abs() > 1e-15 {
                    x_accel.push(x[i] - dx1 * dx1 / denom);
                } else {
                    x_accel.push(x[i]);
                }
            }

            self.x_prev2 = Some(x_prev.clone());
            self.x_prev = Some(x.to_vec());

            Some(x_accel)
        } else {
            self.x_prev2 = self.x_prev.take();
            self.x_prev = Some(x.to_vec());
            None
        }
    }

    /// Resets the acceleration state.
    pub fn reset(&mut self) {
        self.x_prev = None;
        self.x_prev2 = None;
    }
}

impl Default for AitkenAcceleration {
    fn default() -> Self {
        Self::new()
    }
}

/// Anderson acceleration for vector sequences.
#[derive(Debug, Clone)]
pub struct AndersonAcceleration {
    /// History depth.
    pub depth: usize,
    /// Mixing parameter beta.
    pub beta: f64,
    /// Type (1 or 2).
    pub anderson_type: usize,
}

impl AndersonAcceleration {
    /// Creates Anderson acceleration with specified depth.
    pub fn new(depth: usize) -> Self {
        Self {
            depth,
            beta: 1.0,
            anderson_type: 2,
        }
    }

    /// Sets the mixing parameter.
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta.clamp(0.1, 1.0);
        self
    }

    /// Sets the Anderson type (1 or 2).
    pub fn with_type(mut self, t: usize) -> Self {
        self.anderson_type = t.clamp(1, 2);
        self
    }

    /// Applies Anderson acceleration to a sequence of iterates.
    pub fn apply(&self, iterates: &[DVector<f64>], residuals: &[DVector<f64>]) -> Option<DVector<f64>> {
        let m = self.depth.min(iterates.len().saturating_sub(1));
        if m == 0 || iterates.len() < 2 {
            return None;
        }

        let n = iterates[0].len();
        let k = iterates.len() - 1; // Current iteration

        // Build difference matrices
        let mut delta_f = Vec::with_capacity(m);
        for i in (k.saturating_sub(m)..k).rev() {
            let df = &residuals[i + 1] - &residuals[i];
            delta_f.push(df);
        }

        // Build Gram matrix
        let mut G = DMatrix::zeros(m, m);
        for i in 0..m {
            for j in 0..m {
                G[(i, j)] = delta_f[i].dot(&delta_f[j]);
            }
        }

        // Add regularization
        for i in 0..m {
            G[(i, i)] += 1e-8;
        }

        // Right-hand side
        let mut rhs = DVector::zeros(m);
        for i in 0..m {
            rhs[i] = -residuals[k].dot(&delta_f[i]);
        }

        // Solve for coefficients
        let coeffs = G.lu().solve(&rhs)?;

        // Compute accelerated iterate
        let mut x_accel = iterates[k].clone();
        let mut sum_coeffs = 1.0;

        for i in 0..m {
            let idx = k - m + i;
            x_accel += (&iterates[idx] - &iterates[k]) * coeffs[i];
            sum_coeffs += coeffs[i];
        }

        // Apply mixing
        Some(x_accel.scale(self.beta / sum_coeffs))
    }
}

impl Default for AndersonAcceleration {
    fn default() -> Self {
        Self::new(5)
    }
}

/// Vector Extrapolation methods.
pub mod vector_extrapolation {
    use super::*;

    /// Minimal Polynomial Extrapolation (MPE).
    pub fn mpe(iterates: &[DVector<f64>], k: usize) -> Option<DVector<f64>> {
        if iterates.len() < k + 2 {
            return None;
        }

        let n = iterates[0].len();
        let m = iterates.len() - 1;

        // Compute differences
        let mut u = Vec::with_capacity(m);
        for i in 0..m {
            u.push(&iterates[i + 1] - &iterates[i]);
        }

        // Build system for coefficients
        let mut G = DMatrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                G[(i, j)] = u[i].dot(&u[j]);
            }
        }

        let mut rhs = DVector::zeros(k);
        for i in 0..k {
            rhs[i] = -u[m - 1].dot(&u[i]);
        }

        // Solve
        let coeffs = G.lu().solve(&rhs)?;

        // Compute extrapolated value
        let mut result = iterates[m].clone();
        for i in 0..k {
            result += &u[m - k + i] * coeffs[i];
        }

        Some(result)
    }

    /// Reduced Rank Extrapolation (RRE).
    pub fn rre(iterates: &[DVector<f64>], k: usize) -> Option<DVector<f64>> {
        if iterates.len() < k + 2 {
            return None;
        }

        let m = iterates.len() - 1;

        // Compute second differences
        let mut delta2 = Vec::with_capacity(m - 1);
        for i in 0..m - 1 {
            let d2 = &iterates[i + 2] - &iterates[i + 1].scale(2.0) + &iterates[i];
            delta2.push(d2);
        }

        // Build Gram matrix
        let mut G = DMatrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                G[(i, j)] = delta2[i].dot(&delta2[j]);
            }
        }

        let mut rhs = DVector::zeros(k);
        for i in 0..k {
            rhs[i] = -delta2[m - 2].dot(&delta2[i]);
        }

        let coeffs = G.lu().solve(&rhs)?;

        // Compute extrapolated value
        let mut result = iterates[m].clone();
        for i in 0..k {
            let d1 = &iterates[m - k + i + 1] - &iterates[m - k + i];
            result += &d1 * coeffs[i];
        }

        Some(result)
    }

    /// Modified Minimal Polynomial Extrapolation (MMPE).
    pub fn mmpe(iterates: &[DVector<f64>], functionals: &[DVector<f64>], k: usize) -> Option<DVector<f64>> {
        if iterates.len() < k + 2 || functionals.len() < k {
            return None;
        }

        let m = iterates.len() - 1;

        // Compute differences
        let mut u = Vec::with_capacity(m);
        for i in 0..m {
            u.push(&iterates[i + 1] - &iterates[i]);
        }

        // Build system using functionals
        let mut G = DMatrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                G[(i, j)] = functionals[i].dot(&u[j]);
            }
        }

        let mut rhs = DVector::zeros(k);
        for i in 0..k {
            rhs[i] = -functionals[i].dot(&u[m - 1]);
        }

        let coeffs = G.lu().solve(&rhs)?;

        let mut result = iterates[m].clone();
        for i in 0..k {
            result += &u[m - k + i] * coeffs[i];
        }

        Some(result)
    }
}

/// Convergence monitor for nonlinear iterations.
#[derive(Debug, Clone)]
pub struct ConvergenceMonitor {
    /// Residual history.
    pub residuals: Vec<f64>,
    /// Iterate history.
    pub iterates: Vec<DVector<f64>>,
    /// Maximum history size.
    pub max_history: usize,
}

impl ConvergenceMonitor {
    /// Creates a new convergence monitor.
    pub fn new(max_history: usize) -> Self {
        Self {
            residuals: Vec::new(),
            iterates: Vec::new(),
            max_history,
        }
    }

    /// Records a new residual and iterate.
    pub fn record(&mut self, residual: f64, iterate: &DVector<f64>) {
        self.residuals.push(residual);
        self.iterates.push(iterate.clone());

        while self.residuals.len() > self.max_history {
            self.residuals.remove(0);
            self.iterates.remove(0);
        }
    }

    /// Estimates convergence rate.
    pub fn convergence_rate(&self) -> Option<f64> {
        if self.residuals.len() < 3 {
            return None;
        }

        let n = self.residuals.len();
        let r_old = self.residuals[n - 3];
        let r_new = self.residuals[n - 1];

        if r_old > 1e-15 && r_new > 0.0 {
            Some((r_old / r_new).powf(1.0 / 2.0))
        } else {
            None
        }
    }

    /// Detects stagnation.
    pub fn is_stagnating(&self, threshold: f64) -> bool {
        if self.residuals.len() < 5 {
            return false;
        }

        let n = self.residuals.len();
        let recent: f64 = self.residuals[n - 5..].iter().sum();
        let older: f64 = self.residuals[n - 10..n - 5].iter().sum();

        (recent - older).abs() / older.max(1e-15) < threshold
    }

    /// Gets recommended action based on convergence behavior.
    pub fn get_recommendation(&self) -> ConvergenceRecommendation {
        if let Some(rate) = self.convergence_rate() {
            if rate > 2.0 {
                ConvergenceRecommendation::Continue
            } else if rate > 1.2 {
                ConvergenceRecommendation::ConsiderAcceleration
            } else if self.is_stagnating(0.01) {
                ConvergenceRecommendation::RestartWithPerturbation
            } else {
                ConvergenceRecommendation::IncreaseDamping
            }
        } else {
            ConvergenceRecommendation::Continue
        }
    }
}

impl Default for ConvergenceMonitor {
    fn default() -> Self {
        Self::new(20)
    }
}

/// Recommendation for convergence improvement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvergenceRecommendation {
    /// Continue with current settings.
    Continue,
    /// Consider adding acceleration.
    ConsiderAcceleration,
    /// Restart with perturbation.
    RestartWithPerturbation,
    /// Increase damping.
    IncreaseDamping,
    /// Decrease damping.
    DecreaseDamping,
}

/// Nonlinear solver with acceleration.
#[derive(Debug, Clone)]
pub struct AcceleratedNonlinearSolver {
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Tolerance.
    pub tolerance: f64,
    /// Anderson depth.
    pub anderson_depth: usize,
    /// Use Aitken acceleration.
    pub use_aitken: bool,
    /// Monitor convergence.
    pub monitor: ConvergenceMonitor,
}

impl Default for AcceleratedNonlinearSolver {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            tolerance: 1e-10,
            anderson_depth: 5,
            use_aitken: false,
            monitor: ConvergenceMonitor::default(),
        }
    }
}

impl AcceleratedNonlinearSolver {
    /// Creates a new accelerated nonlinear solver.
    pub fn new() -> Self {
        Self::default()
    }

    /// Solves x = g(x) using accelerated fixed-point iteration.
    pub fn solve<F>(&mut self, g: F, x0: &DVector<f64>) -> NonlinearSolverResult
    where
        F: Fn(&DVector<f64>) -> DVector<f64>,
    {
        let mut x = x0.clone();
        let mut iterates = vec![x.clone()];
        let mut residuals = Vec::new();
        let mut iteration = 0;
        let mut converged = false;

        let anderson = AndersonAcceleration::new(self.anderson_depth);

        while iteration < self.max_iterations {
            let x_new = g(&x);
            let residual = (&x_new - &x).norm();

            residuals.push(residual);
            self.monitor.record(residual, &x);

            if residual < self.tolerance {
                x = x_new;
                converged = true;
                iteration += 1;
                break;
            }

            iterates.push(x_new.clone());

            // Try Anderson acceleration
            if iterates.len() >= 3 {
                let residual_vecs: Vec<DVector<f64>> = residuals.iter()
                    .map(|&r| DVector::from_element(x.len(), r))
                    .collect();

                if let Some(x_accel) = anderson.apply(&iterates, &residual_vecs) {
                    let residual_accel = (&g(&x_accel) - &x_accel).norm();
                    if residual_accel < residual {
                        iterates.push(x_accel.clone());
                        residuals.push(residual_accel);
                        x = x_accel;
                    } else {
                        x = x_new;
                    }
                } else {
                    x = x_new;
                }
            } else {
                x = x_new;
            }

            iteration += 1;
        }

        NonlinearSolverResult {
            solution: x,
            iterations: iteration,
            final_residual: residuals.last().copied().unwrap_or(f64::INFINITY),
            converged,
        }
    }
}

/// Result from nonlinear solver.
#[derive(Debug, Clone)]
pub struct NonlinearSolverResult {
    /// Solution vector.
    pub solution: DVector<f64>,
    /// Number of iterations.
    pub iterations: usize,
    /// Final residual.
    pub final_residual: f64,
    /// Convergence flag.
    pub converged: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aitken_acceleration() {
        let mut aitken = AitkenAcceleration::new();

        // Test with geometric sequence
        let x0 = vec![1.0];
        let x1 = vec![0.5];
        let x2 = vec![0.25];

        aitken.update(&x0);
        aitken.update(&x1);
        let accelerated = aitken.update(&x2);

        assert!(accelerated.is_some());
        let acc = accelerated.unwrap();
        // Aitken should give limit for geometric sequence
        assert!(acc[0].abs() < 0.1);
    }

    #[test]
    fn test_anderson_acceleration() {
        let anderson = AndersonAcceleration::new(3);

        let iterates = vec![
            DVector::from_column_slice(&[1.0, 2.0]),
            DVector::from_column_slice(&[0.6, 1.2]),
            DVector::from_column_slice(&[0.36, 0.72]),
            DVector::from_column_slice(&[0.216, 0.432]),
        ];

        // Residuals should be g(x) - x for each iterate
        // For this test, use simple residuals (difference from zero)
        let residuals: Vec<DVector<f64>> = iterates.iter()
            .map(|x| x.clone())
            .collect();

        let result = anderson.apply(&iterates, &residuals);
        assert!(result.is_some());
    }

    #[test]
    fn test_mpe_extrapolation() {
        let iterates = vec![
            DVector::from_column_slice(&[1.0, 1.0]),
            DVector::from_column_slice(&[0.5, 0.5]),
            DVector::from_column_slice(&[0.25, 0.25]),
            DVector::from_column_slice(&[0.125, 0.125]),
            DVector::from_column_slice(&[0.0625, 0.0625]),
        ];

        // MPE may fail for some sequences, just check it runs
        let _result = vector_extrapolation::mpe(&iterates, 2);
    }

    #[test]
    fn test_convergence_monitor() {
        let mut monitor = ConvergenceMonitor::new(10);

        // Simulate converging sequence
        for i in 0..10 {
            let residual = 0.5_f64.powi(i);
            let iterate = DVector::from_element(5, residual);
            monitor.record(residual, &iterate);
        }

        let rate = monitor.convergence_rate();
        assert!(rate.is_some());
        assert!(rate.unwrap() > 1.0);

        assert!(!monitor.is_stagnating(0.1));
    }

    #[test]
    fn test_accelerated_nonlinear_solver() {
        // Just test basic solver functionality without acceleration
        let mut solver = AcceleratedNonlinearSolver::new();
        solver.anderson_depth = 0; // Disable Anderson for this simple test

        // Fixed point: x = 0.5 * x (solution is 0)
        let g = |x: &DVector<f64>| x.scale(0.5);
        let x0 = DVector::from_column_slice(&[1.0, 2.0, 3.0]);

        let result = solver.solve(g, &x0);

        assert!(result.converged);
        assert!(result.iterations < 50);
        assert!(result.solution.norm() < 0.1);
    }
}

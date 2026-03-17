//! Aitken acceleration for iterative methods.
#![allow(non_snake_case)]
#![allow(unused_assignments)]
//!
//! Aitken's delta-squared process accelerates the convergence
//! of linearly convergent sequences.

/// Aitken's delta-squared acceleration.
///
/// Given a sequence x_n converging to x, computes accelerated estimate:
/// x_accelerated = x_n - (x_n - x_{n-1})^2 / (x_n - 2*x_{n-1} + x_{n-2})
#[derive(Debug, Clone, Default)]
pub struct AitkenAcceleration {
    /// Previous iterate.
    pub x_prev: Option<Vec<f64>>,
    /// Second previous iterate.
    pub x_prev2: Option<Vec<f64>>,
}

impl AitkenAcceleration {
    /// Creates a new Aitken accelerator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates with new iterate and returns accelerated value if available.
    pub fn update(&mut self, x: &[f64]) -> Option<Vec<f64>> {
        if let (Some(x_prev), Some(x_prev2)) = (&self.x_prev, &self.x_prev2) {
            // Compute Aitken acceleration
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
            // Not enough history yet
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

/// Minimal residual smoothing for iterative solvers.
#[derive(Debug, Clone, Default)]
pub struct MinimalResidualSmoothing {
    /// Previous smoothed residual.
    pub r_smooth: Option<Vec<f64>>,
}

impl MinimalResidualSmoothing {
    /// Creates a new smoother.
    pub fn new() -> Self {
        Self::default()
    }

    /// Applies minimal residual smoothing.
    ///
    /// Given current residual r and direction p, finds optimal alpha:
    /// alpha = -r^T*p / (p^T*p)
    /// r_smooth = r + alpha * p
    pub fn smooth(&mut self, r: &[f64], p: &[f64]) -> Vec<f64> {
        let n = r.len();
        let r_dot_p: f64 = r.iter().zip(p.iter()).map(|(a, b)| a * b).sum();
        let p_dot_p: f64 = p.iter().map(|v| v * v).sum();

        if p_dot_p > 1e-15 {
            let alpha = -r_dot_p / p_dot_p;
            let mut r_smooth = Vec::with_capacity(n);
            for i in 0..n {
                r_smooth.push(r[i] + alpha * p[i]);
            }
            self.r_smooth = Some(r_smooth.clone());
            r_smooth
        } else {
            self.r_smooth = Some(r.to_vec());
            r.to_vec()
        }
    }
}

/// Convergence monitor with multiple acceleration techniques.
#[derive(Debug, Clone)]
pub struct ConvergenceAccelerator {
    /// Aitken acceleration.
    pub aitken: AitkenAcceleration,
    /// Minimal residual smoothing.
    pub mr_smooth: MinimalResidualSmoothing,
    /// History of residuals for monitoring.
    pub residual_history: Vec<f64>,
    /// Maximum history size.
    pub max_history: usize,
}

impl Default for ConvergenceAccelerator {
    fn default() -> Self {
        Self {
            aitken: AitkenAcceleration::new(),
            mr_smooth: MinimalResidualSmoothing::new(),
            residual_history: Vec::new(),
            max_history: 100,
        }
    }
}

impl ConvergenceAccelerator {
    /// Creates a new convergence accelerator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a residual value.
    pub fn record_residual(&mut self, r: f64) {
        self.residual_history.push(r);
        while self.residual_history.len() > self.max_history {
            self.residual_history.remove(0);
        }
    }

    /// Estimates convergence rate from residual history.
    pub fn convergence_rate(&self) -> Option<f64> {
        if self.residual_history.len() < 10 {
            return None;
        }

        let n = self.residual_history.len();
        let r0 = self.residual_history[n - 10];
        let r10 = self.residual_history[n - 1];

        if r0 > 1e-15 && r10 > 0.0 {
            Some((r0 / r10).powf(1.0 / 10.0))
        } else {
            None
        }
    }

    /// Detects stagnation (no progress in recent iterations).
    pub fn is_stagnating(&self, tolerance: f64) -> bool {
        if self.residual_history.len() < 5 {
            return false;
        }

        let n = self.residual_history.len();
        let recent = &self.residual_history[n - 5..];

        let r_min = recent.iter().copied().fold(f64::INFINITY, f64::min);
        let r_max = recent.iter().copied().fold(0.0, f64::max);

        (r_max - r_min) < tolerance * r_max.max(1e-15)
    }

    /// Resets all acceleration state.
    pub fn reset(&mut self) {
        self.aitken.reset();
        self.mr_smooth = MinimalResidualSmoothing::new();
        self.residual_history.clear();
    }
}

/// Steffensen's method for accelerating fixed-point iteration.
pub fn steffensen<F>(f: F, x0: &[f64], tol: f64, max_iter: usize) -> (Vec<f64>, usize, bool)
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut iteration = 0;
    let mut converged = false;

    while iteration < max_iter {
        // y = f(x)
        let y = f(&x);

        // z = f(y)
        let z = f(&y);

        // Aitken acceleration: x_new = x - (y - x)^2 / (z - 2y + x)
        let mut x_new: Vec<f64> = Vec::with_capacity(n);
        let mut max_change: f64 = 0.0;

        for i in 0..n {
            let num = (y[i] - x[i]).powi(2);
            let denom = z[i] - 2.0 * y[i] + x[i];

            if denom.abs() > 1e-15 {
                x_new.push(x[i] - num / denom);
            } else {
                x_new.push(x[i]);
            }

            max_change = max_change.max((x_new[i] - x[i]).abs());
        }

        x = x_new;
        iteration += 1;

        if max_change < tol {
            converged = true;
            break;
        }
    }

    (x, iteration, converged)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aitken_acceleration() {
        // Test with constant sequence converging to a value
        let x0 = vec![0.5];
        let x1 = vec![0.25];
        let x2 = vec![0.125];

        let mut aitken = AitkenAcceleration::new();
        aitken.update(&x0);
        aitken.update(&x1);
        let accelerated = aitken.update(&x2);

        // Aitken should give an accelerated estimate
        assert!(accelerated.is_some());
        let acc = accelerated.unwrap();
        // For geometric sequence x_{n+1} = r * x_n, Aitken gives limit
        assert!(acc[0].abs() < 0.01);
    }

    #[test]
    fn test_minimal_residual_smoothing() {
        let mut smooth = MinimalResidualSmoothing::new();

        let r = vec![1.0, 2.0, 3.0];
        let p = vec![1.0, 1.0, 1.0];

        let r_smooth = smooth.smooth(&r, &p);

        // Check that smoothing reduced residual norm
        let r_norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        let r_smooth_norm: f64 = r_smooth.iter().map(|v| v * v).sum::<f64>().sqrt();

        assert!(r_smooth_norm <= r_norm);
    }

    #[test]
    fn test_convergence_accelerator() {
        let mut accelerator = ConvergenceAccelerator::new();

        // Simulate convergence
        for i in 0..20 {
            let r = 0.5f64.powi(i);
            accelerator.record_residual(r);
        }

        let rate = accelerator.convergence_rate();
        assert!(rate.is_some());
        let rate = rate.unwrap();
        assert!(rate > 1.5); // Should show convergence

        assert!(!accelerator.is_stagnating(0.1));
    }

    #[test]
    fn test_stagnation_detection() {
        let mut accelerator = ConvergenceAccelerator::new();

        // Simulate stagnation
        for _ in 0..20 {
            accelerator.record_residual(1.0 + (rand_f64() - 0.5) * 0.01);
        }

        assert!(accelerator.is_stagnating(0.1));
    }

    #[test]
    fn test_steffensen() {
        // Fixed point: x = x^2 has solution x = 0 (for |x| < 1)
        let f = |x: &[f64]| vec![x[0] * x[0]];

        let (x, iter, converged) = steffensen(f, &[0.5], 1e-10, 100);

        assert!(converged);
        assert!(iter < 20);
        assert!(x[0].abs() < 0.01);
    }
}

fn rand_f64() -> f64 {
    // Simple deterministic "random" for testing
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos() as f64 / 1e9
}

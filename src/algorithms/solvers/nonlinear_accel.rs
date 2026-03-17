//! Nonlinear Richardson extrapolation and convergence acceleration.
#![allow(private_interfaces)]
#![allow(unused_imports)]
#![allow(non_snake_case)]
#![allow(unused_assignments)]
//!
//! Advanced extrapolation methods for accelerating iterative processes.

use nalgebra::DVector;

/// Epsilon algorithm (Wynn's epsilon algorithm).
///
/// Nonlinear sequence acceleration using the epsilon algorithm.
pub struct EpsilonAlgorithm {
    /// Maximum sequence length.
    max_seq: usize,
    /// Sequence values.
    sequence: Vec<DVector<f64>>,
}

impl EpsilonAlgorithm {
    /// Creates a new Epsilon algorithm accelerator.
    pub fn new(max_seq: usize) -> Self {
        Self {
            max_seq,
            sequence: Vec::new(),
        }
    }

    /// Updates with new iterate.
    pub fn update(&mut self, x: &DVector<f64>) -> Option<DVector<f64>> {
        self.sequence.push(x.clone());

        if self.sequence.len() > self.max_seq {
            self.sequence.remove(0);
        }

        // Need at least 4 points for epsilon algorithm
        if self.sequence.len() < 4 {
            return None;
        }

        // Build epsilon table
        let n = self.sequence.len();
        let mut eps = vec![vec![DVector::zeros(x.len()); n + 1]; n + 1];

        // Initialize with sequence values
        for i in 0..n {
            eps[0][i] = self.sequence[i].clone();
        }

        // Compute epsilon table
        for i in 0..n - 1 {
            for j in 0..n - i - 1 {
                let diff = &eps[i][j + 1] - &eps[i][j];
                let diff_norm_sq = diff.norm_squared();
                if diff_norm_sq > 1e-15 {
                    let inv_diff = diff.scale(1.0 / diff_norm_sq);
                    eps[i + 1][j] = &eps[i][j + 2] + inv_diff;
                } else {
                    eps[i + 1][j] = eps[i][j + 2].clone();
                }
            }
        }

        // Return accelerated value (even column)
        Some(eps[n][0].clone())
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.sequence.clear();
    }
}

/// Levin u-transform for sequence acceleration.
pub struct LevinU {
    max_seq: usize,
    sequence: Vec<DVector<f64>>,
    remainders: Vec<DVector<f64>>,
}

impl LevinU {
    /// Creates a new Levin u-transform accelerator.
    pub fn new(max_seq: usize) -> Self {
        Self {
            max_seq,
            sequence: Vec::new(),
            remainders: Vec::new(),
        }
    }

    /// Updates with new iterate and remainder estimate.
    pub fn update(&mut self, x: &DVector<f64>, r: &DVector<f64>) -> Option<DVector<f64>> {
        self.sequence.push(x.clone());
        self.remainders.push(r.clone());

        if self.sequence.len() > self.max_seq {
            self.sequence.remove(0);
            self.remainders.remove(0);
        }

        if self.sequence.len() < 3 {
            return None;
        }

        let n = self.sequence.len();

        // Compute Levin u-transform
        let mut num = DVector::zeros(x.len());
        let mut den: f64 = 0.0;

        for k in 0..n {
            let coef = (-1.0f64).powi(n as i32 - 1 - k as i32)
                * (k as f64).powf(n as f64 - 1.0)
                / (n as f64 - 1.0 - k as f64).factorial() as f64;

            let denom = (n as f64 - 1.0 - k as f64 + 1.0).recip();
            let term = &self.sequence[k] - &self.remainders[k].scale(denom);

            num = &num + term.scale(coef);
            den += coef;
        }

        if den.abs() > 1e-15 {
            Some(num / den)
        } else {
            None
        }
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.sequence.clear();
        self.remainders.clear();
    }
}

/// Aitken's delta-squared process for scalar sequences.
pub struct AitkenDeltaSquared {
    x_prev: Option<f64>,
    x_prev2: Option<f64>,
}

impl AitkenDeltaSquared {
    /// Creates a new Aitken accelerator.
    pub fn new() -> Self {
        Self {
            x_prev: None,
            x_prev2: None,
        }
    }

    /// Updates with new scalar value.
    pub fn update(&mut self, x: f64) -> Option<f64> {
        self.x_prev2 = self.x_prev;
        self.x_prev = Some(x);

        if let (Some(x_prev), Some(x_prev2)) = (self.x_prev, self.x_prev2) {
            let dx1 = x - x_prev;
            let dx2 = x_prev - x_prev2;

            if dx2.abs() > 1e-15 {
                let accelerated = x - dx1 * dx1 / dx2;
                Some(accelerated)
            } else {
                Some(x)
            }
        } else {
            None
        }
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.x_prev = None;
        self.x_prev2 = None;
    }
}

/// Brezinski's theta algorithm.
pub struct ThetaAlgorithm {
    max_seq: usize,
    sequence: Vec<DVector<f64>>,
}

impl ThetaAlgorithm {
    /// Creates a new Theta algorithm accelerator.
    pub fn new(max_seq: usize) -> Self {
        Self {
            max_seq,
            sequence: Vec::new(),
        }
    }

    /// Updates with new iterate.
    pub fn update(&mut self, x: &DVector<f64>) -> Option<DVector<f64>> {
        self.sequence.push(x.clone());

        if self.sequence.len() > self.max_seq {
            self.sequence.remove(0);
        }

        if self.sequence.len() < 3 {
            return None;
        }

        // Simple theta2 transform
        let n = self.sequence.len() - 1;
        let dx1 = &self.sequence[n] - &self.sequence[n - 1];
        let dx2 = &self.sequence[n - 1] - &self.sequence[n - 2];

        let dx1_norm_sq = dx1.norm_squared();
        let dx2_norm_sq = dx2.norm_squared();

        if dx1_norm_sq > 1e-15 && dx2_norm_sq > 1e-15 {
            let ratio = dx1_norm_sq.sqrt() / dx2_norm_sq.sqrt();
            Some(&self.sequence[n - 1] + dx1.scale(ratio / (1.0 - ratio)))
        } else {
            Some(self.sequence[n].clone())
        }
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.sequence.clear();
    }
}

/// Combined acceleration using multiple methods.
pub struct CombinedAccelerator {
    epsilon: EpsilonAlgorithm,
    levin: LevinU,
    theta: ThetaAlgorithm,
    best_norm: Option<f64>,
}

impl CombinedAccelerator {
    /// Creates a new combined accelerator.
    pub fn new(max_seq: usize) -> Self {
        Self {
            epsilon: EpsilonAlgorithm::new(max_seq),
            levin: LevinU::new(max_seq),
            theta: ThetaAlgorithm::new(max_seq),
            best_norm: None,
        }
    }

    /// Updates all accelerators and returns the best result.
    pub fn update(&mut self, x: &DVector<f64>) -> Option<DVector<f64>> {
        let mut best: Option<DVector<f64>> = None;
        let mut best_norm = f64::INFINITY;

        // Try epsilon algorithm
        if let Some(x_eps) = self.epsilon.update(x) {
            let norm = x_eps.norm();
            if norm < best_norm {
                best_norm = norm;
                best = Some(x_eps);
            }
        }

        // Try theta algorithm
        if let Some(x_theta) = self.theta.update(x) {
            let norm = x_theta.norm();
            if norm < best_norm {
                best_norm = norm;
                best = Some(x_theta);
            }
        }

        // Try Levin u-transform with zero remainder estimate
        if let Some(x_levin) = self.levin.update(x, &DVector::zeros(x.len())) {
            let norm = x_levin.norm();
            if norm < best_norm {
                best_norm = norm;
                best = Some(x_levin);
            }
        }

        self.best_norm = Some(best_norm);
        best
    }

    /// Returns the best norm found.
    pub fn best_norm(&self) -> Option<f64> {
        self.best_norm
    }

    /// Resets all accelerators.
    pub fn reset(&mut self) {
        self.epsilon.reset();
        self.levin.reset();
        self.theta.reset();
        self.best_norm = None;
    }
}

// Helper trait for factorial
trait Factorial {
    fn factorial(&self) -> u64;
}

impl Factorial for u64 {
    fn factorial(&self) -> u64 {
        if *self <= 1 {
            1
        } else {
            (2..=*self).product()
        }
    }
}

impl Factorial for f64 {
    fn factorial(&self) -> u64 {
        (*self as u64).factorial()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aitken_scalar() {
        let mut aitken = AitkenDeltaSquared::new();

        // Geometric sequence converging to 0
        let mut x = 1.0;
        for _ in 0..5 {
            x *= 0.5;
            aitken.update(x);
        }

        if let Some(x_accel) = aitken.update(x * 0.5) {
            assert!(x_accel.abs() < x.abs());
        }
    }

    #[test]
    fn test_epsilon_algorithm() {
        let mut eps = EpsilonAlgorithm::new(10);
        let mut x = DVector::from_element(5, 1.0);

        for i in 0..6 {
            x = x.scale(0.5);
            let _ = eps.update(&x);
        }

        if let Some(x_accel) = eps.update(&x.scale(0.5)) {
            assert!(x_accel.norm() < x.norm());
        }
    }

    #[test]
    fn test_theta_algorithm() {
        let mut theta = ThetaAlgorithm::new(10);
        let mut x = DVector::from_element(5, 1.0);

        for _ in 0..5 {
            x = x.scale(0.5);
            let _ = theta.update(&x);
        }

        if let Some(x_accel) = theta.update(&x.scale(0.5)) {
            // Accelerated should not be worse
            assert!(x_accel.norm() < x.norm() * 2.0);
        }
    }

    #[test]
    fn test_combined_accelerator() {
        let mut combined = CombinedAccelerator::new(10);
        let mut x = DVector::from_element(5, 1.0);

        for _ in 0..5 {
            x = x.scale(0.5);
            let _ = combined.update(&x);
        }

        if let Some(x_accel) = combined.update(&x.scale(0.5)) {
            assert!(x_accel.norm() < x.norm());
        }

        assert!(combined.best_norm().is_some());
    }
}

//! Polynomial and rational approximation acceleration methods.
//!
//! This module provides:
//! - Polynomial acceleration for iterative methods
//! - Rational approximation techniques
//! - Pade approximation acceleration
//! - Chebyshev polynomial methods

use nalgebra::{DMatrix, DVector};

/// Polynomial acceleration configuration.
#[derive(Debug, Clone)]
pub struct PolynomialAccelerationConfig {
    /// Polynomial degree.
    pub degree: usize,
    /// Estimate of smallest eigenvalue.
    pub lambda_min: f64,
    /// Estimate of largest eigenvalue.
    pub lambda_max: f64,
    /// Use minimax polynomial.
    pub minimax: bool,
}

impl Default for PolynomialAccelerationConfig {
    fn default() -> Self {
        Self {
            degree: 3,
            lambda_min: 1e-6,
            lambda_max: 10.0,
            minimax: true,
        }
    }
}

/// Chebyshev polynomial acceleration for iterative methods.
#[derive(Debug, Clone)]
pub struct ChebyshevAcceleration {
    /// Polynomial coefficients.
    pub coefficients: Vec<f64>,
    /// Recurrence storage.
    pub p_storage: Vec<DVector<f64>>,
    /// Current iteration.
    pub iteration: usize,
}

impl ChebyshevAcceleration {
    /// Creates Chebyshev acceleration with given parameters.
    pub fn new(lambda_min: f64, lambda_max: f64, degree: usize) -> Self {
        let mut coefficients = Vec::with_capacity(degree);

        // Compute Chebyshev coefficients for 1/x approximation
        let c = (lambda_max + lambda_min) / (lambda_max - lambda_min);
        let mut t_prev = 1.0;
        let mut t_curr = c;

        coefficients.push(1.0 / lambda_max);
        for i in 1..degree {
            let t_next = 2.0 * c * t_curr - t_prev;
            coefficients.push(2.0 * t_next / lambda_max);
            t_prev = t_curr;
            t_curr = t_next;
        }

        Self {
            coefficients,
            p_storage: Vec::new(),
            iteration: 0,
        }
    }

    /// Applies polynomial acceleration to a sequence.
    pub fn apply(&mut self, residuals: &[DVector<f64>]) -> Option<DVector<f64>> {
        if residuals.len() < self.coefficients.len() {
            return None;
        }

        let n = residuals[0].len();
        let mut accelerated = DVector::zeros(n);

        for (i, coef) in self.coefficients.iter().enumerate() {
            accelerated += residuals[residuals.len() - 1 - i].scale(*coef);
        }

        Some(accelerated)
    }

    /// Performs Chebyshev iteration for solving Ax=b.
    pub fn chebyshev_iteration(
        a: &DMatrix<f64>,
        b: &DVector<f64>,
        lambda_min: f64,
        lambda_max: f64,
        tol: f64,
        max_iter: usize,
    ) -> (DVector<f64>, usize, f64) {
        let n = b.len();
        let mut x = DVector::zeros(n);
        let mut r = b - a * &x;

        let alpha_opt = 4.0 / (lambda_max + lambda_min).powi(2);
        let omega = (lambda_max - lambda_min) / (lambda_max + lambda_min);
        let mut beta = 0.0;
        let mut p = r.clone();

        let b_norm = b.norm();
        let tol_abs = tol * b_norm.max(1e-15);

        let mut iteration = 0;
        let mut converged = false;

        while iteration < max_iter {
            let ap = a * &p;
            let p_ap = p.dot(&ap);

            if p_ap.abs() < 1e-30 {
                break;
            }

            let alpha = alpha_opt / (1.0 - beta);
            x += p.scale(alpha);
            r -= ap.scale(alpha);

            if r.norm() < tol_abs {
                converged = true;
                iteration += 1;
                break;
            }

            beta = omega * omega / (2.0 - omega * omega * (1.0 - beta));
            p = r.clone() + p.scale(beta);

            iteration += 1;
        }

        let final_residual = (b - a * &x).norm();
        (x, iteration, final_residual)
    }
}

/// Pade approximation acceleration.
#[derive(Debug, Clone)]
pub struct PadeAcceleration {
    /// Numerator degree.
    pub m: usize,
    /// Denominator degree.
    pub n: usize,
    /// Numerator coefficients.
    pub num_coeffs: Vec<f64>,
    /// Denominator coefficients.
    pub den_coeffs: Vec<f64>,
}

impl PadeAcceleration {
    /// Creates Pade approximation of order (m, n).
    pub fn new(m: usize, n: usize) -> Self {
        // Compute Pade coefficients for exp approximation
        // These are for accelerating convergence
        let mut num_coeffs = Vec::with_capacity(m + 1);
        let mut den_coeffs = Vec::with_capacity(n + 1);

        // Simple binomial coefficients for now
        for k in 0..=m {
            num_coeffs.push(Self::binomial(m, k));
        }

        for k in 0..=n {
            den_coeffs.push(Self::binomial(n, k));
        }

        Self {
            m,
            n,
            num_coeffs,
            den_coeffs,
        }
    }

    fn binomial(n: usize, k: usize) -> f64 {
        if k > n {
            return 0.0;
        }
        let mut result = 1.0;
        for i in 0..k {
            result = result * (n - i) as f64 / (i + 1) as f64;
        }
        result
    }

    /// Applies Pade acceleration to a sequence.
    pub fn apply(&self, sequence: &[DVector<f64>]) -> Option<DVector<f64>> {
        let needed = self.m.max(self.n) + 1;
        if sequence.len() < needed {
            return None;
        }

        let n = sequence[0].len();

        // Build numerator: sum of num_coeffs[k] * s[k]
        let mut num = DVector::zeros(n);
        for k in 0..=self.m {
            let idx = sequence.len() - 1 - k;
            num += sequence[idx].scale(self.num_coeffs[k]);
        }

        // Build denominator: sum of den_coeffs[k] * s[k]
        let mut den = DVector::zeros(n);
        for k in 0..=self.n {
            let idx = sequence.len() - 1 - k;
            den += sequence[idx].scale(self.den_coeffs[k]);
        }

        // Element-wise division (simplified - assumes positive denominator)
        let mut result = DVector::zeros(n);
        for i in 0..n {
            if den[i].abs() > 1e-15 {
                result[i] = num[i] / den[i];
            } else {
                result[i] = num[i];
            }
        }

        Some(result)
    }
}

/// Minimal polynomial acceleration.
pub struct MinimalPolynomialAcceleration {
    /// Minimal polynomial coefficients.
    pub coeffs: Vec<f64>,
    /// Storage for iterates.
    pub iterates: Vec<DVector<f64>>,
    /// Storage for residuals.
    pub residuals: Vec<DVector<f64>>,
}

impl MinimalPolynomialAcceleration {
    /// Creates new minimal polynomial accelerator.
    pub fn new(max_degree: usize) -> Self {
        Self {
            coeffs: vec![1.0; max_degree + 1],
            iterates: Vec::with_capacity(max_degree + 1),
            residuals: Vec::with_capacity(max_degree + 1),
        }
    }

    /// Estimates minimal polynomial from iterates.
    pub fn estimate_polynomial(
        residuals: &[DVector<f64>],
        max_degree: usize,
    ) -> Vec<f64> {
        let m = max_degree.min(residuals.len() - 1);
        if m == 0 {
            return vec![1.0];
        }

        // Build matrix from residuals
        let n = residuals[0].len();
        let mut gram = DMatrix::zeros(m + 1, m + 1);

        for i in 0..=m {
            for j in 0..=m {
                let idx_i = residuals.len() - 1 - i;
                let idx_j = residuals.len() - 1 - j;
                if idx_i < residuals.len() && idx_j < residuals.len() {
                    gram[(i, j)] = residuals[idx_i].dot(&residuals[idx_j]);
                }
            }
        }

        // Solve for coefficients (simplified)
        let mut coeffs = vec![1.0; m + 1];
        for i in 1..=m {
            coeffs[i] = -gram[(0, i)] / gram[(0, 0)].max(1e-15);
        }

        coeffs
    }

    /// Applies minimal polynomial acceleration.
    pub fn accelerate(&mut self, x: &DVector<f64>, r: &DVector<f64>) -> Option<DVector<f64>> {
        self.iterates.push(x.clone());
        self.residuals.push(r.clone());

        if self.iterates.len() < 3 {
            return None;
        }

        // Estimate minimal polynomial
        self.coeffs = Self::estimate_polynomial(&self.residuals, self.coeffs.len() - 1);

        // Compute accelerated iterate
        let mut x_accel = DVector::zeros(x.len());
        for (k, c) in self.coeffs.iter().enumerate() {
            if k < self.iterates.len() {
                x_accel += self.iterates[k].scale(*c);
            }
        }

        let sum_coeffs: f64 = self.coeffs.iter().sum();
        if sum_coeffs.abs() > 1e-15 {
            x_accel /= sum_coeffs;
        }

        Some(x_accel)
    }
}

/// Steffensen acceleration for fixed-point iteration.
pub struct SteffensenAcceleration;

impl SteffensenAcceleration {
    /// Performs one Steffensen acceleration step.
    pub fn step<F>(f: F, x: &DVector<f64>) -> DVector<f64>
    where
        F: Fn(&DVector<f64>) -> DVector<f64>,
    {
        let y = f(x);
        let z = f(&y);

        // Steffensen formula: x_new = x - (y - x)^2 / (z - 2y + x)
        let dx = &y - x;
        let d2x = &z - &y.scale(2.0) + x;

        let mut x_new = DVector::zeros(x.len());
        for i in 0..x.len() {
            if d2x[i].abs() > 1e-15 {
                x_new[i] = x[i] - dx[i] * dx[i] / d2x[i];
            } else {
                x_new[i] = y[i];
            }
        }

        x_new
    }

    /// Accelerates fixed-point iteration to convergence.
    pub fn solve<F>(f: F, x0: &DVector<f64>, tol: f64, max_iter: usize) -> (DVector<f64>, usize, bool)
    where
        F: Fn(&DVector<f64>) -> DVector<f64>,
    {
        let mut x = x0.clone();
        let mut iteration = 0;
        let mut converged = false;

        while iteration < max_iter {
            let x_new = Self::step(&f, &x);

            let diff = (&x_new - &x).norm();
            if diff < tol {
                converged = true;
                x = x_new;
                iteration += 1;
                break;
            }

            x = x_new;
            iteration += 1;
        }

        (x, iteration, converged)
    }
}

/// Anderson acceleration for fixed-point iteration.
#[derive(Debug, Clone)]
pub struct AndersonAcceleration {
    /// History depth.
    pub depth: usize,
    /// Mixing parameter.
    pub beta: f64,
    /// Type (1 or 2).
    pub anderson_type: usize,
}

impl AndersonAcceleration {
    /// Creates Anderson acceleration.
    pub fn new(depth: usize, beta: f64) -> Self {
        Self {
            depth,
            beta,
            anderson_type: 2,
        }
    }

    /// Performs Anderson acceleration step.
    pub fn step<F>(&self, f: F, iterates: &[DVector<f64>]) -> Option<DVector<f64>>
    where
        F: Fn(&DVector<f64>) -> DVector<f64>,
    {
        let m = self.depth.min(iterates.len());
        if m == 0 {
            return None;
        }

        // Compute residuals
        let mut residuals: Vec<DVector<f64>> = Vec::with_capacity(m);
        for i in 0..m {
            let x = &iterates[iterates.len() - m + i];
            residuals.push(f(x) - x);
        }

        // Simple Anderson mixing (averaging with residual-based weights)
        let mut x_new = iterates[iterates.len() - 1].clone();
        let mut total_weight = 0.0;

        for i in 0..m {
            let weight = 1.0 / (residuals[i].norm().max(1e-15));
            x_new += residuals[i].scale(weight * self.beta);
            total_weight += weight;
        }

        if total_weight > 1e-15 {
            x_new /= total_weight / m as f64;
        }

        Some(x_new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chebyshev_acceleration() {
        let a = DMatrix::from_row_slice(10, 10, &[
            4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0,
        ]);

        let b = DVector::from_element(10, 1.0);

        let (x, iter, residual) = ChebyshevAcceleration::chebyshev_iteration(
            &a, &b, 1.0, 5.0, 1e-10, 100,
        );

        assert!(iter > 0);
        assert!(residual < 0.1);
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_pade_acceleration() {
        let pade = PadeAcceleration::new(2, 2);

        let sequence = vec![
            DVector::from_column_slice(&[1.0, 2.0, 3.0]),
            DVector::from_column_slice(&[1.1, 2.1, 3.1]),
            DVector::from_column_slice(&[1.05, 2.05, 3.05]),
        ];

        let result = pade.apply(&sequence);
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn test_minimal_polynomial() {
        let mut mpa = MinimalPolynomialAcceleration::new(5);

        let residuals = vec![
            DVector::from_column_slice(&[1.0, 0.5, 0.25]),
            DVector::from_column_slice(&[0.5, 0.25, 0.125]),
            DVector::from_column_slice(&[0.25, 0.125, 0.0625]),
        ];

        let coeffs = MinimalPolynomialAcceleration::estimate_polynomial(&residuals, 2);
        assert_eq!(coeffs.len(), 3);
    }

    #[test]
    fn test_steffensen() {
        // Fixed point: x = 0.5 * x
        let f = |x: &DVector<f64>| x.scale(0.5);
        let x0 = DVector::from_column_slice(&[1.0, 2.0, 3.0]);

        let (x, iter, converged) = SteffensenAcceleration::solve(f, &x0, 1e-10, 50);

        assert!(converged);
        assert!(iter < 20);
        assert!(x.norm() < 0.1);
    }

    #[test]
    fn test_anderson_acceleration() {
        let anderson = AndersonAcceleration::new(3, 0.5);

        let iterates = vec![
            DVector::from_column_slice(&[1.0, 2.0, 3.0]),
            DVector::from_column_slice(&[0.6, 1.2, 1.8]),
            DVector::from_column_slice(&[0.36, 0.72, 1.08]),
        ];

        let f = |x: &DVector<f64>| x.scale(0.6);
        let result = anderson.step(f, &iterates);

        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.len(), 3);
    }
}

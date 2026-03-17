//! Polynomial and spectral acceleration methods.
#![allow(non_snake_case)]
#![allow(unused_assignments)]
//!
//! This module provides advanced polynomial-based acceleration:
//! - Chebyshev semi-iterative methods
//! - Lanczos-based spectral acceleration
//! - Rational approximation methods
//! - Krylov subspace recycling with spectral deflation

use nalgebra::{DMatrix, DVector, SymmetricEigen};

/// Chebyshev semi-iterative method with automatic eigenvalue estimation.
///
/// Uses Chebyshev polynomials to minimize the error over an estimated
/// spectral interval [lambda_min, lambda_max].
#[derive(Debug, Clone)]
pub struct ChebyshevSemiIterative {
    /// Estimated minimum eigenvalue.
    pub lambda_min: f64,
    /// Estimated maximum eigenvalue.
    pub lambda_max: f64,
    /// Current iteration count.
    pub iteration: usize,
    /// Recurrence coefficients.
    pub c_k: f64,
    pub c_k_minus_1: f64,
}

impl ChebyshevSemiIterative {
    /// Creates a new Chebyshev accelerator.
    pub fn new(lambda_min: f64, lambda_max: f64) -> Self {
        let sigma = (lambda_max - lambda_min) / (lambda_max + lambda_min);
        Self {
            lambda_min,
            lambda_max,
            iteration: 0,
            c_k: 1.0 / sigma,
            c_k_minus_1: 1.0,
        }
    }

    /// Estimates eigenvalues using Gershgorin circles.
    pub fn estimate_eigenvalues_gershgorin(a: &DMatrix<f64>) -> (f64, f64) {
        let n = a.nrows();
        let mut lambda_min = f64::INFINITY;
        let mut lambda_max = f64::NEG_INFINITY;

        for i in 0..n {
            let a_ii = a[(i, i)];
            let mut sum_off_diag = 0.0;
            for j in 0..n {
                if i != j {
                    sum_off_diag += a[(i, j)].abs();
                }
            }
            let lower = a_ii - sum_off_diag;
            let upper = a_ii + sum_off_diag;
            lambda_min = lambda_min.min(lower);
            lambda_max = lambda_max.max(upper);
        }

        // Ensure positivity for SPD matrices
        if lambda_min < 1e-15 {
            lambda_min = lambda_max.max(1e-15) * 0.01;
        }

        (lambda_min, lambda_max)
    }

    /// Estimates eigenvalues using a few Lanczos iterations.
    pub fn estimate_eigenvalues_lanczos(a: &DMatrix<f64>, num_iter: usize) -> (f64, f64) {
        let n = a.nrows();
        let m = num_iter.min(n - 1);

        if m < 2 {
            return Self::estimate_eigenvalues_gershgorin(a);
        }

        // Lanczos tridiagonalization
        let mut alpha = Vec::with_capacity(m);
        let mut beta = Vec::with_capacity(m);

        let mut v = DVector::from_element(n, 1.0_f64 / (n as f64).sqrt());
        let mut w = a * &v;

        let mut alpha_0 = v.dot(&w);
        alpha.push(alpha_0);
        w -= v.scale(alpha_0);
        let mut beta_0 = w.norm();
        beta.push(beta_0);

        let mut v_prev = v.clone();
        v = w.scale(1.0 / beta_0);

        for _ in 1..m {
            w = a * &v;
            let alpha_k = v.dot(&w);
            alpha.push(alpha_k);

            w -= v.scale(alpha_k);
            w += v_prev.scale(-beta_0);
            v_prev = v.clone();
            let scale = 1.0 / beta_0.min(1e15).max(1e-15);
            v = w.scale(scale);

            beta_0 = w.norm();
            beta.push(beta_0);
        }

        // Compute eigenvalues of tridiagonal matrix
        let mut tridiag = DMatrix::zeros(m, m);
        for i in 0..m {
            tridiag[(i, i)] = alpha[i];
            if i < m - 1 {
                tridiag[(i, i + 1)] = beta[i + 1];
                tridiag[(i + 1, i)] = beta[i + 1];
            }
        }

        let eigen = SymmetricEigen::new(tridiag);
        let lambda_min = eigen.eigenvalues.min();
        let lambda_max = eigen.eigenvalues.max();

        (lambda_min.max(1e-15), lambda_max)
    }

    /// Performs one Chebyshev iteration.
    ///
    /// Updates: x_{k+1} = x_k + omega_k * (b - A * x_k)
    ///          with optimal Chebyshev parameter omega_k
    pub fn iterate(&mut self, x: &mut DVector<f64>, a: &DMatrix<f64>, b: &DVector<f64>) {
        let rho = (self.lambda_max - self.lambda_min) / (self.lambda_max + self.lambda_min);

        // Compute optimal parameter for this iteration
        let base_theta = 2.0_f64 / (self.lambda_max + self.lambda_min);
        let theta = if self.iteration == 0 {
            base_theta
        } else {
            let c = 2.0_f64 * rho / (1.0_f64 + rho * rho);
            let omega = 1.0_f64 / (1.0_f64 - c * c / 4.0_f64);
            omega * base_theta
        };

        // Compute residual
        let r = b - a * &*x;

        // Update solution
        if self.iteration == 0 {
            *x += r.scale(theta);
        } else {
            let r_scaled = r.scale(theta);
            *x += r_scaled;
        }

        self.iteration += 1;
    }

    /// Performs Chebyshev iteration with precomputed residual.
    pub fn iterate_with_residual(
        &mut self,
        x: &mut DVector<f64>,
        r: &mut DVector<f64>,
        a: &DMatrix<f64>,
        p: &DVector<f64>,
    ) {
        let rho = (self.lambda_max - self.lambda_min) / (self.lambda_max + self.lambda_min);

        // Chebyshev recurrence
        let sigma = 1.0 / rho;
        let c_new = 1.0 / (sigma - self.c_k_minus_1 / 4.0 / self.c_k);

        // Relaxation parameter
        let omega = c_new * 4.0 / (self.lambda_max + self.lambda_min);

        // Update solution
        *x += r.scale(omega);

        // Update residual
        let ap = a * p;
        *r -= ap.scale(omega);

        // Update recurrence coefficients
        self.c_k_minus_1 = self.c_k;
        self.c_k = c_new;
        self.iteration += 1;
    }

    /// Resets the iteration counter.
    pub fn reset(&mut self) {
        self.iteration = 0;
        self.c_k = 1.0;
        self.c_k_minus_1 = 1.0;
    }
}

/// Spectral deflation preconditioner.
///
/// Removes the effect of smallest eigenvalues to improve convergence
/// of iterative solvers.
#[derive(Debug, Clone)]
pub struct SpectralDeflation {
    /// Deflation subspace (eigenvectors).
    pub eigenvectors: DMatrix<f64>,
    /// Corresponding eigenvalues.
    pub eigenvalues: DVector<f64>,
    /// Projection operator.
    pub projection: DMatrix<f64>,
}

impl SpectralDeflation {
    /// Creates a spectral deflation preconditioner.
    pub fn new(eigenvectors: DMatrix<f64>, eigenvalues: DVector<f64>) -> Self {
        let (n, k) = eigenvectors.shape();

        // Compute projection: P = I - Z * (Z^T * A * Z)^{-1} * Z^T * A
        // Simplified: P = I - Z * Z^T (for orthonormal Z)

        let zt = eigenvectors.transpose();
        let projection = DMatrix::identity(n, n) - &eigenvectors * &zt;

        Self {
            eigenvectors,
            eigenvalues,
            projection,
        }
    }

    /// Computes approximate eigenvectors using Lanczos with restarts.
    pub fn compute_deflation_subspace(a: &DMatrix<f64>, num_eigs: usize, max_iter: usize) -> Self {
        let n = a.nrows();
        let k = num_eigs.min(n - 1);

        if k == 0 {
            return Self {
                eigenvectors: DMatrix::zeros(n, 0),
                eigenvalues: DVector::zeros(0),
                projection: DMatrix::identity(n, n),
            };
        }

        // Lanczos with full reorthogonalization
        let m = (2 * k).min(n - 1);
        let mut v = Vec::with_capacity(m + 1);
        let mut alpha = Vec::with_capacity(m);
        let mut beta = Vec::with_capacity(m);

        // Random starting vector
        let mut v0 = DVector::from_fn(n, |i, _| ((i + 1) as f64).sin());
        v0.normalize_mut();
        v.push(v0);

        let mut beta_prev = 0.0_f64;
        for j in 0..m {
            let mut w = a * &v[j];

            // Orthogonalize against all previous vectors
            for i in 0..=j {
                let h = v[i].dot(&w);
                if i == j {
                    alpha.push(h);
                }
                w -= v[i].scale(h);
            }

            // Reorthogonalization pass
            for i in 0..=j {
                let dot = v[i].dot(&w);
                w -= v[i].scale(dot);
            }

            let beta_new = w.norm();
            beta.push(beta_prev);
            beta_prev = beta_new;

            if j < m - 1 && beta_new > 1e-15 {
                v.push(w / beta_new);
            } else {
                break;
            }
        }

        // Build tridiagonal and compute eigenpairs
        let actual_m = alpha.len();
        let mut tridiag = DMatrix::zeros(actual_m, actual_m);
        for i in 0..actual_m {
            tridiag[(i, i)] = alpha[i];
            if i < actual_m - 1 && beta[i + 1].abs() > 1e-15 {
                tridiag[(i, i + 1)] = beta[i + 1];
                tridiag[(i + 1, i)] = beta[i + 1];
            }
        }

        let eigen = SymmetricEigen::new(tridiag);

        // Get k smallest eigenvalues and corresponding Ritz vectors
        let mut indices: Vec<usize> = (0..actual_m).collect();
        indices.sort_by(|&a, &b| eigen.eigenvalues[a].partial_cmp(&eigen.eigenvalues[b]).unwrap());

        let mut z = DMatrix::zeros(n, k);
        let mut lambdas = DVector::zeros(k);

        for i in 0..k {
            let idx = indices[i];
            lambdas[i] = eigen.eigenvalues[idx];

            // Ritz vector: y_i = V * s_i where s_i is eigenvector of tridiag
            let mut y = DVector::zeros(n);
            for j in 0..actual_m {
                y += v[j].scale(eigen.eigenvectors[(j, idx)]);
            }
            y.normalize_mut();

            for j in 0..n {
                z[(j, i)] = y[j];
            }
        }

        Self::new(z, lambdas)
    }

    /// Applies the deflation preconditioner.
    pub fn apply(&self, r: &DVector<f64>) -> DVector<f64> {
        &self.projection * r
    }

    /// Applies deflation-corrected solve.
    pub fn apply_corrected(&self, r: &DVector<f64>, z_coarse: &DVector<f64>) -> DVector<f64> {
        let coarse_corr = &self.eigenvectors * z_coarse;
        let fine_part = self.apply(r);
        fine_part + coarse_corr
    }

    /// Returns the condition number of the deflated system.
    pub fn deflated_condition_number(&self) -> f64 {
        if self.eigenvalues.len() == 0 {
            return f64::INFINITY;
        }

        let lambda_min_deflated = self.eigenvalues[self.eigenvalues.len() - 1];
        let lambda_max = self.eigenvalues.max();

        if lambda_min_deflated > 1e-15 {
            lambda_max / lambda_min_deflated
        } else {
            f64::INFINITY
        }
    }
}

/// Rational Chebyshev filter for eigenvalue computation.
#[derive(Debug, Clone)]
pub struct RationalChebyshevFilter {
    /// Filter center.
    pub center: f64,
    /// Filter width.
    pub width: f64,
    /// Filter order.
    pub order: usize,
    /// Chebyshev nodes.
    pub nodes: Vec<f64>,
    /// Chebyshev weights.
    pub weights: Vec<f64>,
}

impl RationalChebyshevFilter {
    /// Creates a new rational Chebyshev filter.
    pub fn new(center: f64, width: f64, order: usize) -> Self {
        let mut nodes = Vec::with_capacity(order);
        let mut weights = Vec::with_capacity(order);

        // Chebyshev nodes on [-1, 1]
        for k in 0..order {
            let theta = (2 * k + 1) as f64 * std::f64::consts::PI / (2 * order) as f64;
            let xi = theta.cos();
            nodes.push(center + width * xi);

            // Chebyshev weights
            let w = std::f64::consts::PI / order as f64 * theta.sin();
            weights.push(w);
        }

        Self {
            center,
            width,
            order,
            nodes,
            weights,
        }
    }

    /// Applies the filter to approximate the spectral projector.
    pub fn apply(&self, a: &DMatrix<f64>, v: &DVector<f64>) -> DVector<f64> {
        let n = a.nrows();
        let mut result = DVector::zeros(n);

        for (i, (&node, &weight)) in self.nodes.iter().zip(self.weights.iter()).enumerate() {
            // Solve (A - node * I)^{-1} * v using shifted LU
            let shifted = a - DMatrix::from_diagonal(&DVector::from_element(n, node));

            let lu = shifted.lu();
            if let Some(sol) = lu.solve(v) {
                result += sol.scale(weight);
            }
        }

        // Normalize
        let norm = result.norm();
        if norm > 1e-15 {
            result /= norm;
        }

        result
    }

    /// Applies filter without factorization (using CG for each shift).
    pub fn apply_iterative(
        &self,
        a: &DMatrix<f64>,
        v: &DVector<f64>,
        tol: f64,
        max_iter: usize,
    ) -> DVector<f64> {
        let n = a.nrows();
        let mut result = DVector::zeros(n);

        for (&node, &weight) in self.nodes.iter().zip(self.weights.iter()) {
            // Solve (A - node * I) * x = v using CG
            let shifted = a - DMatrix::from_diagonal(&DVector::from_element(n, node));

            // CG solve
            let mut x = DVector::zeros(n);
            let mut r = v.clone();
            let mut p = r.clone();
            let rs_old = r.dot(&r);

            for _ in 0..max_iter {
                let ap = &shifted * &p;
                let p_ap = p.dot(&ap);

                if p_ap.abs() < 1e-15 {
                    break;
                }

                let alpha = rs_old / p_ap;
                x += &p * alpha;
                r -= &ap * alpha;

                let rs_new = r.dot(&r);
                if rs_new.sqrt() < tol {
                    break;
                }

                p = &r + p.scale(rs_new / rs_old);
            }

            result += p.scale(weight);
        }

        result
    }
}

/// Polynomial preconditioner based on matrix powers.
#[derive(Debug, Clone)]
pub struct MatrixPowerPreconditioner {
    /// Polynomial coefficients.
    pub coefficients: Vec<f64>,
    /// Precomputed matrix powers.
    pub a_powers: Vec<DMatrix<f64>>,
}

impl MatrixPowerPreconditioner {
    /// Creates a polynomial preconditioner of given degree.
    pub fn new(a: &DMatrix<f64>, degree: usize) -> Self {
        let mut powers = vec![a.clone()];
        let mut current = a.clone();

        for _ in 1..degree {
            let next = &current * a;
            current = next;
            powers.push(current.clone());
        }

        // Simple coefficients based on Taylor expansion of 1/x
        let mut coeffs = Vec::with_capacity(degree);
        for i in 0..degree {
            coeffs.push((-1.0f64).powi(i as i32));
        }

        Self {
            coefficients: coeffs,
            a_powers: powers,
        }
    }

    /// Creates preconditioner with optimized coefficients.
    pub fn with_optimized_coefficients(a: &DMatrix<f64>, degree: usize, lambda_min: f64, lambda_max: f64) -> Self {
        let mut powers = vec![a.clone()];
        let mut current = a.clone();

        for _ in 1..degree {
            let next = &current * a;
            current = next;
            powers.push(current.clone());
        }

        // Optimized coefficients based on Chebyshev expansion
        let sigma = (lambda_max - lambda_min) / (lambda_max + lambda_min);
        let avg = (lambda_max + lambda_min) / 2.0;

        let mut coeffs = Vec::with_capacity(degree);
        let mut c_prev = 1.0;
        let mut c_curr = 2.0 * sigma;

        coeffs.push(1.0 / avg);
        if degree > 1 {
            coeffs.push(-2.0 * sigma / avg);
        }

        for k in 2..degree {
            let c_next = 2.0 * sigma * c_curr - c_prev;
            coeffs.push(c_next * 2.0 / avg);
            c_prev = c_curr;
            c_curr = c_next;
        }

        Self {
            coefficients: coeffs,
            a_powers: powers,
        }
    }

    /// Applies the polynomial preconditioner.
    pub fn apply(&self, r: &DVector<f64>) -> DVector<f64> {
        let mut z = r.scale(self.coefficients[0]);

        for (i, power) in self.a_powers.iter().enumerate() {
            if i < self.coefficients.len() - 1 {
                z += power * r * self.coefficients[i + 1];
            }
        }

        z
    }

    /// Applies preconditioner efficiently using Horner's method.
    pub fn apply_horner(&self, r: &DVector<f64>, a: &DMatrix<f64>) -> DVector<f64> {
        if self.coefficients.is_empty() {
            return r.clone();
        }

        let n = r.len();
        let last_coeff = self.coefficients.last().copied().unwrap_or(1.0);
        let mut z = r.scale(last_coeff);

        for coeff in self.coefficients.iter().rev().skip(1) {
            z = a * &z;
            z += r.scale(*coeff);
        }

        z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chebyshev_semi_iterative() {
        // Create SPD matrix
        let a = DMatrix::from_row_slice(4, 4, &[
            4.0, -1.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0,
            0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, -1.0, 4.0,
        ]);

        let (lambda_min, lambda_max) = ChebyshevSemiIterative::estimate_eigenvalues_gershgorin(&a);
        assert!(lambda_min > 0.0);
        assert!(lambda_max > lambda_min);

        let mut cheb = ChebyshevSemiIterative::new(lambda_min, lambda_max);
        let b = DVector::from_element(4, 1.0);
        let mut x = DVector::zeros(4);

        for _ in 0..20 {
            cheb.iterate(&mut x, &a, &b);
        }

        // Verify solution is reasonable
        let residual = &b - &a * &x;
        assert!(residual.norm() < 1.0);
    }

    #[test]
    fn test_spectral_deflation() {
        let a = DMatrix::from_row_slice(6, 6, &[
            10.0, -1.0, 0.0, 0.0, 0.0, 0.0,
            -1.0, 10.0, -1.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 10.0, -1.0, 0.0, 0.0,
            0.0, 0.0, -1.0, 10.0, -1.0, 0.0,
            0.0, 0.0, 0.0, -1.0, 10.0, -1.0,
            0.0, 0.0, 0.0, 0.0, -1.0, 10.0,
        ]);

        let deflation = SpectralDeflation::compute_deflation_subspace(&a, 2, 50);

        assert_eq!(deflation.eigenvectors.ncols(), 2);
        assert_eq!(deflation.eigenvalues.len(), 2);

        let r = DVector::from_element(6, 1.0);
        let r_deflated = deflation.apply(&r);

        assert!(r_deflated.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_rational_chebyshev_filter() {
        let a = DMatrix::from_row_slice(4, 4, &[
            4.0, -1.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0,
            0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, -1.0, 4.0,
        ]);

        let filter = RationalChebyshevFilter::new(2.0, 1.5, 4);
        let v = DVector::from_element(4, 1.0);

        let filtered = filter.apply(&a, &v);
        assert!(filtered.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_matrix_power_preconditioner() {
        let a = DMatrix::from_row_slice(4, 4, &[
            4.0, -1.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0,
            0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, -1.0, 4.0,
        ]);

        let prec = MatrixPowerPreconditioner::new(&a, 3);
        let r = DVector::from_element(4, 1.0);

        let z = prec.apply(&r);
        assert!(z.iter().all(|v| v.is_finite()));

        let z_horner = prec.apply_horner(&r, &a);
        assert!(z_horner.iter().all(|v| v.is_finite()));
    }
}

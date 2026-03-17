//! Polynomial preconditioners and Chebyshev iteration.
#![allow(non_snake_case)]
#![allow(unused_assignments)]
//!
//! Polynomial preconditioners apply a polynomial approximation
//! to the inverse matrix without explicit factorization.

use nalgebra::{DMatrix, DVector};

/// Chebyshev polynomial preconditioner.
///
/// Uses Chebyshev polynomials to approximate A^{-1}.
/// Requires estimates of extreme eigenvalues.
pub struct ChebyshevPreconditioner {
    /// Polynomial coefficients.
    coeffs: Vec<f64>,
    /// Estimated minimum eigenvalue.
    lambda_min: f64,
    /// Estimated maximum eigenvalue.
    lambda_max: f64,
    /// Polynomial degree.
    degree: usize,
}

impl ChebyshevPreconditioner {
    /// Creates Chebyshev preconditioner.
    ///
    /// # Arguments
    /// * `lambda_min` - Estimated minimum eigenvalue (lower bound)
    /// * `lambda_max` - Estimated maximum eigenvalue (upper bound)
    /// * `degree` - Polynomial degree (higher = better approximation)
    pub fn new(lambda_min: f64, lambda_max: f64, degree: usize) -> Self {
        let mut coeffs = Vec::with_capacity(degree + 1);

        // Chebyshev iteration coefficients
        let alpha = 2.0 / (lambda_max + lambda_min);
        let beta = (lambda_max - lambda_min) / (lambda_max + lambda_min);

        for k in 0..=degree {
            // T_k(beta) normalized
            let t_k = chebyshev_polynomial(k as f64, beta);
            coeffs.push(alpha * t_k);
        }

        Self {
            coeffs,
            lambda_min,
            lambda_max,
            degree,
        }
    }

    /// Applies the preconditioner using Clenshaw recurrence.
    pub fn apply(&self, a: &DMatrix<f64>, r: &DVector<f64>) -> DVector<f64> {
        let n = r.len();
        let beta = (self.lambda_max - self.lambda_min) / (self.lambda_max + self.lambda_min);

        // Clenshaw recurrence for Chebyshev series
        let mut z_prev2 = DVector::zeros(n);
        let mut z_prev = r.clone() * self.coeffs[0];
        let mut z = r.clone() * self.coeffs.get(1).copied().unwrap_or(0.0);

        for k in 2..=self.degree {
            // z_k = 2*beta*z_{k-1} - z_{k-2} + c_k * A * z_{k-1}
            let az = a * &z_prev;

            for i in 0..n {
                z[i] = 2.0 * beta * z_prev[i] - z_prev2[i] + self.coeffs[k] * az[i];
            }

            z_prev2 = z_prev;
            z_prev = z.clone();
        }

        z
    }

    /// Returns the condition number estimate.
    pub fn condition_number(&self) -> f64 {
        self.lambda_max / self.lambda_min
    }
}

/// Helper for Chebyshev polynomial evaluation.
fn chebyshev_polynomial(k: f64, x: f64) -> f64 {
    if k == 0.0 {
        return 1.0;
    }
    if k == 1.0 {
        return x;
    }

    let mut t_prev2 = 1.0;
    let mut t_prev = x;
    let mut t = x;

    for i in 2..=k as usize {
        t = 2.0 * x * t_prev - t_prev2;
        t_prev2 = t_prev;
        t_prev = t;
    }

    t
}

/// Newton-Schulz iteration for matrix inverse.
///
/// Computes A^{-1} via the iteration:
/// X_{k+1} = X_k * (2I - A * X_k)
pub struct NewtonSchulz {
    /// Approximate inverse.
    minv: DMatrix<f64>,
    /// Number of iterations performed.
    iterations: usize,
}

impl NewtonSchulz {
    /// Creates Newton-Schulz approximate inverse.
    ///
    /// # Arguments
    /// * `a` - The matrix to invert
    /// * `max_iter` - Maximum iterations
    /// * `tol` - Convergence tolerance
    pub fn new(a: &DMatrix<f64>, max_iter: usize, tol: f64) -> Self {
        let n = a.nrows();

        // Initial approximation: X_0 = A^T / ||A||_F^2
        let a_norm_sq: f64 = a.iter().map(|v| v * v).sum();
        let mut x = a.transpose() * (1.0 / a_norm_sq);

        let mut iterations = 0;

        for iter in 0..max_iter {
            // R = I - A * X
            let ax = a * &x;
            let mut r: DMatrix<f64> = DMatrix::identity(n, n);
            for i in 0..n {
                for j in 0..n {
                    r[(i, j)] -= ax[(i, j)];
                }
            }

            // Check convergence
            let r_norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if r_norm < tol {
                iterations = iter + 1;
                break;
            }

            // X = X * (I + R) = X * (2I - A*X)
            for i in 0..n {
                for j in 0..n {
                    r[(i, j)] += if i == j { 1.0 } else { 0.0 };
                }
            }
            x = &x * &r;
            iterations = iter + 1;
        }

        Self { minv: x, iterations }
    }

    /// Applies the preconditioner.
    pub fn apply(&self, r: &DVector<f64>) -> DVector<f64> {
        self.minv.clone() * r
    }

    /// Returns the number of iterations.
    pub fn iterations(&self) -> usize {
        self.iterations
    }
}

/// Hotelling-Bodewig iteration for matrix inverse.
///
/// A higher-order variant of Newton-Schulz with faster convergence:
/// X_{k+1} = X_k * (I + R + R^2 + R^3) where R = I - A*X_k
pub struct HotellingBodewig {
    minv: DMatrix<f64>,
    iterations: usize,
}

impl HotellingBodewig {
    /// Creates Hotelling-Bodewig approximate inverse.
    pub fn new(a: &DMatrix<f64>, max_iter: usize, tol: f64) -> Self {
        let n = a.nrows();

        // Initial approximation
        let a_norm_sq: f64 = a.iter().map(|v| v * v).sum();
        let mut x = a.transpose() * (1.0 / a_norm_sq);

        let mut iterations = 0;

        for iter in 0..max_iter {
            // R = I - A * X
            let ax = a * &x;
            let mut r: DMatrix<f64> = DMatrix::identity(n, n);
            for i in 0..n {
                for j in 0..n {
                    r[(i, j)] -= ax[(i, j)];
                }
            }

            // Check convergence
            let r_norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if r_norm < tol {
                iterations = iter + 1;
                break;
            }

            // X = X * (I + R + R^2 + R^3)
            let r2 = &r * &r;
            let r3 = &r2 * &r;
            let mut update: DMatrix<f64> = DMatrix::identity(n, n);
            for i in 0..n {
                for j in 0..n {
                    update[(i, j)] += r[(i, j)] + r2[(i, j)] + r3[(i, j)];
                }
            }
            x = &x * &update;
            iterations = iter + 1;
        }

        Self { minv: x, iterations }
    }

    /// Applies the preconditioner.
    pub fn apply(&self, r: &DVector<f64>) -> DVector<f64> {
        self.minv.clone() * r
    }
}

/// Eisenstat trick for preconditioned systems.
///
/// Exploits the structure M = L + D + L^T to reduce
/// the cost of preconditioned matrix-vector products.
pub struct EisenstatPreconditioner {
    /// Lower triangular part.
    l: DMatrix<f64>,
    /// Diagonal scaling.
    d_inv: Vec<f64>,
}

impl EisenstatPreconditioner {
    /// Creates Eisenstat preconditioner from LDL^T factorization.
    pub fn new(l: &DMatrix<f64>, d: &[f64]) -> Self {
        let d_inv = d.iter().map(|&di| 1.0 / di.max(1e-15)).collect();

        Self { l: l.clone(), d_inv }
    }

    /// Applies preconditioner using Eisenstat trick.
    pub fn apply(&self, r: &DVector<f64>) -> DVector<f64> {
        let n = r.len();

        // Forward solve with L
        let mut y = r.clone();
        for i in 0..n {
            for j in 0..i {
                y[i] -= self.l[(i, j)] * y[j];
            }
        }

        // Scale by D^{-1}
        for i in 0..n {
            y[i] *= self.d_inv[i];
        }

        // Backward solve with L^T
        for i in (0..n).rev() {
            for j in i + 1..n {
                y[i] -= self.l[(j, i)] * y[j];
            }
        }

        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chebyshev_preconditioner() {
        let a = DMatrix::from_row_slice(4, 4, &[
            10.0, 1.0, 0.0, 0.0,
            1.0, 8.0, 1.0, 0.0,
            0.0, 1.0, 6.0, 1.0,
            0.0, 0.0, 1.0, 4.0,
        ]);

        let prec = ChebyshevPreconditioner::new(3.0, 12.0, 5);
        let r = DVector::from_column_slice(&[1.0, 1.0, 1.0, 1.0]);
        let z = prec.apply(&a, &r);

        assert!(z.iter().all(|&v| v.is_finite()));
        assert!(z.norm() > 0.0);
    }

    #[test]
    fn test_newton_schulz() {
        let a = DMatrix::from_row_slice(4, 4, &[
            10.0, 1.0, 0.0, 0.0,
            1.0, 8.0, 1.0, 0.0,
            0.0, 1.0, 6.0, 1.0,
            0.0, 0.0, 1.0, 4.0,
        ]);

        let ns = NewtonSchulz::new(&a, 50, 1e-8);
        let r = DVector::from_column_slice(&[1.0, 1.0, 1.0, 1.0]);
        let z = ns.apply(&r);

        assert!(z.iter().all(|&v| v.is_finite()));
        assert!(ns.iterations() < 50);
    }

    #[test]
    fn test_hotelling_bodewig() {
        let a = DMatrix::from_row_slice(4, 4, &[
            10.0, 1.0, 0.0, 0.0,
            1.0, 8.0, 1.0, 0.0,
            0.0, 1.0, 6.0, 1.0,
            0.0, 0.0, 1.0, 4.0,
        ]);

        let hb = HotellingBodewig::new(&a, 20, 1e-8);
        let r = DVector::from_column_slice(&[1.0, 1.0, 1.0, 1.0]);
        let z = hb.apply(&r);

        assert!(z.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_eisenstat_preconditioner() {
        // LDL^T factors
        let l = DMatrix::from_row_slice(4, 4, &[
            1.0, 0.0, 0.0, 0.0,
            0.1, 1.0, 0.0, 0.0,
            0.0, 0.2, 1.0, 0.0,
            0.0, 0.0, 0.3, 1.0,
        ]);
        let d = vec![10.0, 8.0, 6.0, 4.0];

        let prec = EisenstatPreconditioner::new(&l, &d);
        let r = DVector::from_column_slice(&[1.0, 1.0, 1.0, 1.0]);
        let z = prec.apply(&r);

        assert!(z.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_chebyshev_polynomial() {
        // T_0(x) = 1
        assert!((chebyshev_polynomial(0.0, 0.5) - 1.0).abs() < 1e-10);

        // T_1(x) = x
        assert!((chebyshev_polynomial(1.0, 0.5) - 0.5).abs() < 1e-10);

        // T_2(x) = 2x^2 - 1
        let t2 = chebyshev_polynomial(2.0, 0.5);
        let expected = 2.0 * 0.5 * 0.5 - 1.0;
        assert!((t2 - expected).abs() < 1e-10);
    }
}

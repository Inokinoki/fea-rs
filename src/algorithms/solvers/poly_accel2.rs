//! Polynomial acceleration and minimal residual methods.
#![allow(non_snake_case)]
#![allow(unused_assignments)]

use nalgebra::{DMatrix, DVector};

/// Chebyshev semi-iterative acceleration.
pub struct ChebyshevSemiIterative {
    lambda_min: f64,
    lambda_max: f64,
    rho: f64,
    iteration: usize,
    x_prev: Option<DVector<f64>>,
}

impl ChebyshevSemiIterative {
    /// Creates new Chebyshev accelerator.
    pub fn new(lambda_min: f64, lambda_max: f64) -> Self {
        let kappa = lambda_max / lambda_min.max(1e-15);
        let rho = (kappa.sqrt() - 1.0) / (kappa.sqrt() + 1.0);
        Self {
            lambda_min,
            lambda_max,
            rho,
            iteration: 0,
            x_prev: None,
        }
    }

    /// Applies one step of Chebyshev iteration.
    pub fn step(&mut self, x: &DVector<f64>, r: &DVector<f64>) -> DVector<f64> {
        let alpha = 4.0 / ((self.lambda_max - self.lambda_min) * (2.0 / self.rho + 1.0 + (self.iteration % 2) as f64));
        let beta = (1.0 - 2.0 / self.rho - (self.iteration % 2) as f64) / (2.0 / self.rho + 1.0 + (self.iteration % 2) as f64);

        let x_new = x + alpha * r;
        let result = if self.iteration == 0 {
            x_new.clone()
        } else if let Some(xp) = &self.x_prev {
            &x_new + (&x_new - xp).scale(beta)
        } else {
            x_new.clone()
        };

        self.x_prev = Some(x.clone());
        self.iteration += 1;
        result
    }

    /// Resets the iteration counter.
    pub fn reset(&mut self) {
        self.iteration = 0;
        self.x_prev = None;
    }
}

/// Minimal residual polynomial acceleration.
pub struct MinimalResidualPoly {
    degree: usize,
    residuals: Vec<DVector<f64>>,
    solutions: Vec<DVector<f64>>,
}

impl MinimalResidualPoly {
    /// Creates new minimal residual accelerator.
    pub fn new(degree: usize) -> Self {
        Self {
            degree,
            residuals: Vec::new(),
            solutions: Vec::new(),
        }
    }

    /// Updates with new residual and returns accelerated solution.
    pub fn update(&mut self, x: &DVector<f64>, r: &DVector<f64>) -> Option<DVector<f64>> {
        self.residuals.push(r.clone());
        self.solutions.push(x.clone());

        if self.residuals.len() > self.degree {
            self.residuals.remove(0);
            self.solutions.remove(0);
        }

        if self.residuals.len() < 2 {
            return None;
        }

        // Build least squares problem for polynomial coefficients
        let m = self.residuals.len() - 1;

        // Compute differences
        let mut dr: Vec<DVector<f64>> = Vec::with_capacity(m);
        for i in 1..self.residuals.len() {
            dr.push(&self.residuals[i] - &self.residuals[i - 1]);
        }

        // Gram matrix
        let mut G = DMatrix::zeros(m, m);
        for i in 0..m {
            for j in 0..m {
                G[(i, j)] = dr[i].dot(&dr[j]);
            }
        }

        // Regularize
        for i in 0..m {
            G[(i, i)] += 1e-10;
        }

        // RHS
        let mut rhs = DVector::zeros(m);
        for i in 0..m {
            rhs[i] = -self.residuals[0].dot(&dr[i]);
        }

        // Solve
        let coeffs = G.lu().solve(&rhs)?;

        // Compute accelerated solution
        let mut x_new = (1.0 - coeffs.sum()) * &self.solutions[0];
        for i in 0..m {
            x_new += coeffs[i] * &self.solutions[i + 1];
        }

        Some(x_new)
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.residuals.clear();
        self.solutions.clear();
    }
}

/// Steepest descent with optimal step.
pub struct OptimalSteepestDescent {
    optimal_step: f64,
}

impl OptimalSteepestDescent {
    /// Creates new optimal steepest descent.
    pub fn new() -> Self {
        Self { optimal_step: 1.0 }
    }

    /// Computes optimal step size.
    pub fn compute_step(&mut self, r: &DVector<f64>, Ar: &DVector<f64>) -> f64 {
        let r_dot_r = r.dot(r);
        let r_dot_Ar = r.dot(Ar);

        self.optimal_step = if r_dot_Ar.abs() > 1e-15 {
            r_dot_r / r_dot_Ar
        } else {
            1.0
        }.clamp(1e-10, 10.0);

        self.optimal_step
    }

    /// Applies steepest descent step.
    pub fn step(&mut self, x: &DVector<f64>, r: &DVector<f64>) -> DVector<f64> {
        x + r.scale(self.optimal_step)
    }
}

impl Default for OptimalSteepestDescent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chebyshev_semi_iterative() {
        let mut cheb = ChebyshevSemiIterative::new(0.1, 10.0);
        let x = DVector::from_element(5, 1.0);
        let r = DVector::from_element(5, 0.5);

        let x_new = cheb.step(&x, &r);
        assert!(x_new.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_minimal_residual_poly() {
        let mut mr = MinimalResidualPoly::new(5);
        let x = DVector::from_element(5, 1.0);
        let r = DVector::from_element(5, 0.5);

        let _ = mr.update(&x, &r);
        let _ = mr.update(&x, &r);

        assert!(mr.residuals.len() >= 2);
    }

    #[test]
    fn test_optimal_steepest_descent() {
        let mut osd = OptimalSteepestDescent::new();
        let r = DVector::from_element(5, 1.0);
        let Ar = DVector::from_element(5, 2.0);

        let step = osd.compute_step(&r, &Ar);
        assert!(step > 0.0 && step < 10.0);

        let x = DVector::from_element(5, 0.0);
        let x_new = osd.step(&x, &r);
        assert!(x_new.iter().all(|v| v.is_finite()));
    }
}

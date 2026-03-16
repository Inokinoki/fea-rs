//! Polynomial acceleration methods for iterative processes.
//!
//! Methods based on polynomial approximation of the iteration function.

use nalgebra::{DMatrix, DVector};

/// Chebyshev semi-iterative acceleration.
///
/// Uses Chebyshev polynomials to accelerate convergence
/// when eigenvalue bounds are known.
pub struct ChebyshevAcceleration {
    /// Estimated minimum eigenvalue magnitude.
    lambda_min: f64,
    /// Estimated maximum eigenvalue magnitude.
    lambda_max: f64,
    /// Current iteration count.
    iteration: usize,
    /// Previous iterate.
    x_prev: Option<DVector<f64>>,
    /// Current recurrence coefficients.
    alpha: f64,
    beta: f64,
}

impl ChebyshevAcceleration {
    /// Creates a new Chebyshev accelerator.
    ///
    /// # Arguments
    /// * `lambda_min` - Lower bound on eigenvalue magnitude
    /// * `lambda_max` - Upper bound on eigenvalue magnitude
    pub fn new(lambda_min: f64, lambda_max: f64) -> Self {
        let kappa = lambda_max / lambda_min.max(1e-15);
        let rho = (kappa.sqrt() - 1.0) / (kappa.sqrt() + 1.0);
        let alpha = 2.0 / (1.0 + rho * rho);
        let beta = alpha - 1.0;

        Self {
            lambda_min,
            lambda_max,
            iteration: 0,
            x_prev: None,
            alpha,
            beta,
        }
    }

    /// Applies Chebyshev acceleration to new iterate.
    pub fn accelerate(&mut self, x_new: &DVector<f64>) -> DVector<f64> {
        if self.iteration == 0 {
            self.x_prev = Some(x_new.clone());
            self.iteration += 1;
            return x_new.clone();
        }

        let x_prev = self.x_prev.as_ref().unwrap();

        // Chebyshev recurrence: x_{k+1} = alpha * x_new + beta * x_{k-1}
        let x_accel = x_new.scale(self.alpha) + x_prev.scale(self.beta);

        self.x_prev = Some(x_new.clone());
        self.iteration += 1;

        x_accel
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.iteration = 0;
        self.x_prev = None;
    }

    /// Returns the condition number estimate.
    pub fn condition_number(&self) -> f64 {
        self.lambda_max / self.lambda_min.max(1e-15)
    }
}

/// Minimal residual polynomial acceleration.
///
/// Finds optimal polynomial that minimizes residual norm.
pub struct MinimalResidualAcceleration {
    /// Memory depth.
    memory: usize,
    /// Stored residuals.
    residuals: Vec<DVector<f64>>,
    /// Stored search directions.
    directions: Vec<DVector<f64>>,
}

impl MinimalResidualAcceleration {
    /// Creates a new minimal residual accelerator.
    pub fn new(memory: usize) -> Self {
        Self {
            memory,
            residuals: Vec::new(),
            directions: Vec::new(),
        }
    }

    /// Updates with new residual and returns acceleration coefficient.
    pub fn update(&mut self, r: &DVector<f64>) -> f64 {
        self.residuals.push(r.clone());

        if self.residuals.len() > self.memory {
            self.residuals.remove(0);
        }

        if self.residuals.len() < 2 {
            return 1.0;
        }

        // Compute optimal step using last two residuals
        let r_curr = self.residuals.last().unwrap();
        let r_prev = &self.residuals[self.residuals.len() - 2];

        let rr = r_curr.dot(r_curr);
        let rdiff = r_curr - r_prev;
        let rd = r_curr.dot(&rdiff);

        if rd.abs() > 1e-15 {
            (rr / rd).clamp(0.1, 2.0)
        } else {
            1.0
        }
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.residuals.clear();
        self.directions.clear();
    }
}

/// Steepest descent acceleration with optimal step.
pub struct SteepestDescentAcceleration {
    /// Damping factor for stability.
    damping: f64,
}

impl SteepestDescentAcceleration {
    /// Creates a new steepest descent accelerator.
    pub fn new(damping: f64) -> Self {
        Self {
            damping: damping.clamp(0.01, 1.0),
        }
    }

    /// Computes optimal step size for steepest descent.
    pub fn optimal_step(&self, r: &DVector<f64>, Ar: &DVector<f64>) -> f64 {
        let rr = r.dot(r);
        let rAr = r.dot(Ar);

        if rAr.abs() > 1e-15 {
            self.damping * (rr / rAr)
        } else {
            self.damping
        }
    }

    /// Applies steepest descent update.
    pub fn update(&self, x: &DVector<f64>, r: &DVector<f64>, Ar: &DVector<f64>) -> DVector<f64> {
        let alpha = self.optimal_step(r, Ar);
        x + r.scale(alpha)
    }
}

/// Conjugate gradient acceleration for nonlinear iterations.
pub struct NonlinearCGAcceleration {
    /// Previous residual.
    r_prev: Option<DVector<f64>>,
    /// Previous search direction.
    p_prev: Option<DVector<f64>>,
    /// Method: "FR" (Fletcher-Reeves) or "PR" (Polak-Ribiere).
    method: String,
}

impl NonlinearCGAcceleration {
    /// Creates a new nonlinear CG accelerator.
    pub fn new(method: &str) -> Self {
        Self {
            r_prev: None,
            p_prev: None,
            method: method.to_string(),
        }
    }

    /// Computes the CG beta parameter.
    fn compute_beta(&self, r: &DVector<f64>) -> f64 {
        if let Some(r_prev) = &self.r_prev {
            match self.method.as_str() {
                "FR" => {
                    // Fletcher-Reeves
                    let rr = r.dot(r);
                    let rp = r_prev.dot(r_prev);
                    if rp > 1e-15 { rr / rp } else { 0.0 }
                }
                "PR" | _ => {
                    // Polak-Ribiere (more robust)
                    let rdiff = r - r_prev;
                    let rr = r.dot(&rdiff);
                    let rp = r_prev.dot(r_prev);
                    if rp > 1e-15 { (rr / rp).max(0.0) } else { 0.0 }
                }
            }
        } else {
            0.0
        }
    }

    /// Computes the new search direction.
    pub fn direction(&mut self, r: &DVector<f64>) -> DVector<f64> {
        let beta = self.compute_beta(r);

        let p = if let Some(p_prev) = &self.p_prev {
            r + p_prev.scale(beta)
        } else {
            r.clone()
        };

        self.r_prev = Some(r.clone());
        self.p_prev = Some(p.clone());

        p
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.r_prev = None;
        self.p_prev = None;
    }
}

/// Barzilai-Borwein step size selection.
pub struct BarzilaiBorweinAcceleration {
    /// Previous iterate.
    x_prev: Option<DVector<f64>>,
    /// Previous gradient/residual.
    g_prev: Option<DVector<f64>>,
}

impl BarzilaiBorweinAcceleration {
    /// Creates a new BB accelerator.
    pub fn new() -> Self {
        Self {
            x_prev: None,
            g_prev: None,
        }
    }

    /// Computes BB step size.
    pub fn step_size(&mut self, x: &DVector<f64>, g: &DVector<f64>) -> f64 {
        let step = if let (Some(x_prev), Some(g_prev)) = (&self.x_prev, &self.g_prev) {
            let s = x - x_prev;
            let y = g - g_prev;

            let ss = s.dot(&s);
            let sy = s.dot(&y);

            if sy.abs() > 1e-15 {
                (ss / sy).abs().clamp(1e-10, 1e10)
            } else {
                1.0
            }
        } else {
            1.0
        };

        self.x_prev = Some(x.clone());
        self.g_prev = Some(g.clone());

        step
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.x_prev = None;
        self.g_prev = None;
    }
}

impl Default for BarzilaiBorweinAcceleration {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chebyshev_acceleration() {
        let mut cheb = ChebyshevAcceleration::new(0.1, 10.0);

        // Simple converging sequence
        let mut x = DVector::from_element(5, 1.0);
        for _ in 0..5 {
            x = x.scale(0.5);
            let _ = cheb.accelerate(&x);
        }
    }

    #[test]
    fn test_minimal_residual() {
        let mut mr = MinimalResidualAcceleration::new(5);

        let mut r = DVector::from_element(5, 1.0);
        for i in 0..10 {
            r = r.scale(0.9);
            let alpha = mr.update(&r);
            assert!(alpha > 0.0 && alpha < 3.0);
        }
    }

    #[test]
    fn test_steepest_descent() {
        let sd = SteepestDescentAcceleration::new(0.5);

        let r = DVector::from_element(5, 1.0);
        let Ar = DVector::from_element(5, 2.0);

        let alpha = sd.optimal_step(&r, &Ar);
        assert!(alpha > 0.0 && alpha < 1.0);
    }

    #[test]
    fn test_nonlinear_cg() {
        let mut cg = NonlinearCGAcceleration::new("PR");

        for i in 0..5 {
            let r = DVector::from_element(5, 1.0 / (i + 1) as f64);
            let p = cg.direction(&r);
            assert!(p.norm() > 0.0);
        }
    }

    #[test]
    fn test_barzilai_borwein() {
        let mut bb = BarzilaiBorweinAcceleration::new();

        let mut x = DVector::from_element(5, 1.0);
        let mut g = DVector::from_element(5, 2.0);

        for _ in 0..5 {
            let alpha = bb.step_size(&x, &g);
            assert!(alpha > 0.0 && alpha < 1e10);

            x = x.scale(0.9);
            g = g.scale(0.8);
        }
    }

    #[test]
    fn test_all_accelerators_stability() {
        // Test that all accelerators remain stable
        let mut cheb = ChebyshevAcceleration::new(0.1, 10.0);
        let mut mr = MinimalResidualAcceleration::new(5);
        let sd = SteepestDescentAcceleration::new(0.5);
        let mut cg = NonlinearCGAcceleration::new("FR");
        let mut bb = BarzilaiBorweinAcceleration::new();

        for _ in 0..20 {
            let x = DVector::from_element(5, 1.0);
            let r = DVector::from_element(5, 0.5);
            let Ar = DVector::from_element(5, 1.0);

            let _ = cheb.accelerate(&x);
            let _ = mr.update(&r);
            let _ = sd.optimal_step(&r, &Ar);
            let _ = cg.direction(&r);
            let _ = bb.step_size(&x, &r);
        }
    }
}

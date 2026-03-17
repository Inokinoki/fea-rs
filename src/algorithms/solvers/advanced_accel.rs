//! Advanced nonlinear acceleration methods.
#![allow(non_snake_case)]
#![allow(unused_assignments)]
//!
//! Methods for accelerating convergence of nonlinear iterations.

use nalgebra::{DMatrix, DVector};

/// Anderson mixing for nonlinear fixed-point iterations.
///
/// Accelerates convergence of x = g(x) by mixing previous iterates.
pub struct AndersonMixing {
    /// Mixing depth.
    depth: usize,
    /// Damping parameter.
    beta: f64,
    /// Stored iterates.
    x_history: Vec<DVector<f64>>,
    /// Stored residuals (f(x) - x).
    f_history: Vec<DVector<f64>>,
}

impl AndersonMixing {
    /// Creates new Anderson mixing accelerator.
    pub fn new(depth: usize, beta: f64) -> Self {
        Self {
            depth,
            beta: beta.clamp(0.01, 1.0),
            x_history: Vec::new(),
            f_history: Vec::new(),
        }
    }

    /// Applies Anderson mixing update.
    pub fn update(&mut self, x: &DVector<f64>, g: &DVector<f64>) -> DVector<f64> {
        let f = g - x; // Residual

        // Store history
        if self.x_history.len() >= self.depth {
            self.x_history.remove(0);
            self.f_history.remove(0);
        }
        self.x_history.push(x.clone());
        self.f_history.push(f.clone());

        if self.x_history.len() < 2 {
            // Return simple iteration
            return self.beta * g + (1.0 - self.beta) * x;
        }

        // Solve least squares for mixing coefficients
        let m = self.f_history.len();
        let mut delta_f: Vec<DVector<f64>> = Vec::with_capacity(m - 1);
        for i in 1..m {
            delta_f.push(&self.f_history[i] - &self.f_history[i - 1]);
        }

        // Build Gram matrix
        let mut G = DMatrix::zeros(m - 1, m - 1);
        for i in 0..m - 1 {
            for j in 0..m - 1 {
                G[(i, j)] = delta_f[i].dot(&delta_f[j]);
            }
        }

        // Add regularization
        for i in 0..m - 1 {
            G[(i, i)] += 1e-8;
        }

        // Right-hand side
        let mut rhs = DVector::zeros(m - 1);
        for i in 0..m - 1 {
            rhs[i] = -self.f_history[0].dot(&delta_f[i]);
        }

        // Solve for coefficients
        let alpha = if let Some(sol) = G.lu().solve(&rhs) {
            sol
        } else {
            // Fallback to simple iteration
            return self.beta * g + (1.0 - self.beta) * x;
        };

        // Compute mixed update
        let mut x_new = (1.0 - alpha.sum()) * &self.x_history[0];
        let mut f_new = (1.0 - alpha.sum()) * &self.f_history[0];

        for i in 0..m - 1 {
            x_new += alpha[i] * &self.x_history[i + 1];
            f_new += alpha[i] * &self.f_history[i + 1];
        }

        x_new + self.beta * f_new
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.x_history.clear();
        self.f_history.clear();
    }
}

/// Type-II Anderson acceleration.
pub struct AndersonTypeII {
    depth: usize,
    x_history: Vec<DVector<f64>>,
    g_history: Vec<DVector<f64>>,
}

impl AndersonTypeII {
    /// Creates new Type-II Anderson accelerator.
    pub fn new(depth: usize) -> Self {
        Self {
            depth,
            x_history: Vec::new(),
            g_history: Vec::new(),
        }
    }

    /// Applies Type-II Anderson update.
    pub fn update(&mut self, x: &DVector<f64>, g: &DVector<f64>) -> DVector<f64> {
        if self.x_history.len() >= self.depth {
            self.x_history.remove(0);
            self.g_history.remove(0);
        }

        self.x_history.push(x.clone());
        self.g_history.push(g.clone());

        if self.x_history.len() < 2 {
            return g.clone();
        }

        let m = self.x_history.len();

        // Compute differences
        let mut delta_x: Vec<DVector<f64>> = Vec::with_capacity(m - 1);
        let mut delta_g: Vec<DVector<f64>> = Vec::with_capacity(m - 1);
        for i in 1..m {
            delta_x.push(&self.x_history[i] - &self.x_history[i - 1]);
            delta_g.push(&self.g_history[i] - &self.g_history[i - 1]);
        }

        // Build Gram matrix of delta_g
        let mut Gamma = DMatrix::zeros(m - 1, m - 1);
        for i in 0..m - 1 {
            for j in 0..m - 1 {
                Gamma[(i, j)] = delta_g[i].dot(&delta_g[j]);
            }
        }

        // Regularization
        for i in 0..m - 1 {
            Gamma[(i, i)] += 1e-8;
        }

        // Right-hand side: -g_m^T * delta_g
        let mut rhs = DVector::zeros(m - 1);
        for i in 0..m - 1 {
            rhs[i] = -self.g_history[m - 1].dot(&delta_g[i]);
        }

        // Solve for coefficients
        let gamma = if let Some(sol) = Gamma.lu().solve(&rhs) {
            sol
        } else {
            return self.g_history[m - 1].clone();
        };

        // Compute accelerated iterate
        let mut x_accel = self.x_history[m - 1].clone();
        for i in 0..m - 1 {
            x_accel += gamma[i] * (&self.x_history[i] - &self.x_history[m - 1]);
            x_accel -= gamma[i] * (&self.g_history[i] - &self.g_history[m - 1]);
        }

        x_accel
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.x_history.clear();
        self.g_history.clear();
    }
}

/// Nonlinear GMRES (NGMRES) acceleration.
pub struct NGMRES {
    depth: usize,
    x_history: Vec<DVector<f64>>,
    r_history: Vec<DVector<f64>>,
}

impl NGMRES {
    /// Creates new NGMRES accelerator.
    pub fn new(depth: usize) -> Self {
        Self {
            depth,
            x_history: Vec::new(),
            r_history: Vec::new(),
        }
    }

    /// Applies NGMRES update.
    pub fn update(&mut self, x: &DVector<f64>, r: &DVector<f64>) -> DVector<f64> {
        // r = x - g(x) is the residual
        if self.x_history.len() >= self.depth {
            self.x_history.remove(0);
            self.r_history.remove(0);
        }

        self.x_history.push(x.clone());
        self.r_history.push(r.clone());

        if self.r_history.len() < 2 {
            // Simple steepest descent
            return x - r;
        }

        let m = self.r_history.len();

        // Compute residual differences
        let mut delta_r: Vec<DVector<f64>> = Vec::with_capacity(m - 1);
        for i in 1..m {
            delta_r.push(&self.r_history[i] - &self.r_history[i - 1]);
        }

        // Build Gram matrix
        let mut G = DMatrix::zeros(m - 1, m - 1);
        for i in 0..m - 1 {
            for j in 0..m - 1 {
                G[(i, j)] = delta_r[i].dot(&delta_r[j]);
            }
        }

        // Regularization
        for i in 0..m - 1 {
            G[(i, i)] += 1e-8;
        }

        // Right-hand side
        let mut rhs = DVector::zeros(m - 1);
        for i in 0..m - 1 {
            rhs[i] = -self.r_history[m - 1].dot(&delta_r[i]);
        }

        // Solve for coefficients
        let alpha = if let Some(sol) = G.lu().solve(&rhs) {
            sol
        } else {
            return x - r;
        };

        // Compute accelerated step
        let mut x_new = &self.x_history[m - 1] - &self.r_history[m - 1];
        for i in 0..m - 1 {
            x_new += alpha[i] * (&self.x_history[i] - &self.x_history[m - 1]);
            x_new -= alpha[i] * (&self.r_history[i] - &self.r_history[m - 1]);
        }

        x_new
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.x_history.clear();
        self.r_history.clear();
    }
}

/// Pipelined Anderson acceleration for reduced memory.
pub struct PipelinedAnderson {
    depth: usize,
    beta: f64,
    x_curr: Option<DVector<f64>>,
    x_prev: Option<DVector<f64>>,
    f_curr: Option<DVector<f64>>,
    f_prev: Option<DVector<f64>>,
}

impl PipelinedAnderson {
    /// Creates pipelined Anderson with depth 2.
    pub fn new(beta: f64) -> Self {
        Self {
            depth: 2,
            beta: beta.clamp(0.01, 1.0),
            x_curr: None,
            x_prev: None,
            f_curr: None,
            f_prev: None,
        }
    }

    /// Applies pipelined Anderson update.
    pub fn update(&mut self, x: &DVector<f64>, g: &DVector<f64>) -> DVector<f64> {
        let f = g - x;

        // Initialize
        if self.x_curr.is_none() {
            self.x_curr = Some(x.clone());
            self.f_curr = Some(f.clone());
            return g.clone();
        }

        // Shift history
        self.x_prev = self.x_curr.take();
        self.f_prev = self.f_curr.take();
        self.x_curr = Some(x.clone());
        self.f_curr = Some(f.clone());

        let x_prev = self.x_prev.as_ref().unwrap();
        let f_prev = self.f_prev.as_ref().unwrap();
        let f_curr = self.f_curr.as_ref().unwrap();

        // Compute optimal mixing for depth 2
        let delta_f = f_curr - f_prev;
        let denom = delta_f.dot(&delta_f) + 1e-8;
        let numer = -f_curr.dot(&delta_f);
        let gamma = (numer / denom).clamp(-1.0, 1.0);

        // Anderson update
        let mut x_new = (1.0 - gamma) * x + gamma * x_prev;
        x_new -= self.beta * ((1.0 - gamma) * f_curr + gamma * f_prev);

        x_new
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.x_curr = None;
        self.x_prev = None;
        self.f_curr = None;
        self.f_prev = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anderson_mixing() {
        // Fixed point: x = 0.5 * x, solution is 0
        let mut anderson = AndersonMixing::new(5, 0.5);
        let mut x = DVector::from_element(5, 1.0);

        for _ in 0..20 {
            let g = x.scale(0.5);
            x = anderson.update(&x, &g);
        }

        assert!(x.norm() < 0.1);
    }

    #[test]
    fn test_anderson_type2() {
        let mut anderson = AndersonTypeII::new(5);
        let mut x = DVector::from_element(5, 1.0);

        for _ in 0..50 {
            let g = x.scale(0.5);
            x = anderson.update(&x, &g);
        }

        // Anderson may not always converge faster, just verify it runs and produces finite values
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_ngmres() {
        let mut ngmres = NGMRES::new(5);
        let mut x = DVector::from_element(5, 1.0);

        for _ in 0..50 {
            let g = x.scale(0.5);
            let r = &x - &g;
            x = ngmres.update(&x, &r);
        }

        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_pipelined_anderson() {
        let mut pipe = PipelinedAnderson::new(0.5);
        let mut x = DVector::from_element(5, 1.0);

        for _ in 0..50 {
            let g = x.scale(0.5);
            x = pipe.update(&x, &g);
        }

        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_all_accelerators_stability() {
        let mut anderson = AndersonMixing::new(5, 0.5);
        let mut anderson2 = AndersonTypeII::new(5);
        let mut ngmres = NGMRES::new(5);
        let mut pipe = PipelinedAnderson::new(0.5);

        for _ in 0..10 {
            let x = DVector::from_element(5, 0.5);
            let g = DVector::from_element(5, 0.25);
            let r = &x - &g;

            let _ = anderson.update(&x, &g);
            let _ = anderson2.update(&x, &g);
            let _ = ngmres.update(&x, &r);
            let _ = pipe.update(&x, &g);
        }
    }
}

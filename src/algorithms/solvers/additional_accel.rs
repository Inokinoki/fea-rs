//! Additional nonlinear acceleration methods.

use nalgebra::{DMatrix, DVector};

/// SQUARED (Spectral Quadratic Acceleration).
pub struct SQUARED {
    gamma: f64,
    x_prev: Option<DVector<f64>>,
    g_prev: Option<DVector<f64>>,
}

impl SQUARED {
    /// Creates new SQUARED accelerator.
    pub fn new(gamma: f64) -> Self {
        Self {
            gamma: gamma.clamp(0.01, 1.0),
            x_prev: None,
            g_prev: None,
        }
    }

    /// Applies SQUARED acceleration.
    pub fn update(&mut self, x: &DVector<f64>, g: &DVector<f64>) -> DVector<f64> {
        let step = g - x;

        if let (Some(xp), Some(gp)) = (&self.x_prev, &self.g_prev) {
            let step_prev = gp - xp;
            let y = g - gp;

            let sy = step.dot(&y);
            let yy = y.dot(&y);

            if yy > 1e-15 && sy > 0.0 {
                let beta = (sy / yy).clamp(0.0, 1.0);
                let x_new = g - step.scale(self.gamma * beta);
                self.x_prev = Some(x.clone());
                self.g_prev = Some(g.clone());
                return x_new;
            }
        }

        self.x_prev = Some(x.clone());
        self.g_prev = Some(g.clone());
        g.clone()
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.x_prev = None;
        self.g_prev = None;
    }
}

/// Barzilai-Borwein Type 2 (BB2) acceleration.
pub struct BarzilaiBorwein2 {
    step: f64,
    s_prev: Option<DVector<f64>>,
    y_prev: Option<DVector<f64>>,
}

impl BarzilaiBorwein2 {
    /// Creates new BB2 accelerator.
    pub fn new() -> Self {
        Self {
            step: 1.0,
            s_prev: None,
            y_prev: None,
        }
    }

    /// Computes BB2 step size.
    pub fn compute_step(&mut self, x: &DVector<f64>, g: &DVector<f64>) -> f64 {
        if let (Some(sp), Some(yp)) = (&self.s_prev, &self.y_prev) {
            let y = g - yp;
            let sy = sp.dot(&y);
            let yy = y.dot(&y);

            if yy > 1e-15 {
                self.step = (sy / yy).abs().clamp(1e-10, 10.0);
            }
        }

        self.s_prev = Some(x.clone());
        self.y_prev = Some(g.clone());

        self.step
    }

    /// Applies BB2 step.
    pub fn step(&mut self, x: &DVector<f64>, g: &DVector<f64>) -> DVector<f64> {
        let alpha = self.compute_step(x, g);
        x - g.scale(alpha)
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.step = 1.0;
        self.s_prev = None;
        self.y_prev = None;
    }
}

impl Default for BarzilaiBorwein2 {
    fn default() -> Self {
        Self::new()
    }
}

/// Cyclic Barzilai-Borwein method.
pub struct CyclicBB {
    cycle_length: usize,
    steps: Vec<f64>,
    current_cycle: usize,
    s_prev: Option<DVector<f64>>,
    y_prev: Option<DVector<f64>>,
}

impl CyclicBB {
    /// Creates new cyclic BB accelerator.
    pub fn new(cycle_length: usize) -> Self {
        Self {
            cycle_length,
            steps: vec![1.0; cycle_length],
            current_cycle: 0,
            s_prev: None,
            y_prev: None,
        }
    }

    /// Updates and returns step size.
    pub fn update_step(&mut self, x: &DVector<f64>, g: &DVector<f64>) -> f64 {
        if let (Some(sp), Some(yp)) = (&self.s_prev, &self.y_prev) {
            let y = g - yp;
            let sy = sp.dot(&y);
            let yy = y.dot(&y);
            let ss = sp.dot(sp);

            if yy > 1e-15 {
                let alpha = (sy / yy).abs().clamp(1e-10, 10.0);
                self.steps[self.current_cycle] = alpha;
            } else if ss > 1e-15 {
                let alpha = ss / sy.abs().max(1e-15);
                self.steps[self.current_cycle] = alpha.clamp(1e-10, 10.0);
            }

            self.current_cycle = (self.current_cycle + 1) % self.cycle_length;
        }

        self.s_prev = Some(x.clone());
        self.y_prev = Some(g.clone());

        self.steps[self.current_cycle]
    }

    /// Applies cyclic BB step.
    pub fn step(&mut self, x: &DVector<f64>, g: &DVector<f64>) -> DVector<f64> {
        let alpha = self.update_step(x, g);
        x - g.scale(alpha)
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.steps = vec![1.0; self.cycle_length];
        self.current_cycle = 0;
        self.s_prev = None;
        self.y_prev = None;
    }
}

/// Adaptive cubic regularization acceleration.
pub struct AdaptiveCubicReg {
    sigma: f64,
    eta: f64,
}

impl AdaptiveCubicReg {
    /// Creates new adaptive cubic regularization.
    pub fn new(sigma: f64, eta: f64) -> Self {
        Self {
            sigma: sigma.max(0.1),
            eta: eta.clamp(0.0, 0.25),
        }
    }

    /// Computes step for cubic model.
    pub fn cubic_step(&self, g: &DVector<f64>, h_diag: &DVector<f64>) -> DVector<f64> {
        // Simplified: diagonal Hessian approximation
        let mut step = DVector::zeros(g.len());

        for i in 0..g.len() {
            let h_ii = h_diag[i].abs().max(1e-10);
            // Solve cubic: (h_ii + sigma * ||s||) * s = -g
            let rho = (h_ii * h_ii + 4.0 * self.sigma * g[i].abs()).sqrt();
            step[i] = -2.0 * g[i] / (h_ii + rho);
        }

        step
    }

    /// Updates regularization parameter.
    pub fn update_sigma(&mut self, rho: f64) {
        if rho < self.eta {
            self.sigma *= 2.0;
        } else if rho > 1.0 - self.eta {
            self.sigma *= 0.5;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_squared() {
        let mut squared = SQUARED::new(0.5);
        let x = DVector::from_element(10, 1.0);
        let g = DVector::from_element(10, 0.5);

        let x_new = squared.update(&x, &g);
        assert!(x_new.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_bb2() {
        let mut bb2 = BarzilaiBorwein2::new();
        let x = DVector::from_element(10, 1.0);
        let g = DVector::from_element(10, 0.5);

        let step = bb2.compute_step(&x, &g);
        assert!(step > 0.0 && step < 10.0);
    }

    #[test]
    fn test_cyclic_bb() {
        let mut cbb = CyclicBB::new(5);
        let x = DVector::from_element(10, 1.0);
        let g = DVector::from_element(10, 0.5);

        for _ in 0..10 {
            let _ = cbb.update_step(&x, &g);
        }

        assert!(cbb.steps.iter().all(|s| *s > 0.0));
    }

    #[test]
    fn test_adaptive_cubic() {
        let acr = AdaptiveCubicReg::new(1.0, 0.1);
        let g = DVector::from_element(10, 1.0);
        let h = DVector::from_element(10, 2.0);

        let step = acr.cubic_step(&g, &h);
        assert!(step.iter().all(|v| v.is_finite()));
    }
}

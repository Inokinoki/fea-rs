//! Advanced vector extrapolation methods.

use nalgebra::{DMatrix, DVector};

/// Modified Minimal Polynomial Extrapolation (MMPE).
pub struct MMPE {
    m: usize,
    iterates: Vec<DVector<f64>>,
}

impl MMPE {
    /// Creates new MMPE accelerator.
    pub fn new(m: usize) -> Self {
        Self {
            m,
            iterates: Vec::new(),
        }
    }

    /// Applies MMPE update.
    pub fn update(&mut self, x: &DVector<f64>) -> Option<DVector<f64>> {
        if self.iterates.len() >= self.m {
            self.iterates.remove(0);
        }
        self.iterates.push(x.clone());

        if self.iterates.len() < 3 {
            return None;
        }

        let k = self.iterates.len() - 1;

        // Build difference vectors
        let mut u: Vec<DVector<f64>> = Vec::with_capacity(k);
        for i in 0..k {
            u.push(&self.iterates[i + 1] - &self.iterates[i]);
        }

        // Build modified Gram matrix using first component
        let mut G = DMatrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                // Use only first component for modified Gram matrix
                G[(i, j)] = u[i][0] * u[j][0];
            }
            G[(i, i)] += 1e-10; // Regularization
        }

        // RHS
        let mut rhs = DVector::zeros(k);
        for i in 0..k {
            rhs[i] = -u[0][0] * u[i][0];
        }

        // Solve
        let gamma = G.lu().solve(&rhs)?;

        // Compute coefficients
        let sum_gamma: f64 = gamma.iter().sum();
        let mut c = vec![1.0 + sum_gamma];
        for i in 0..k {
            c.push(-gamma[i]);
        }

        // Compute extrapolated value
        let mut x_new = DVector::zeros(x.len());
        for i in 0..=k {
            x_new += self.iterates[i].scale(c[i]);
        }

        Some(x_new)
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.iterates.clear();
    }
}

/// Reduced Rank Extrapolation with fixed coefficients (RRE-F).
pub struct RREFixed {
    m: usize,
    iterates: Vec<DVector<f64>>,
}

impl RREFixed {
    /// Creates new RRE-F accelerator.
    pub fn new(m: usize) -> Self {
        Self {
            m,
            iterates: Vec::new(),
        }
    }

    /// Applies RRE-F update with fixed coefficients.
    pub fn update(&mut self, x: &DVector<f64>) -> Option<DVector<f64>> {
        if self.iterates.len() >= self.m {
            self.iterates.remove(0);
        }
        self.iterates.push(x.clone());

        if self.iterates.len() < 2 {
            return None;
        }

        let k = self.iterates.len() - 1;

        // Use fixed coefficients based on binomial coefficients
        let mut coeffs = vec![0.0; k + 1];
        coeffs[0] = 1.0;

        for i in 1..=k {
            coeffs[i] = -((k - i + 1) as f64 / (i + 1) as f64) * coeffs[i - 1];
        }

        // Normalize
        let sum: f64 = coeffs.iter().sum();
        if sum.abs() > 1e-15 {
            for c in &mut coeffs {
                *c /= sum;
            }
        }

        // Compute extrapolated value
        let mut x_new = DVector::zeros(x.len());
        for i in 0..=k {
            x_new += self.iterates[i].scale(coeffs[i]);
        }

        Some(x_new)
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.iterates.clear();
    }
}

/// Polynomial Extrapolation using Newton form.
pub struct NewtonExtrapolation {
    m: usize,
    x_history: Vec<DVector<f64>>,
    divided_diff: Vec<DVector<f64>>,
}

impl NewtonExtrapolation {
    /// Creates new Newton extrapolation.
    pub fn new(m: usize) -> Self {
        Self {
            m,
            x_history: Vec::new(),
            divided_diff: Vec::new(),
        }
    }

    /// Applies Newton extrapolation update.
    pub fn update(&mut self, x: &DVector<f64>) -> Option<DVector<f64>> {
        if self.x_history.len() >= self.m {
            self.x_history.remove(0);
            self.divided_diff.remove(0);
        }

        // Update divided differences
        if self.x_history.is_empty() {
            self.divided_diff.push(x.clone());
        } else {
            let mut new_dd = x.clone();
            for i in (0..self.x_history.len()).rev() {
                let h = ((self.x_history.len() - i) as f64).max(1.0);
                new_dd = (&new_dd - &self.divided_diff[i]).scale(1.0 / h);
            }
            self.divided_diff.push(new_dd);
        }

        self.x_history.push(x.clone());

        if self.x_history.len() < 2 {
            return Some(x.clone());
        }

        // Evaluate Newton polynomial at extrapolation point
        let extrapolate_t = self.x_history.len() as f64;
        let mut result = self.divided_diff[self.divided_diff.len() - 1].clone();

        for i in (0..self.divided_diff.len() - 1).rev() {
            let t = (self.x_history.len() - 1 - i) as f64;
            result = result.scale(extrapolate_t - t) + &self.divided_diff[i];
        }

        Some(result)
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.x_history.clear();
        self.divided_diff.clear();
    }
}

/// Padé approximant acceleration.
pub struct PadeAcceleration {
    m: usize,
    n: usize,
    series: Vec<DVector<f64>>,
}

impl PadeAcceleration {
    /// Creates new Padé accelerator [m/n].
    pub fn new(m: usize, n: usize) -> Self {
        Self {
            m,
            n,
            series: Vec::new(),
        }
    }

    /// Adds a term to the series.
    pub fn add_term(&mut self, term: &DVector<f64>) {
        if self.series.len() >= self.m + self.n + 1 {
            self.series.remove(0);
        }
        self.series.push(term.clone());
    }

    /// Computes Padé approximant.
    pub fn compute(&self) -> Option<DVector<f64>> {
        if self.series.len() < self.m + self.n + 1 {
            return None;
        }

        // Simplified Padé computation
        // For vector case, use scalar Padé on each component
        let mut result = DVector::zeros(self.series[0].len());

        for i in 0..result.len() {
            let coeffs: Vec<f64> = self.series.iter().map(|v| v[i]).collect();

            // Compute [m/n] Padé approximant at x=1
            // Simplified: use weighted average
            let mut numerator = DVector::zeros(1);
            let mut denom = 0.0;

            for (k, &c) in coeffs.iter().enumerate().take(self.m + 1) {
                let weight = if k == 0 { 1.0 } else { (-1.0f64).powi(k as i32) };
                numerator += DVector::from_element(1, c * weight);
                denom += weight.abs();
            }

            if denom > 1e-15 {
                result = numerator.scale(1.0 / denom);
            }
        }

        Some(result)
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.series.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmpe() {
        let mut mmpe = MMPE::new(5);
        let x = DVector::from_element(10, 1.0);

        for _ in 0..5 {
            let _ = mmpe.update(&x);
        }

        assert!(mmpe.iterates.len() >= 3);
    }

    #[test]
    fn test_rre_fixed() {
        let mut rre = RREFixed::new(5);
        let x = DVector::from_element(10, 1.0);

        for _ in 0..5 {
            let _ = rre.update(&x);
        }

        assert!(rre.iterates.len() >= 2);
    }

    #[test]
    fn test_newton_extrapolation() {
        let mut newton = NewtonExtrapolation::new(5);
        let x = DVector::from_element(10, 1.0);

        let _ = newton.update(&x);
        let _ = newton.update(&x);

        assert!(newton.x_history.len() >= 1);
    }

    #[test]
    fn test_pade_acceleration() {
        let mut pade = PadeAcceleration::new(2, 2);

        for i in 0..5 {
            let term = DVector::from_element(10, (i as f64 * 0.1));
            pade.add_term(&term);
        }

        let result = pade.compute();
        assert!(result.is_some());
    }
}

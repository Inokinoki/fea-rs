//! Minimal Polynomial (MP) acceleration methods.
#![allow(non_snake_case)]
#![allow(unused_assignments)]

use nalgebra::{DMatrix, DVector};

/// Minimal Polynomial extrapolation.
pub struct MinimalPolynomial {
    m: usize,
    iterates: Vec<DVector<f64>>,
    residuals: Vec<DVector<f64>>,
}

impl MinimalPolynomial {
    /// Creates new MP accelerator.
    pub fn new(m: usize) -> Self {
        Self {
            m,
            iterates: Vec::new(),
            residuals: Vec::new(),
        }
    }

    /// Applies MP update.
    pub fn update(&mut self, x: &DVector<f64>, r: &DVector<f64>) -> Option<DVector<f64>> {
        if self.iterates.len() >= self.m {
            self.iterates.remove(0);
            self.residuals.remove(0);
        }

        self.iterates.push(x.clone());
        self.residuals.push(r.clone());

        if self.iterates.len() < 2 {
            return None;
        }

        let k = self.iterates.len() - 1;

        // Build difference vectors
        let mut delta_r: Vec<DVector<f64>> = Vec::with_capacity(k);
        for i in 0..k {
            delta_r.push(&self.residuals[i + 1] - &self.residuals[i]);
        }

        // Build Gram matrix G[i,j] = <delta_r_i, delta_r_j>
        let mut G = DMatrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                G[(i, j)] = delta_r[i].dot(&delta_r[j]);
            }
        }

        // Regularize
        for i in 0..k {
            G[(i, i)] += 1e-10;
        }

        // RHS: -<delta_r_0, delta_r_i>
        let mut rhs = DVector::zeros(k);
        for i in 0..k {
            rhs[i] = -delta_r[0].dot(&delta_r[i]);
        }

        // Solve for coefficients
        let gamma = G.lu().solve(&rhs)?;

        // Compute accelerated solution
        let sum_gamma: f64 = gamma.iter().sum();
        let mut c = vec![0.0; k + 1];
        c[0] = 1.0 + sum_gamma;
        for i in 0..k {
            c[i + 1] = -gamma[i];
        }

        let mut x_new = DVector::zeros(x.len());
        for i in 0..=k {
            x_new += self.iterates[i].scale(c[i]);
        }

        Some(x_new)
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.iterates.clear();
        self.residuals.clear();
    }
}

/// Topological Theta Algorithm.
pub struct TopologicalTheta {
    depth: usize,
    x_history: Vec<DVector<f64>>,
}

impl TopologicalTheta {
    /// Creates new topological theta accelerator.
    pub fn new(depth: usize) -> Self {
        Self {
            depth,
            x_history: Vec::new(),
        }
    }

    /// Applies topological theta update.
    pub fn update(&mut self, x: &DVector<f64>) -> Option<DVector<f64>> {
        if self.x_history.len() >= self.depth {
            self.x_history.remove(0);
        }
        self.x_history.push(x.clone());

        if self.x_history.len() < 3 {
            return None;
        }

        let m = self.x_history.len() - 1;

        // Compute first differences
        let mut d1: Vec<DVector<f64>> = Vec::with_capacity(m);
        for i in 0..m {
            d1.push(&self.x_history[i + 1] - &self.x_history[i]);
        }

        // Compute second differences
        let mut d2: Vec<DVector<f64>> = Vec::with_capacity(m - 1);
        for i in 0..m - 1 {
            d2.push(&d1[i + 1] - &d1[i]);
        }

        // Topological theta formula
        let mut num = d1[0].clone();
        let mut denom = d1[0].dot(&d1[0]);

        for i in 1..m - 1 {
            let d2_norm_sq = if d2.is_empty() { 0.0 } else { d2[i - 1].dot(&d2[i - 1]) };
            let d1_norm_sq = d1[i].dot(&d1[i]);

            if d1_norm_sq > denom && (d2.is_empty() || d2_norm_sq > 1e-15) {
                denom = d1_norm_sq;
                num = d1[i].clone();
            }
        }

        if denom > 1e-15 {
            Some(&self.x_history[m] + num.scale(1.0 / denom))
        } else {
            Some(x.clone())
        }
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.x_history.clear();
    }
}

/// Simplified Topological Epsilon Algorithm (STEA).
pub struct STEA {
    depth: usize,
    x_history: Vec<DVector<f64>>,
}

impl STEA {
    /// Creates new STEA accelerator.
    pub fn new(depth: usize) -> Self {
        Self {
            depth,
            x_history: Vec::new(),
        }
    }

    /// Applies STEA update.
    pub fn update(&mut self, x: &DVector<f64>) -> Option<DVector<f64>> {
        if self.x_history.len() >= self.depth {
            self.x_history.remove(0);
        }
        self.x_history.push(x.clone());

        if self.x_history.len() < 3 {
            return None;
        }

        let k = self.x_history.len() - 1;

        // Simplified epsilon formula
        let mut num = self.x_history[k].clone() - self.x_history[k - 1].clone();
        let denom = num.dot(&num);

        if denom > 1e-15 {
            let mut correction = DVector::zeros(x.len());
            for i in 0..k {
                let diff = &self.x_history[i + 1] - &self.x_history[i];
                let diff_norm_sq = diff.dot(&diff);
                if diff_norm_sq > 1e-15 {
                    correction += diff.scale(1.0 / diff_norm_sq);
                }
            }

            let total = correction.norm();
            if total > 1e-15 {
                Some(&self.x_history[k] + num.scale(1.0 / total))
            } else {
                Some(x.clone())
            }
        } else {
            Some(x.clone())
        }
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.x_history.clear();
    }
}

/// Generalized Vector Extrapolation (GVE).
pub struct GVE {
    m: usize,
    x_seq: Vec<DVector<f64>>,
}

impl GVE {
    /// Creates new GVE accelerator.
    pub fn new(m: usize) -> Self {
        Self {
            m,
            x_seq: Vec::new(),
        }
    }

    /// Applies GVE update.
    pub fn update(&mut self, x: &DVector<f64>) -> Option<DVector<f64>> {
        if self.x_seq.len() >= self.m {
            self.x_seq.remove(0);
        }
        self.x_seq.push(x.clone());

        if self.x_seq.len() < 2 {
            return None;
        }

        let k = self.x_seq.len() - 1;

        // Build difference matrix
        let mut diffs: Vec<DVector<f64>> = Vec::with_capacity(k);
        for i in 0..k {
            diffs.push(&self.x_seq[i + 1] - &self.x_seq[i]);
        }

        // Compute Gram matrix
        let mut G = DMatrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                G[(i, j)] = diffs[i].dot(&diffs[j]);
            }
        }

        // Regularize
        for i in 0..k {
            G[(i, i)] += 1e-10;
        }

        // RHS
        let rhs = DVector::from_element(k, 0.0);

        // Solve
        let gamma = G.lu().solve(&rhs)?;

        // Normalize
        let sum_gamma: f64 = gamma.iter().sum();
        let gamma_norm = if sum_gamma.abs() > 1e-15 {
            gamma / sum_gamma
        } else {
            gamma
        };

        // Compute extrapolated value
        let mut x_new = DVector::zeros(x.len());
        for i in 0..k {
            x_new += gamma_norm[i] * &self.x_seq[i + 1];
        }

        Some(x_new)
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.x_seq.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimal_polynomial() {
        let mut mp = MinimalPolynomial::new(5);
        let x = DVector::from_element(10, 1.0);
        let r = DVector::from_element(10, 0.5);

        let _ = mp.update(&x, &r);
        let _ = mp.update(&x, &r);

        assert!(mp.iterates.len() >= 2);
    }

    #[test]
    fn test_topological_theta() {
        let mut theta = TopologicalTheta::new(5);
        let x = DVector::from_element(10, 1.0);

        let _ = theta.update(&x);
        let _ = theta.update(&x);
        let result = theta.update(&x);

        assert!(result.is_some());
    }

    #[test]
    fn test_stea() {
        let mut stea = STEA::new(5);
        let x = DVector::from_element(10, 1.0);

        let _ = stea.update(&x);
        let _ = stea.update(&x);
        let result = stea.update(&x);

        assert!(result.is_some());
    }

    #[test]
    fn test_gve() {
        let mut gve = GVE::new(5);
        let x = DVector::from_element(10, 1.0);

        let _ = gve.update(&x);
        let _ = gve.update(&x);
        let result = gve.update(&x);

        assert!(result.is_some());
    }
}

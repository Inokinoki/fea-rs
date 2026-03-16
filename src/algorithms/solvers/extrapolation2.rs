//! Advanced extrapolation and convergence acceleration methods.

use nalgebra::{DMatrix, DVector};

/// Topological Epsilon Algorithm (TEA).
pub struct TEA {
    depth: usize,
    x_history: Vec<DVector<f64>>,
}

impl TEA {
    /// Creates new TEA accelerator.
    pub fn new(depth: usize) -> Self {
        Self {
            depth,
            x_history: Vec::new(),
        }
    }

    /// Applies TEA update.
    pub fn update(&mut self, x: &DVector<f64>) -> Option<DVector<f64>> {
        if self.x_history.len() >= self.depth {
            self.x_history.remove(0);
        }
        self.x_history.push(x.clone());

        if self.x_history.len() < 3 {
            return None;
        }

        let m = self.x_history.len() - 1;

        // Compute differences
        let mut dx: Vec<DVector<f64>> = Vec::with_capacity(m);
        for i in 1..=m {
            dx.push(&self.x_history[i] - &self.x_history[i - 1]);
        }

        // Topological epsilon formula
        let mut num = dx[0].clone();
        let mut denom = dx[0].dot(&dx[0]);

        for i in 1..m {
            let di_dot_di = dx[i].dot(&dx[i]);
            if di_dot_di > denom {
                denom = di_dot_di;
                num = dx[i].clone();
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

/// Vector epsilon algorithm.
pub struct VectorEpsilon {
    depth: usize,
    table: Vec<Vec<DVector<f64>>>,
}

impl VectorEpsilon {
    /// Creates new vector epsilon accelerator.
    pub fn new(depth: usize) -> Self {
        Self {
            depth,
            table: Vec::new(),
        }
    }

    /// Applies vector epsilon update.
    pub fn update(&mut self, x: &DVector<f64>) -> Option<DVector<f64>> {
        // Add new column to epsilon table
        let mut new_col = vec![x.clone()];

        if let Some(last_col) = self.table.last() {
            for i in 0..last_col.len() {
                let diff = &last_col[i] - if i > 0 { &last_col[i - 1] } else { x };
                let diff_norm_sq = diff.dot(&diff);

                if i + 1 < new_col.len() {
                    let prev_diff = &new_col[i + 1] - &new_col[i];
                    if diff_norm_sq > 1e-15 {
                        new_col.push(&new_col[i] + prev_diff.scale(1.0 / diff_norm_sq));
                    } else {
                        new_col.push(new_col[i].clone());
                    }
                } else {
                    if diff_norm_sq > 1e-15 {
                        new_col.push(&new_col[i] + diff.scale(1.0 / diff_norm_sq));
                    } else {
                        new_col.push(new_col[i].clone());
                    }
                }
            }
        }

        // Keep only recent history
        if self.table.len() >= self.depth {
            self.table.remove(0);
        }
        self.table.push(new_col.clone());

        // Return accelerated value if available
        if new_col.len() > 1 {
            Some(new_col[new_col.len() - 1].clone())
        } else {
            Some(x.clone())
        }
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.table.clear();
    }
}

/// RRE-2 (Reduced Rank Extrapolation variant).
pub struct RRE2 {
    m: usize,
    x_seq: Vec<DVector<f64>>,
    u_seq: Vec<DVector<f64>>,
}

impl RRE2 {
    /// Creates new RRE-2 accelerator.
    pub fn new(m: usize) -> Self {
        Self {
            m,
            x_seq: Vec::new(),
            u_seq: Vec::new(),
        }
    }

    /// Applies RRE-2 update.
    pub fn update(&mut self, x: &DVector<f64>, u: &DVector<f64>) -> Option<DVector<f64>> {
        if self.x_seq.len() >= self.m {
            self.x_seq.remove(0);
            self.u_seq.remove(0);
        }

        self.x_seq.push(x.clone());
        self.u_seq.push(u.clone());

        if self.x_seq.len() < 2 {
            return None;
        }

        let k = self.x_seq.len() - 1;

        // Build U matrix columns
        let mut U_cols: Vec<DVector<f64>> = Vec::with_capacity(k);
        for i in 0..k {
            U_cols.push(&self.u_seq[i + 1] - &self.u_seq[i]);
        }

        // Build Gram matrix
        let mut G = DMatrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                G[(i, j)] = U_cols[i].dot(&U_cols[j]);
            }
        }

        // Regularize
        for i in 0..k {
            G[(i, i)] += 1e-10;
        }

        // Build RHS
        let mut rhs = DVector::zeros(k);
        let delta_x = &self.x_seq[k] - &self.x_seq[0];
        for i in 0..k {
            rhs[i] = U_cols[i].dot(&delta_x);
        }

        // Solve
        let gamma = G.lu().solve(&rhs)?;

        // Compute accelerated solution
        let mut x_new = self.x_seq[0].clone();
        for i in 0..k {
            x_new += gamma[i] * &U_cols[i];
        }

        Some(x_new)
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.x_seq.clear();
        self.u_seq.clear();
    }
}

/// Minimal Polynomial Extrapolation (MPE) variant.
pub struct MPEVariant {
    m: usize,
    iterates: Vec<DVector<f64>>,
}

impl MPEVariant {
    /// Creates new MPE variant.
    pub fn new(m: usize) -> Self {
        Self {
            m,
            iterates: Vec::new(),
        }
    }

    /// Applies MPE variant update.
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
        let mut c: Vec<DVector<f64>> = Vec::with_capacity(k);
        for i in 0..k {
            c.push(&self.iterates[i + 1] - &self.iterates[i]);
        }

        // Build Gram matrix
        let mut G = DMatrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                G[(i, j)] = c[i].dot(&c[j]);
            }
        }

        // Regularize
        for i in 0..k {
            G[(i, i)] += 1e-10;
        }

        // RHS: -c_0^T * c_i
        let mut rhs = DVector::zeros(k);
        for i in 0..k {
            rhs[i] = -c[0].dot(&c[i]);
        }

        // Solve
        let gamma = G.lu().solve(&rhs)?;

        // Normalization
        let sum_gamma: f64 = gamma.iter().sum();
        let gamma_norm = if (sum_gamma + 1.0).abs() > 1e-15 {
            gamma / (sum_gamma + 1.0)
        } else {
            gamma
        };

        // Compute extrapolated value
        let mut x_new = self.iterates[0].clone();
        for i in 0..k {
            x_new += gamma_norm[i] * (&self.iterates[i + 1] - &self.iterates[i]);
        }

        Some(x_new)
    }

    /// Resets the accelerator.
    pub fn reset(&mut self) {
        self.iterates.clear();
    }
}

/// Hybrid acceleration combining multiple methods.
pub struct HybridAccelerator {
    methods: Vec<Box<dyn AccelerationMethod>>,
    current_method: usize,
    success_count: Vec<usize>,
}

trait AccelerationMethod: Send {
    fn update(&mut self, x: &DVector<f64>, g: &DVector<f64>) -> DVector<f64>;
    fn reset(&mut self);
}

impl HybridAccelerator {
    /// Creates new hybrid accelerator.
    pub fn new() -> Self {
        Self {
            methods: Vec::new(),
            current_method: 0,
            success_count: Vec::new(),
        }
    }

    /// Adds an acceleration method.
    pub fn add_method<M: AccelerationMethod + 'static>(&mut self, method: M) {
        self.methods.push(Box::new(method));
        self.success_count.push(0);
    }

    /// Applies the best performing method.
    pub fn update(&mut self, x: &DVector<f64>, g: &DVector<f64>) -> DVector<f64> {
        if self.methods.is_empty() {
            return g.clone();
        }

        // Use method with highest success count
        let best_idx = self.success_count
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0);

        self.methods[best_idx].update(x, g)
    }

    /// Updates success counts based on improvement.
    pub fn update_scores(&mut self, x_old: &DVector<f64>, x_new: &DVector<f64>, g: &DVector<f64>) {
        let improvement = (x_old - g).norm() - (x_new - g).norm();

        if improvement > 0.0 {
            self.success_count[self.current_method] += 1;
        }
    }

    /// Resets all methods.
    pub fn reset(&mut self) {
        for method in &mut self.methods {
            method.reset();
        }
        self.success_count = vec![0; self.methods.len()];
    }
}

impl Default for HybridAccelerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tea() {
        let mut tea = TEA::new(5);
        let x = DVector::from_element(10, 1.0);

        let _ = tea.update(&x);
        let _ = tea.update(&x);
        let result = tea.update(&x);

        assert!(result.is_some());
    }

    #[test]
    fn test_vector_epsilon() {
        let mut ve = VectorEpsilon::new(5);
        let x = DVector::from_element(10, 1.0);

        let _ = ve.update(&x);
        let _ = ve.update(&x);
        let result = ve.update(&x);

        assert!(result.is_some());
    }

    #[test]
    fn test_rre2() {
        let mut rre2 = RRE2::new(5);
        let x = DVector::from_element(10, 1.0);
        let u = DVector::from_element(10, 0.5);

        let _ = rre2.update(&x, &u);
        let _ = rre2.update(&x, &u);
        let result = rre2.update(&x, &u);

        assert!(result.is_some());
    }

    #[test]
    fn test_mpe_variant() {
        let mut mpe = MPEVariant::new(5);
        let x = DVector::from_element(10, 1.0);

        let _ = mpe.update(&x);
        let _ = mpe.update(&x);
        let result = mpe.update(&x);

        assert!(result.is_some());
    }

    #[test]
    fn test_hybrid() {
        let mut hybrid = HybridAccelerator::new();

        let x = DVector::from_element(10, 1.0);
        let g = DVector::from_element(10, 0.5);

        let result = hybrid.update(&x, &g);
        assert!(result.iter().all(|v| v.is_finite()));
    }
}

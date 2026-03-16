//! Tensor product and multilevel acceleration methods.
//!
//! This module provides:
//! - Tensor product preconditioners
//! - Multilevel acceleration techniques
//! - Hierarchical basis methods
//! - Wavelet-based acceleration

use nalgebra::{DMatrix, DVector};

/// Tensor product preconditioner for structured grids.
#[derive(Debug, Clone)]
pub struct TensorProductPreconditioner {
    /// 1D preconditioner matrices for each dimension.
    pub dim_preconditioners: Vec<DMatrix<f64>>,
    /// Number of dimensions.
    pub num_dims: usize,
}

impl TensorProductPreconditioner {
    /// Creates a tensor product preconditioner from 1D operators.
    pub fn new(dim_operators: Vec<DMatrix<f64>>) -> Self {
        let num_dims = dim_operators.len();
        Self {
            dim_preconditioners: dim_operators,
            num_dims,
        }
    }

    /// Creates from Laplacian stencil sizes.
    pub fn from_laplacian_sizes(sizes: &[usize]) -> Self {
        let mut dim_ops = Vec::with_capacity(sizes.len());

        for &n in sizes {
            let mut op = DMatrix::zeros(n, n);
            for i in 0..n {
                op[(i, i)] = 2.0;
                if i > 0 {
                    op[(i, i - 1)] = -1.0;
                }
                if i < n - 1 {
                    op[(i, i + 1)] = -1.0;
                }
            }
            dim_ops.push(op);
        }

        Self::new(dim_ops)
    }

    /// Applies tensor product preconditioner.
    pub fn apply(&self, r: &DVector<f64>, grid_sizes: &[usize]) -> DVector<f64> {
        match self.num_dims {
            1 => {
                let lu = self.dim_preconditioners[0].clone().lu();
                lu.solve(r).unwrap_or_else(|| r.clone())
            }
            2 => self.apply_2d(r, grid_sizes),
            3 => self.apply_3d(r, grid_sizes),
            _ => r.clone(), // Fallback for higher dimensions
        }
    }

    fn apply_2d(&self, r: &DVector<f64>, sizes: &[usize]) -> DVector<f64> {
        let nx = sizes[0];
        let ny = sizes[1];
        let n = nx * ny;

        if r.len() != n {
            return r.clone();
        }

        // Reshape to 2D, apply preconditioner in each direction
        let mut u = DMatrix::zeros(ny, nx);
        for i in 0..ny {
            for j in 0..nx {
                u[(i, j)] = r[i * nx + j];
            }
        }

        // Apply in x-direction
        let lu_x = self.dim_preconditioners[0].clone().lu();
        for i in 0..ny {
            let row = u.row(i).into_owned();
            let solved = lu_x.solve(&row).unwrap_or(row.clone());
            for j in 0..nx {
                u[(i, j)] = solved[j];
            }
        }

        // Apply in y-direction
        let lu_y = self.dim_preconditioners[1].clone().lu();
        for j in 0..nx {
            let col = u.column(j).into_owned();
            let solved = lu_y.solve(&col).unwrap_or(col.clone());
            for i in 0..ny {
                u[(i, j)] = solved[i];
            }
        }

        // Flatten back
        DVector::from_fn(n, |i, _| u[(i / nx, i % nx)])
    }

    fn apply_3d(&self, r: &DVector<f64>, sizes: &[usize]) -> DVector<f64> {
        let nx = sizes[0];
        let ny = sizes[1];
        let nz = sizes[2];
        let n = nx * ny * nz;

        if r.len() != n {
            return r.clone();
        }

        // Simplified: apply 1D preconditioners sequentially
        let mut result = r.clone();

        for dim in 0..self.num_dims.min(3) {
            let lu = self.dim_preconditioners[dim].clone().lu();
            let mut temp = result.clone();
            // Apply along this dimension
            if dim == 0 {
                for k in 0..nz {
                    for j in 0..ny {
                        let line = DVector::from_fn(nx, |i, _| {
                            result[k * nx * ny + j * nx + i]
                        });
                        if let Some(solved) = lu.solve(&line) {
                            for i in 0..nx {
                                temp[k * nx * ny + j * nx + i] = solved[i];
                            }
                        }
                    }
                }
            }
            result = temp;
        }

        result
    }
}

/// Hierarchical basis transformation for multilevel methods.
#[derive(Debug, Clone)]
pub struct HierarchicalBasis {
    /// Level information.
    pub levels: Vec<LevelInfo>,
    /// Transfer operators.
    pub prolongation: Vec<DMatrix<f64>>,
    pub restriction: Vec<DMatrix<f64>>,
}

#[derive(Debug, Clone)]
pub struct LevelInfo {
    pub num_nodes: usize,
    pub level: usize,
}

impl HierarchicalBasis {
    /// Creates a hierarchical basis for 1D grid.
    pub fn new_1d(num_levels: usize, fine_size: usize) -> Self {
        let mut levels = Vec::with_capacity(num_levels);
        let mut prolongation = Vec::with_capacity(num_levels - 1);
        let mut restriction = Vec::with_capacity(num_levels - 1);

        let mut size = fine_size;
        for level in 0..num_levels {
            levels.push(LevelInfo { num_nodes: size, level });

            if level < num_levels - 1 {
                let coarse_size = (size + 1) / 2;

                // Simple linear interpolation
                let mut p = DMatrix::zeros(size, coarse_size);
                for i in 0..coarse_size {
                    if i * 2 < size {
                        p[(i * 2, i)] = 1.0;
                    }
                    if i * 2 + 1 < size {
                        p[(i * 2 + 1, i)] = 0.5;
                    }
                }

                // Restriction is transpose of prolongation (scaled)
                let r = p.transpose() * (1.0 / 2.0);

                prolongation.push(p);
                restriction.push(r);

                size = coarse_size;
            }
        }

        Self {
            levels,
            prolongation,
            restriction,
        }
    }

    /// Returns the number of levels.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Restricts a vector from fine to coarse level.
    pub fn restrict(&self, v: &DVector<f64>, from_level: usize) -> DVector<f64> {
        if from_level >= self.restriction.len() {
            return v.clone();
        }
        &self.restriction[from_level] * v
    }

    /// Prolongates a vector from coarse to fine level.
    pub fn prolongate(&self, v: &DVector<f64>, to_level: usize) -> DVector<f64> {
        if to_level >= self.prolongation.len() {
            return v.clone();
        }
        &self.prolongation[to_level] * v
    }

    /// Computes hierarchical surplus.
    pub fn compute_surplus(&self, v: &DVector<f64>, level: usize) -> DVector<f64> {
        if level == 0 || level >= self.restriction.len() {
            return v.clone();
        }

        // Coarsen then interpolate back
        let coarse = self.restrict(v, level - 1);
        let interpolated = self.prolongate(&coarse, level - 1);

        // Surplus is the difference
        v - interpolated
    }
}

/// Wavelet-based acceleration for sparse systems.
#[derive(Debug, Clone)]
pub struct WaveletPreconditioner {
    /// Wavelet level.
    pub level: usize,
    /// Transformation matrix.
    pub wavelet_transform: DMatrix<f64>,
}

impl WaveletPreconditioner {
    /// Creates a Haar wavelet preconditioner.
    pub fn haar(size: usize) -> Self {
        let n = size.next_power_of_two();
        let mut w = DMatrix::identity(n, n);

        // Build Haar wavelet transform
        let mut temp = DMatrix::identity(n, n);

        let mut block_size = n;
        while block_size >= 2 {
            let half = block_size / 2;
            for i in 0..half {
                // Averaging
                for j in 0..n {
                    w[(i, j)] = (temp[(i * 2, j)] + temp[(i * 2 + 1, j)]) / 2.0_f64.sqrt();
                    w[(half + i, j)] = (temp[(i * 2, j)] - temp[(i * 2 + 1, j)])
                        / 2.0_f64.sqrt();
                }
            }
            block_size = half;
            temp = w.clone();
        }

        Self {
            level: n.ilog2() as usize,
            wavelet_transform: w,
        }
    }

    /// Applies wavelet preconditioner.
    pub fn apply(&self, r: &DVector<f64>) -> DVector<f64> {
        // Transform to wavelet space
        let r_waved = &self.wavelet_transform * r;

        // Scale by approximate eigenvalues (diagonal preconditioning in wavelet space)
        let n = r_waved.len();
        let mut scaled = r_waved.clone();

        for i in 0..n {
            let scale = if i < n / 2 {
                1.0 // Low frequencies
            } else {
                2.0 // High frequencies - damp
            };
            scaled[i] /= scale;
        }

        // Transform back
        self.wavelet_transform.transpose() * &scaled
    }

    /// Applies wavelet transform only.
    pub fn transform(&self, v: &DVector<f64>) -> DVector<f64> {
        &self.wavelet_transform * v
    }

    /// Applies inverse wavelet transform.
    pub fn inverse_transform(&self, v: &DVector<f64>) -> DVector<f64> {
        self.wavelet_transform.transpose() * v
    }
}

/// Multilevel acceleration coordinator.
#[derive(Debug, Clone)]
pub struct MultilevelAccelerator {
    /// Number of levels.
    pub num_levels: usize,
    /// Coarse grid correction factor.
    pub coarse_factor: f64,
    /// Smoothing iterations per level.
    pub smoothing_iterations: Vec<usize>,
}

impl MultilevelAccelerator {
    /// Creates a new multilevel accelerator.
    pub fn new(num_levels: usize) -> Self {
        Self {
            num_levels,
            coarse_factor: 0.5,
            smoothing_iterations: vec![2; num_levels],
        }
    }

    /// Sets smoothing iterations per level.
    pub fn with_smoothing(mut self, iterations: Vec<usize>) -> Self {
        self.smoothing_iterations = iterations;
        self
    }

    /// Performs one V-cycle.
    pub fn vcycle(
        &self,
        level: usize,
        residual: &DVector<f64>,
        a_matrices: &[DMatrix<f64>],
        hb: &HierarchicalBasis,
    ) -> DVector<f64> {
        let n = residual.len();

        // Bottom level: solve directly
        if level >= self.num_levels - 1 || n < 4 {
            let lu = a_matrices[level].clone().lu();
            if let Some(sol) = lu.solve(residual) {
                return sol;
            }
            // Diagonal solve fallback
            let mut corr = DVector::zeros(n);
            for i in 0..n {
                let a_ii = a_matrices[level][(i, i)].abs().max(1e-15);
                corr[i] = residual[i] / a_ii;
            }
            return corr;
        }

        // Pre-smoothing
        let mut x = DVector::zeros(n);
        for _ in 0..self.smoothing_iterations[level] {
            for i in 0..n {
                let a_ii = a_matrices[level][(i, i)].abs().max(1e-15);
                x[i] += residual[i] / a_ii;
            }
        }

        // Compute residual and restrict
        let ax = &a_matrices[level] * &x;
        let fine_residual = residual - ax;
        let coarse_residual = hb.restrict(&fine_residual, level);

        // Coarse grid solve (recursive)
        let coarse_correction = self.vcycle(
            level + 1,
            &coarse_residual,
            a_matrices,
            hb,
        );

        // Prolongate and correct
        let fine_correction = hb.prolongate(&coarse_correction, level);
        x += fine_correction;

        // Post-smoothing
        for _ in 0..self.smoothing_iterations[level] {
            for i in 0..n {
                let a_ii = a_matrices[level][(i, i)].abs().max(1e-15);
                x[i] += residual[i] / a_ii;
            }
        }

        x
    }

    /// Performs a W-cycle (two coarse grid solves).
    pub fn wcycle(
        &self,
        level: usize,
        residual: &DVector<f64>,
        a_matrices: &[DMatrix<f64>],
        hb: &HierarchicalBasis,
    ) -> DVector<f64> {
        let n = residual.len();

        if level >= self.num_levels - 1 || n < 4 {
            let lu = a_matrices[level].clone().lu();
            if let Some(sol) = lu.solve(residual) {
                return sol;
            }
            let mut corr = DVector::zeros(n);
            for i in 0..n {
                let a_ii = a_matrices[level][(i, i)].abs().max(1e-15);
                corr[i] = residual[i] / a_ii;
            }
            return corr;
        }

        // Pre-smoothing
        let mut x = DVector::zeros(n);
        for _ in 0..self.smoothing_iterations[level] {
            for i in 0..n {
                let a_ii = a_matrices[level][(i, i)].abs().max(1e-15);
                x[i] += residual[i] / a_ii;
            }
        }

        // First coarse grid solve
        let ax = &a_matrices[level] * &x;
        let fine_residual = residual - ax;
        let coarse_residual = hb.restrict(&fine_residual, level);
        let coarse_correction1 = self.wcycle(level + 1, &coarse_residual, a_matrices, hb);

        // Update and compute new residual
        let fine_correction1 = hb.prolongate(&coarse_correction1, level);
        x += fine_correction1;

        // Second coarse grid solve
        let ax = &a_matrices[level] * &x;
        let fine_residual = residual - ax;
        let coarse_residual = hb.restrict(&fine_residual, level);
        let coarse_correction2 = self.wcycle(level + 1, &coarse_residual, a_matrices, hb);

        let fine_correction2 = hb.prolongate(&coarse_correction2, level);
        x += fine_correction2;

        // Post-smoothing
        for _ in 0..self.smoothing_iterations[level] {
            for i in 0..n {
                let a_ii = a_matrices[level][(i, i)].abs().max(1e-15);
                x[i] += residual[i] / a_ii;
            }
        }

        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_product_preconditioner() {
        // 1D case
        let sizes = vec![10];
        let tp = TensorProductPreconditioner::from_laplacian_sizes(&sizes);

        let n = sizes[0];
        let r = DVector::from_element(n, 1.0);
        let z = tp.apply(&r, &sizes);

        assert_eq!(z.len(), n);
        assert!(z.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_hierarchical_basis() {
        let hb = HierarchicalBasis::new_1d(3, 8);

        assert_eq!(hb.num_levels(), 3);

        let v = DVector::from_fn(8, |i, _| i as f64);

        // Test restriction
        let v_coarse = hb.restrict(&v, 0);
        assert_eq!(v_coarse.len(), 4);

        // Test prolongation
        let v_fine = hb.prolongate(&v_coarse, 0);
        assert_eq!(v_fine.len(), 8);

        // Test surplus
        let surplus = hb.compute_surplus(&v, 1);
        assert_eq!(surplus.len(), 8);
        assert!(surplus.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_wavelet_preconditioner() {
        let wp = WaveletPreconditioner::haar(8);

        let v = DVector::from_fn(8, |i, _| (i as f64).sin());

        let transformed = wp.transform(&v);
        let back = wp.inverse_transform(&transformed);

        // Check reconstruction (approximately)
        let error = (v.clone() - back).norm();
        assert!(error < 1e-10);

        let preconditioned = wp.apply(&v);
        assert!(preconditioned.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_multilevel_accelerator() {
        let mla = MultilevelAccelerator::new(3);

        // Create hierarchy of matrices (simplified 1D Laplacian)
        let mut a_matrices = Vec::new();
        let mut size = 8;
        for _ in 0..3 {
            let mut a = DMatrix::zeros(size, size);
            for i in 0..size {
                a[(i, i)] = 2.0;
                if i > 0 {
                    a[(i, i - 1)] = -1.0;
                }
                if i < size - 1 {
                    a[(i, i + 1)] = -1.0;
                }
            }
            a_matrices.push(a);
            size = (size + 1) / 2;
        }

        let hb = HierarchicalBasis::new_1d(3, 8);
        let r = DVector::from_element(8, 1.0);

        let correction = mla.vcycle(0, &r, &a_matrices, &hb);

        assert_eq!(correction.len(), 8);
        assert!(correction.iter().all(|v| v.is_finite()));
    }
}

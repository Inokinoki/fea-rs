//! Advanced preconditioning techniques for iterative solvers.
#![allow(non_snake_case)]
#![allow(unused_assignments)]
//!
//! This module provides state-of-the-art preconditioners:
//! - Factorized Sparse Approximate Inverse (FSAI)
//! - Physics-based preconditioners for FEA
//! - Multilevel additive Schwarz
//! - Block recursive preconditioners

use nalgebra::{DMatrix, DVector};

/// Factorized Sparse Approximate Inverse (FSAI) preconditioner.
///
/// FSAI computes a lower triangular matrix L such that L*L^T ≈ A^{-1}.
#[derive(Debug, Clone)]
pub struct FSAIPreconditioner {
    /// Lower triangular factor.
    pub l_matrix: DMatrix<f64>,
    /// Pattern of non-zero elements.
    pub pattern: Vec<Vec<usize>>,
}

impl FSAIPreconditioner {
    /// Creates a new FSAI preconditioner.
    pub fn new(a: &DMatrix<f64>, pattern: Vec<Vec<usize>>) -> Option<Self> {
        let n = a.nrows();
        let mut l = DMatrix::zeros(n, n);

        for i in 0..n {
            let mut sum_diag = 1.0_f64;
            for &k in &pattern[i] {
                if k < i {
                    let l_ik = l[(i, k)];
                    sum_diag -= l_ik * l_ik;
                }
            }

            if sum_diag <= 0.0_f64 {
                return None;
            }

            l[(i, i)] = sum_diag.sqrt();

            for &j in &pattern[i] {
                if j < i {
                    let mut sum = 0.0_f64;
                    for &k in &pattern[i] {
                        if k < j {
                            sum += l[(i, k)] * l[(j, k)];
                        }
                    }
                    let a_ij = a[(i, j)];
                    l[(i, j)] = (a_ij - sum) / l[(j, j)];
                }
            }
        }

        Some(Self { l_matrix: l, pattern })
    }

    /// Creates FSAI with automatic pattern detection.
    pub fn with_auto_pattern(a: &DMatrix<f64>, fill_level: usize) -> Option<Self> {
        let n = a.nrows();
        let mut pattern = Vec::with_capacity(n);

        for i in 0..n {
            let mut row_pattern = Vec::new();
            for j in 0..=i {
                if a[(i, j)].abs() > 1e-15 {
                    row_pattern.push(j);
                }
            }
            if fill_level > 1 {
                for &j in &row_pattern.clone() {
                    for k in (j + 1)..i {
                        if a[(j, k)].abs() > 1e-15 && !row_pattern.contains(&k) {
                            row_pattern.push(k);
                        }
                    }
                }
            }
            row_pattern.sort();
            pattern.push(row_pattern);
        }

        Self::new(a, pattern)
    }

    /// Applies the preconditioner: z = L^T * L * r
    pub fn apply(&self, r: &DVector<f64>) -> DVector<f64> {
        let y = self.forward_solve(r);
        self.transpose_solve(&y)
    }

    fn forward_solve(&self, r: &DVector<f64>) -> DVector<f64> {
        let n = r.len();
        let mut y = r.clone();

        for i in 0..n {
            let l_ii = self.l_matrix[(i, i)];
            if l_ii.abs() > 1e-15 {
                y[i] /= l_ii;
            }
            for &j in &self.pattern[i] {
                if j < i {
                    let l_ij = self.l_matrix[(i, j)];
                    y[j] -= l_ij * y[i];
                }
            }
        }
        y
    }

    fn transpose_solve(&self, y: &DVector<f64>) -> DVector<f64> {
        let n = y.len();
        let mut z = y.clone();

        for i in (0..n).rev() {
            let l_ii = self.l_matrix[(i, i)];
            if l_ii.abs() > 1e-15 {
                z[i] /= l_ii;
            }
            for &j in &self.pattern[i] {
                if j < i {
                    let l_ij = self.l_matrix[(i, j)];
                    z[i] -= l_ij * z[j];
                }
            }
        }
        z
    }
}

/// Physics-based preconditioner for elasticity problems.
#[derive(Debug, Clone)]
pub struct ElasticityPreconditioner {
    pub material_scaling: DVector<f64>,
    pub geometric_scaling: DVector<f64>,
    pub preconditioner_diag: DVector<f64>,
}

impl ElasticityPreconditioner {
    /// Creates a new elasticity preconditioner.
    pub fn new(k: &DMatrix<f64>, youngs_modulus: f64) -> Self {
        let n = k.nrows();
        let mut material_scaling = DVector::zeros(n);
        let mut geometric_scaling = DVector::zeros(n);
        let mut preconditioner_diag = DVector::zeros(n);

        for i in 0..n {
            geometric_scaling[i] = k[(i, i)].abs().max(1e-15);
            material_scaling[i] = 1.0 / youngs_modulus.sqrt();
            preconditioner_diag[i] = (geometric_scaling[i] * youngs_modulus).sqrt();
            if preconditioner_diag[i] > 1e-15 {
                preconditioner_diag[i] = 1.0 / preconditioner_diag[i];
            }
        }

        Self {
            material_scaling,
            geometric_scaling,
            preconditioner_diag,
        }
    }

    /// Applies the preconditioner (diagonal scaling).
    pub fn apply(&self, r: &DVector<f64>) -> DVector<f64> {
        r.component_mul(&self.preconditioner_diag)
    }

    /// Creates preconditioner for heterogeneous materials.
    pub fn heterogeneous(k: &DMatrix<f64>, element_stiffness: &[f64], dof_per_node: usize) -> Self {
        let n = k.nrows();
        let num_nodes = n / dof_per_node;
        let mut preconditioner_diag = DVector::zeros(n);

        for node in 0..num_nodes {
            let mut avg_stiffness = 0.0_f64;
            for dof in 0..dof_per_node {
                let idx = node * dof_per_node + dof;
                if idx < n {
                    avg_stiffness += k[(idx, idx)].abs();
                }
            }
            avg_stiffness /= dof_per_node as f64;

            for dof in 0..dof_per_node {
                let idx = node * dof_per_node + dof;
                if idx < n {
                    preconditioner_diag[idx] = if avg_stiffness > 1e-15 {
                        1.0 / avg_stiffness.sqrt()
                    } else {
                        1.0
                    };
                }
            }
        }

        Self {
            material_scaling: DVector::from_element(n, 1.0),
            geometric_scaling: DVector::from_element(n, 1.0),
            preconditioner_diag,
        }
    }
}

/// Multilevel additive Schwarz preconditioner.
#[derive(Debug, Clone)]
pub struct AdditiveSchwarzMultilevel {
    pub levels: usize,
    pub restrictions: Vec<DMatrix<f64>>,
    pub coarse_matrices: Vec<DMatrix<f64>>,
    pub coarse_grid_size: Vec<usize>,
    pub smoothing_iterations: Vec<usize>,
}

impl AdditiveSchwarzMultilevel {
    /// Creates a new multilevel additive Schwarz preconditioner.
    pub fn new(fine_matrix: &DMatrix<f64>, num_levels: usize) -> Self {
        let mut restrictions = Vec::with_capacity(num_levels - 1);
        let mut coarse_matrices = Vec::with_capacity(num_levels);
        let mut coarse_grid_size = Vec::with_capacity(num_levels);
        let mut smoothing_iterations = Vec::with_capacity(num_levels);

        coarse_matrices.push(fine_matrix.clone());
        coarse_grid_size.push(fine_matrix.nrows());
        smoothing_iterations.push(2);

        let mut current_a = fine_matrix.clone();
        for _level in 0..num_levels - 1 {
            let n = current_a.nrows();
            let coarse_n = (n + 1) / 2;

            let mut r = DMatrix::zeros(coarse_n, n);
            for i in 0..coarse_n {
                if i * 2 < n {
                    r[(i, i * 2)] = 1.0;
                }
            }

            let coarse_a = &r * &current_a * &r.transpose();

            restrictions.push(r);
            coarse_matrices.push(coarse_a);
            coarse_grid_size.push(coarse_n);
            smoothing_iterations.push(2);

            current_a = coarse_matrices.last().unwrap().clone();
        }

        Self {
            levels: num_levels,
            restrictions,
            coarse_matrices,
            coarse_grid_size,
            smoothing_iterations,
        }
    }

    /// Applies the multilevel preconditioner (V-cycle).
    pub fn apply_vcycle(&self, r: &DVector<f64>) -> DVector<f64> {
        self.vcycle(0, r, 0)
    }

    fn vcycle(&self, level: usize, residual: &DVector<f64>, _depth: usize) -> DVector<f64> {
        let n = residual.len();

        if level == self.levels - 1 || n < 10 {
            let mut corr = DVector::zeros(n);
            for i in 0..n {
                let a_ii = self.coarse_matrices[level][(i, i)].abs().max(1e-15);
                corr[i] = residual[i] / a_ii;
            }
            return corr;
        }

        let mut x = DVector::zeros(n);
        let num_smooth = self.smoothing_iterations[level];

        for _ in 0..num_smooth {
            for i in 0..n {
                let a_ii = self.coarse_matrices[level][(i, i)].abs().max(1e-15);
                x[i] += residual[i] / a_ii;
            }
        }

        let ax = &self.coarse_matrices[level] * &x;
        let fine_residual = residual - ax;

        let restricted = if level < self.restrictions.len() {
            &self.restrictions[level] * &fine_residual
        } else {
            fine_residual.clone()
        };

        let coarse_corr = self.vcycle(level + 1, &restricted, _depth + 1);

        if level < self.restrictions.len() {
            let coarse_n = coarse_corr.len();
            let mut prolongated = DVector::zeros(n);
            for i in 0..coarse_n {
                if i * 2 < n {
                    prolongated[i * 2] += coarse_corr[i];
                }
                if i * 2 + 1 < n {
                    prolongated[i * 2 + 1] += coarse_corr[i];
                }
            }
            x += prolongated;
        }

        for _ in 0..num_smooth {
            for i in 0..n {
                let a_ii = self.coarse_matrices[level][(i, i)].abs().max(1e-15);
                x[i] += residual[i] / a_ii;
            }
        }

        x
    }
}

/// Block recursive preconditioner for block systems.
#[derive(Debug, Clone)]
pub struct BlockRecursivePreconditioner {
    pub block_diag: Vec<DMatrix<f64>>,
    pub coupling_strength: f64,
}

impl BlockRecursivePreconditioner {
    /// Creates a block recursive preconditioner.
    pub fn new(a: &DMatrix<f64>, block_size: usize) -> Self {
        let n = a.nrows();
        let num_blocks = (n + block_size - 1) / block_size;

        let mut block_diag = Vec::with_capacity(num_blocks);

        for b in 0..num_blocks {
            let start = b * block_size;
            let end = (start + block_size).min(n);
            let bs = end - start;

            let mut block = DMatrix::zeros(bs, bs);
            for i in 0..bs {
                for j in 0..bs {
                    block[(i, j)] = a[(start + i, start + j)];
                }
            }

            block_diag.push(block);
        }

        let mut off_diag_norm = 0.0_f64;
        let mut diag_norm = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                if (i / block_size) != (j / block_size) {
                    off_diag_norm += a[(i, j)].abs();
                } else {
                    diag_norm += a[(i, j)].abs();
                }
            }
        }
        let coupling = if diag_norm > 1e-15 {
            off_diag_norm / diag_norm
        } else {
            1.0
        };

        Self {
            block_diag,
            coupling_strength: coupling,
        }
    }

    /// Applies the block preconditioner.
    pub fn apply(&self, r: &[f64]) -> Vec<f64> {
        let n = r.len();
        let mut z = vec![0.0; n];

        for (b, block) in self.block_diag.iter().enumerate() {
            let start = b * block.nrows();
            let end = start + block.nrows();
            let r_block = DVector::from_column_slice(&r[start..end]);

            let lu = block.clone().lu();
            let z_block = lu.solve(&r_block).unwrap_or(r_block);
            for (i, zi) in z_block.iter().enumerate() {
                z[start + i] = *zi;
            }
        }

        z
    }

    /// Returns coupling strength estimate.
    pub fn coupling_strength(&self) -> f64 {
        self.coupling_strength
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fsai_preconditioner() {
        let a = DMatrix::from_row_slice(4, 4, &[
            4.0, -1.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0,
            0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, -1.0, 4.0,
        ]);

        let pattern = vec![vec![0], vec![0, 1], vec![1, 2], vec![2, 3]];
        let fsai = FSAIPreconditioner::new(&a, pattern);
        assert!(fsai.is_some());

        let fsai = fsai.unwrap();
        let r = DVector::from_column_slice(&[1.0, 2.0, 3.0, 4.0]);
        let z = fsai.apply(&r);

        assert_eq!(z.len(), 4);
        assert!(z.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_elasticity_preconditioner() {
        let a = DMatrix::from_row_slice(6, 6, &[
            10.0, -2.0, 0.0, 0.0, 0.0, 0.0,
            -2.0, 10.0, -2.0, 0.0, 0.0, 0.0,
            0.0, -2.0, 10.0, -2.0, 0.0, 0.0,
            0.0, 0.0, -2.0, 10.0, -2.0, 0.0,
            0.0, 0.0, 0.0, -2.0, 10.0, -2.0,
            0.0, 0.0, 0.0, 0.0, -2.0, 10.0,
        ]);

        let prec = ElasticityPreconditioner::new(&a, 200e9);
        let r = DVector::from_element(6, 1.0);
        let z = prec.apply(&r);

        assert_eq!(z.len(), 6);
        assert!(z.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_block_recursive_preconditioner() {
        let a = DMatrix::from_row_slice(6, 6, &[
            4.0, -1.0, -0.5, 0.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, -0.5, 0.0, 0.0,
            -0.5, -1.0, 4.0, -1.0, -0.5, 0.0,
            0.0, -0.5, -1.0, 4.0, -1.0, 0.0,
            0.0, 0.0, -0.5, -1.0, 4.0, -1.0,
            0.0, 0.0, 0.0, 0.0, -1.0, 4.0,
        ]);

        let prec = BlockRecursivePreconditioner::new(&a, 3);
        let r: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let z = prec.apply(&r);

        assert_eq!(z.len(), 6);
        assert!(z.iter().all(|v| v.is_finite()));
        assert!(prec.coupling_strength() >= 0.0);
    }

    #[test]
    fn test_additive_schwarz_multilevel() {
        let a = DMatrix::from_row_slice(8, 8, &[
            4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0,
        ]);

        let asm = AdditiveSchwarzMultilevel::new(&a, 3);
        let r = DVector::from_column_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let corr = asm.apply_vcycle(&r);

        assert_eq!(corr.len(), 8);
        assert!(corr.iter().all(|v| v.is_finite()));
    }
}

//! Block-Jacobi preconditioner with multiple block sizes.
//!
//! Block-Jacobi is more effective than point-Jacobi for systems
//! with block structure, such as those from coupled physics problems.

use nalgebra::{DMatrix, DVector};

/// Block-Jacobi preconditioner.
pub struct BlockJacobi {
    /// Block diagonal inverse matrices.
    blocks: Vec<DMatrix<f64>>,
    /// Mapping from global index to block index.
    block_mapping: Vec<usize>,
}

impl BlockJacobi {
    /// Creates a new Block-Jacobi preconditioner.
    ///
    /// # Arguments
    /// * `a` - The system matrix
    /// * `block_size` - Size of each block
    pub fn new(a: &DMatrix<f64>, block_size: usize) -> Option<Self> {
        let n = a.nrows();
        let num_blocks = (n + block_size - 1) / block_size;
        let mut blocks = Vec::with_capacity(num_blocks);
        let mut block_mapping = vec![0; n];

        for b in 0..num_blocks {
            let start = b * block_size;
            let end = (start + block_size).min(n);
            let bs = end - start;

            // Extract block
            let mut block = DMatrix::zeros(bs, bs);
            for i in 0..bs {
                for j in 0..bs {
                    block[(i, j)] = a[(start + i, start + j)];
                }
                block_mapping[start + i] = b;
            }

            // Invert block
            let block_inv = block.try_inverse()?;
            blocks.push(block_inv);
        }

        Some(Self { blocks, block_mapping })
    }

    /// Applies the preconditioner: z = M^{-1} * r.
    pub fn apply(&self, r: &DVector<f64>) -> DVector<f64> {
        let n = r.len();
        let mut z = DVector::zeros(n);

        for (b, block_inv) in self.blocks.iter().enumerate() {
            let start = b * block_inv.nrows();
            let end = (start + block_inv.nrows()).min(n);
            let bs = end - start;

            let r_block = r.rows(start, bs);
            let z_block = block_inv * r_block;

            z.rows_mut(start, bs).copy_from(&z_block);
        }

        z
    }
}

/// Additive Schwarz preconditioner.
///
/// Domain decomposition preconditioner that solves local problems
/// on overlapping subdomains.
pub struct AdditiveSchwarz {
    /// Local solve matrices.
    local_solves: Vec<DMatrix<f64>>,
    /// Subdomain index mappings.
    subdomain_indices: Vec<Vec<usize>>,
}

impl AdditiveSchwarz {
    /// Creates Additive Schwarz preconditioner.
    ///
    /// # Arguments
    /// * `a` - The system matrix
    /// * `overlap` - Number of overlapping elements between subdomains
    pub fn new(a: &DMatrix<f64>, num_subdomains: usize, overlap: usize) -> Option<Self> {
        let n = a.nrows();
        let base_size = n / num_subdomains;

        let mut local_solves = Vec::with_capacity(num_subdomains);
        let mut subdomain_indices = Vec::with_capacity(num_subdomains);

        for s in 0..num_subdomains {
            // Define subdomain with overlap
            let start = s.saturating_sub(overlap) * base_size;
            let end = ((s + 1 + overlap) * base_size).min(n);

            let indices: Vec<usize> = (start..end).collect();
            let size = indices.len();

            // Extract and invert local matrix
            let mut local = DMatrix::zeros(size, size);
            for (i, &gi) in indices.iter().enumerate() {
                for (j, &gj) in indices.iter().enumerate() {
                    local[(i, j)] = a[(gi, gj)];
                }
            }

            let local_inv = local.try_inverse()?;
            local_solves.push(local_inv);
            subdomain_indices.push(indices);
        }

        Some(Self { local_solves, subdomain_indices })
    }

    /// Applies the preconditioner.
    pub fn apply(&self, r: &DVector<f64>) -> DVector<f64> {
        let n = r.len();
        let mut z = DVector::zeros(n);

        for (s, (local_inv, indices)) in self.local_solves.iter().zip(&self.subdomain_indices).enumerate() {
            // Extract residual on subdomain
            let r_local = DVector::from_fn(indices.len(), |i, _| r[indices[i]]);

            // Local solve
            let z_local = local_inv * &r_local;

            // Accumulate (additive)
            for (i, &gi) in indices.iter().enumerate() {
                z[gi] += z_local[i];
            }
        }

        z
    }
}

/// Sparse approximate inverse (SPAI) preconditioner.
///
/// Constructs a sparse matrix M such that ||I - AM|| is minimized.
pub struct SPAI {
    /// Approximate inverse matrix.
    minv: DMatrix<f64>,
}

impl SPAI {
    /// Creates SPAI preconditioner (simplified Frobenius norm minimization).
    pub fn new(a: &DMatrix<f64>, drop_tol: f64) -> Self {
        let n = a.nrows();
        let mut minv = DMatrix::zeros(n, n);

        // For each column, solve min ||Ae_i - m_i|| where m_i is sparse
        for i in 0..n {
            // Use Jacobi as initial approximation
            let a_ii = a[(i, i)];
            if a_ii.abs() > drop_tol {
                minv[(i, i)] = 1.0 / a_ii;
            }

            // Simple improvement: minimize ||A*m_i - e_i|| for tridiagonal pattern
            if i > 0 && i < n - 1 {
                // Solve 3x3 local system
                let mut local_a = DMatrix::zeros(3, 3);
                let mut local_b = DVector::zeros(3);

                for ii in 0..3 {
                    for jj in 0..3 {
                        local_a[(ii, jj)] = a[(i - 1 + ii, i - 1 + jj)];
                    }
                    local_b[ii] = if ii == 1 { 1.0 } else { 0.0 };
                }

                if let Some(local_x) = local_a.lu().solve(&local_b) {
                    for ii in 0..3 {
                        let val = local_x[ii];
                        if val.abs() > drop_tol {
                            minv[(i - 1 + ii, i)] = val;
                        }
                    }
                }
            }
        }

        Self { minv }
    }

    /// Applies the preconditioner.
    pub fn apply(&self, r: &DVector<f64>) -> DVector<f64> {
        self.minv.clone() * r
    }
}

/// AINV (Approximate Inverse) factorization preconditioner.
///
/// Computes sparse factors W and Z such that WAZ ≈ D (diagonal).
pub struct AINV {
    /// Inverse factors.
    w: DMatrix<f64>,
    z: DMatrix<f64>,
    /// Diagonal scaling.
    d: Vec<f64>,
}

impl AINV {
    /// Creates AINV preconditioner (simplified).
    pub fn new(a: &DMatrix<f64>, drop_tol: f64) -> Self {
        let n = a.nrows();

        // Initialize with identity
        let mut w = DMatrix::identity(n, n);
        let mut z = DMatrix::identity(n, n);
        let mut d = vec![1.0; n];

        // Simplified AINV: just use diagonal scaling
        for i in 0..n {
            let a_ii = a[(i, i)];
            if a_ii.abs() > drop_tol {
                d[i] = a_ii;
                w[(i, i)] = 1.0 / a_ii.sqrt();
                z[(i, i)] = 1.0 / a_ii.sqrt();
            }
        }

        Self { w, z, d }
    }

    /// Applies the preconditioner.
    pub fn apply(&self, r: &DVector<f64>) -> DVector<f64> {
        // M^{-1} = Z * D^{-1} * W
        let y = self.w.clone() * r;
        let z = DVector::from_fn(y.len(), |i, _| y[i] / self.d[i]);
        self.z.clone() * &z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_jacobi() {
        let a = DMatrix::from_row_slice(4, 4, &[
            4.0, 1.0, 0.0, 0.0,
            1.0, 4.0, 1.0, 0.0,
            0.0, 1.0, 4.0, 1.0,
            0.0, 0.0, 1.0, 4.0,
        ]);

        let bj = BlockJacobi::new(&a, 2).expect("Block-Jacobi creation failed");

        let r = DVector::from_column_slice(&[1.0, 1.0, 1.0, 1.0]);
        let z = bj.apply(&r);

        // Should reduce residual
        let r_norm = r.norm();
        let z_norm = z.norm();
        assert!(z_norm > 0.0);
        assert!(z_norm < r_norm * 2.0); // Should not amplify too much
    }

    #[test]
    fn test_additive_schwarz() {
        let a = DMatrix::from_row_slice(6, 6, &[
            4.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 4.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 4.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 4.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 4.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 4.0,
        ]);

        let aschwarz = AdditiveSchwarz::new(&a, 3, 1).expect("Additive Schwarz creation failed");

        let r = DVector::from_element(6, 1.0);
        let z = aschwarz.apply(&r);

        assert!(z.norm() > 0.0);
    }

    #[test]
    fn test_spai() {
        let a = DMatrix::from_row_slice(4, 4, &[
            10.0, 1.0, 0.0, 0.0,
            1.0, 10.0, 1.0, 0.0,
            0.0, 1.0, 10.0, 1.0,
            0.0, 0.0, 1.0, 10.0,
        ]);

        let spai = SPAI::new(&a, 1e-12);
        let r = DVector::from_column_slice(&[1.0, 1.0, 1.0, 1.0]);
        let z = spai.apply(&r);

        assert!(z.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_ainv() {
        let a = DMatrix::from_row_slice(4, 4, &[
            4.0, 1.0, 0.0, 0.0,
            1.0, 4.0, 1.0, 0.0,
            0.0, 1.0, 4.0, 1.0,
            0.0, 0.0, 1.0, 4.0,
        ]);

        let ainv = AINV::new(&a, 1e-12);
        let r = DVector::from_column_slice(&[1.0, 1.0, 1.0, 1.0]);
        let z = ainv.apply(&r);

        assert!(z.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_block_vs_point_jacobi() {
        // Block matrix system
        let mut a = DMatrix::zeros(6, 6);
        for i in 0..6 {
            a[(i, i)] = 10.0;
            if i > 0 {
                a[(i, i - 1)] = 2.0;
            }
            if i < 5 {
                a[(i, i + 1)] = 2.0;
            }
        }

        // Add 2x2 block structure
        a[(0, 1)] = 5.0;
        a[(1, 0)] = 5.0;
        a[(2, 3)] = 5.0;
        a[(3, 2)] = 5.0;

        let r = DVector::from_element(6, 1.0);

        // Point Jacobi
        let bj1 = BlockJacobi::new(&a, 1).expect("Block-Jacobi failed");
        let z1 = bj1.apply(&r);

        // Block Jacobi with 2x2 blocks
        let bj2 = BlockJacobi::new(&a, 2).expect("Block-Jacobi failed");
        let z2 = bj2.apply(&r);

        // Block version should give different (typically better) results
        assert_ne!(z1.norm(), z2.norm());
    }
}

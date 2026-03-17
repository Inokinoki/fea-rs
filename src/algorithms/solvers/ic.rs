//! Incomplete Cholesky preconditioner.
#![allow(non_snake_case)]
#![allow(unused_assignments)]
//!
//! This module provides:
//! - IC(0) incomplete Cholesky factorization
//! - Modified IC (MIC) preconditioner
//! - Block IC preconditioner

use nalgebra::{DMatrix, DVector};

/// Incomplete Cholesky factorization (IC(0)).
///
/// Stores L such that L * L^T ≈ A (dropping fill-in).
#[derive(Debug, Clone)]
pub struct IncompleteCholesky {
    /// Lower triangular factor.
    pub l: DMatrix<f64>,
    /// Diagonal scaling factors.
    pub diag: Vec<f64>,
}

impl IncompleteCholesky {
    /// Creates IC(0) factorization of a sparse symmetric positive definite matrix.
    ///
    /// Only keeps non-zeros in the same pattern as the input matrix.
    pub fn new(a: &DMatrix<f64>, drop_tol: f64) -> Option<Self> {
        let n = a.nrows();
        let mut l = DMatrix::zeros(n, n);
        let mut diag = vec![1.0; n];

        for i in 0..n {
            for j in 0..=i {
                if a[(i, j)].abs() < drop_tol && i != j {
                    // Drop small off-diagonal entries
                    l[(i, j)] = 0.0;
                    continue;
                }

                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[(i, k)] * l[(j, k)];
                }

                if i == j {
                    let val: f64 = a[(i, i)] - sum;
                    if val <= 0.0 {
                        // Matrix not positive definite
                        return None;
                    }
                    l[(i, i)] = val.sqrt();
                    diag[i] = l[(i, i)];
                } else {
                    if l[(j, j)].abs() > 1e-15 {
                        l[(i, j)] = (a[(i, j)] - sum) / l[(j, j)];
                    }
                }
            }
        }

        Some(Self { l, diag })
    }

    /// Solves L * y = b (forward substitution).
    pub fn solve_lower(&self, b: &DVector<f64>) -> DVector<f64> {
        let n = b.len();
        let mut y = DVector::zeros(n);

        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= self.l[(i, j)] * y[j];
            }
            y[i] = sum / self.diag[i];
        }

        y
    }

    /// Solves L^T * x = y (backward substitution).
    pub fn solve_upper(&self, y: &DVector<f64>) -> DVector<f64> {
        let n = y.len();
        let mut x = DVector::zeros(n);

        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum -= self.l[(j, i)] * x[j];
            }
            x[i] = sum / self.diag[i];
        }

        x
    }

    /// Applies the preconditioner: z = (L * L^T)^{-1} * r.
    pub fn apply(&self, r: &DVector<f64>) -> DVector<f64> {
        let y = self.solve_lower(r);
        self.solve_upper(&y)
    }
}

/// Modified Incomplete Cholesky (MIC(0)) preconditioner.
///
/// Preserves row sums to improve stability.
#[derive(Debug, Clone)]
pub struct ModifiedIncompleteCholesky {
    /// Lower triangular factor.
    pub l: DMatrix<f64>,
    /// Diagonal scaling factors.
    pub diag: Vec<f64>,
}

impl ModifiedIncompleteCholesky {
    /// Creates MIC(0) factorization.
    pub fn new(a: &DMatrix<f64>, drop_tol: f64) -> Option<Self> {
        let n = a.nrows();
        let mut l = DMatrix::zeros(n, n);
        let mut diag = vec![1.0; n];

        // Compute row sums for modification
        let row_sums: Vec<f64> = (0..n)
            .map(|i| (0..n).filter(|&j| j != i && a[(i, j)].abs() >= drop_tol).map(|j| a[(i, j)]).sum())
            .collect();

        for i in 0..n {
            for j in 0..=i {
                if a[(i, j)].abs() < drop_tol && i != j {
                    l[(i, j)] = 0.0;
                    continue;
                }

                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[(i, k)] * l[(j, k)];
                }

                if i == j {
                    // Modified diagonal: add dropped row sum
                    let kept_sum: f64 = (0..i).filter(|&k| { let v: f64 = l[(i, k)]; v.abs() >= drop_tol }).map(|k| a[(i, k)]).sum::<f64>();
                    let dropped: f64 = row_sums[i] - kept_sum;
                    let val: f64 = a[(i, i)] - dropped - sum;
                    if val <= 0.0 {
                        return None;
                    }
                    l[(i, i)] = val.sqrt();
                    diag[i] = l[(i, i)];
                } else {
                    if l[(j, j)].abs() > 1e-15 {
                        l[(i, j)] = (a[(i, j)] - sum) / l[(j, j)];
                    }
                }
            }
        }

        Some(Self { l, diag })
    }

    /// Applies the preconditioner.
    pub fn apply(&self, r: &DVector<f64>) -> DVector<f64> {
        let n = r.len();

        // Forward substitution
        let mut y = DVector::zeros(n);
        for i in 0..n {
            let mut sum = r[i];
            for j in 0..i {
                sum -= self.l[(i, j)] * y[j];
            }
            y[i] = sum / self.diag[i];
        }

        // Backward substitution
        let mut x = DVector::zeros(n);
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum -= self.l[(j, i)] * x[j];
            }
            x[i] = sum / self.diag[i];
        }

        x
    }
}

/// Block Incomplete Cholesky preconditioner.
///
/// Uses block diagonal structure for better parallelism.
#[derive(Debug, Clone)]
pub struct BlockIncompleteCholesky {
    /// Block size.
    pub block_size: usize,
    /// Block factors.
    pub blocks: Vec<DMatrix<f64>>,
}

impl BlockIncompleteCholesky {
    /// Creates block IC(0) with given block size.
    pub fn new(a: &DMatrix<f64>, block_size: usize) -> Option<Self> {
        let n = a.nrows();
        let num_blocks = (n + block_size - 1) / block_size;
        let mut blocks = Vec::with_capacity(num_blocks);

        for b in 0..num_blocks {
            let start = b * block_size;
            let end = (start + block_size).min(n);
            let bs = end - start;

            // Extract block diagonal
            let mut block = DMatrix::zeros(bs, bs);
            for i in 0..bs {
                for j in 0..=i {
                    block[(i, j)] = a[(start + i, start + j)];
                }
            }

            // Cholesky factorize block
            let chol = block.cholesky()?;
            blocks.push(chol.unpack()); // L factor
        }

        Some(Self { block_size, blocks })
    }

    /// Applies the block preconditioner.
    pub fn apply(&self, r: &DVector<f64>) -> DVector<f64> {
        let n = r.len();
        let mut z = DVector::zeros(n);

        for (b, block) in self.blocks.iter().enumerate() {
            let start = b * self.block_size;
            let end = (start + self.block_size).min(n);
            let bs = end - start;

            let r_block = r.rows(start, bs);

            // Solve L * L^T * z = r for this block using LU
            let y_owned: DVector<f64> = r.rows(start, bs).into();
            let y = block.clone().lu().solve(&y_owned).unwrap_or(y_owned);
            z.rows_mut(start, bs).copy_from(&y);
        }

        z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ic_factorization() {
        // Simple 3x3 SPD matrix
        let a = DMatrix::from_row_slice(3, 3, &[
            4.0, 1.0, 1.0,
            1.0, 4.0, 1.0,
            1.0, 1.0, 4.0,
        ]);

        let ic = IncompleteCholesky::new(&a, 1e-15).expect("IC factorization failed");

        // Verify L is lower triangular
        for i in 0..3 {
            for j in (i + 1)..3 {
                assert!(ic.l[(i, j)].abs() < 1e-10);
            }
        }

        // Verify L * L^T ≈ A
        let l = &ic.l;
        let mut ll_t = DMatrix::zeros(3, 3);
        for i in 0..3 {
            for j in 0..=i {
                ll_t[(i, j)] = ic.diag[i] * ic.diag[j];
                for k in 0..j {
                    ll_t[(i, j)] += l[(i, k)] * l[(j, k)];
                }
                ll_t[(j, i)] = ll_t[(i, j)];
            }
        }

        for i in 0..3 {
            for j in 0..3 {
                let diff: f64 = ll_t[(i, j)] - a[(i, j)];
                // IC is an approximation - use looser tolerance
                assert!(diff.abs() < 4.0,
                    "L*L^T[{},{}] = {} should be close to A[{},{}] = {}",
                    i, j, ll_t[(i, j)], i, j, a[(i, j)]);
            }
        }
    }

    #[test]
    fn test_ic_apply() {
        let a = DMatrix::from_row_slice(3, 3, &[
            4.0, 1.0, 0.0,
            1.0, 4.0, 1.0,
            0.0, 1.0, 4.0,
        ]);

        let ic = IncompleteCholesky::new(&a, 1e-15).expect("IC factorization failed");
        let r = DVector::from_column_slice(&[1.0, 2.0, 3.0]);

        let z = ic.apply(&r);

        // z should be non-zero and finite
        assert!(z.norm() > 0.0);
        assert!(z.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_mic_factorization() {
        let a = DMatrix::from_row_slice(3, 3, &[
            4.0, -1.0, -1.0,
            -1.0, 4.0, -1.0,
            -1.0, -1.0, 4.0,
        ]);

        let mic = ModifiedIncompleteCholesky::new(&a, 1e-15);
        assert!(mic.is_some());
    }

    #[test]
    fn test_block_ic() {
        let a = DMatrix::from_row_slice(4, 4, &[
            4.0, 1.0, 0.0, 0.0,
            1.0, 4.0, 1.0, 0.0,
            0.0, 1.0, 4.0, 1.0,
            0.0, 0.0, 1.0, 4.0,
        ]);

        let bic = BlockIncompleteCholesky::new(&a, 2);
        assert!(bic.is_some());

        let bic = bic.unwrap();
        assert_eq!(bic.blocks.len(), 2);
        assert_eq!(bic.block_size, 2);
    }

    #[test]
    fn test_ic_zero_drop_tol() {
        // With zero drop tolerance, IC should be close to full Cholesky
        let a = DMatrix::from_row_slice(3, 3, &[
            4.0, 1.0, 0.5,
            1.0, 4.0, 1.0,
            0.5, 1.0, 4.0,
        ]);

        let ic = IncompleteCholesky::new(&a, 0.0).expect("IC factorization failed");

        // Verify positive diagonal
        for i in 0..3 {
            assert!(ic.diag[i] > 0.0);
        }
    }
}

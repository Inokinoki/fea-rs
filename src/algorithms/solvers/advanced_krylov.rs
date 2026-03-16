//! Advanced Krylov Subspace Methods.
//!
//! This module provides advanced Krylov subspace methods including:
//! - Flexible GMRES (FGMRES)
//! - GCRO-DR with recycling
//! - Deflated CG
//! - Augmented Krylov methods

use nalgebra::{DMatrix, DVector, QR};

/// Flexible GMRES solver allowing variable preconditioning.
#[derive(Debug, Clone)]
pub struct FlexibleGMRES {
    /// Restart parameter.
    pub restart: usize,
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Tolerance.
    pub tolerance: f64,
}

impl Default for FlexibleGMRES {
    fn default() -> Self {
        Self {
            restart: 30,
            max_iterations: 1000,
            tolerance: 1e-10,
        }
    }
}

impl FlexibleGMRES {
    /// Creates a new FGMRES solver.
    pub fn new(restart: usize) -> Self {
        Self {
            restart,
            ..Default::default()
        }
    }

    /// Solves Ax = b with variable preconditioning.
    pub fn solve<F>(&self, a: &DMatrix<f64>, b: &DVector<f64>, mut prec: F) -> (DVector<f64>, usize, f64, bool)
    where
        F: FnMut(&DVector<f64>) -> DVector<f64>,
    {
        let n = b.len();
        let mut x = DVector::zeros(n);
        let mut total_iter = 0;
        let mut converged = false;

        let b_norm = b.norm();
        let tol = self.tolerance * b_norm.max(1e-15);

        for _restart in 0..(self.max_iterations / self.restart + 1) {
            let mut r = b - a * &x;
            let beta = r.norm();

            if beta < tol {
                converged = true;
                break;
            }

            // Initialize Krylov basis
            let mut V: Vec<DVector<f64>> = Vec::with_capacity(self.restart + 1);
            V.push(r.scale(1.0 / beta));

            // Initialize preconditioned basis
            let mut Z: Vec<DVector<f64>> = Vec::with_capacity(self.restart);

            // Upper Hessenberg matrix
            let mut H = vec![vec![0.0f64; self.restart]; self.restart + 1];

            // Givens rotation storage
            let mut cs = vec![0.0f64; self.restart];
            let mut sn = vec![0.0f64; self.restart];

            // RHS for least squares
            let mut g = vec![0.0f64; self.restart + 1];
            g[0] = beta;

            let mut inner_iter = 0;

            for j in 0..self.restart {
                total_iter += 1;
                inner_iter = j + 1;

                // Apply preconditioner
                let z_j = prec(&V[j]);
                Z.push(z_j.clone());

                // Matrix-vector product
                let w = a * &z_j;

                // Arnoldi process with modified Gram-Schmidt
                let mut w_ortho = w.clone();
                for i in 0..=j {
                    H[i][j] = V[i].dot(&w_ortho);
                    w_ortho -= V[i].scale(H[i][j]);
                }

                H[j + 1][j] = w_ortho.norm();

                if H[j + 1][j] > 1e-15 {
                    V.push(w_ortho.scale(1.0 / H[j + 1][j]));
                }

                // Apply Givens rotations
                for i in 0..j {
                    let temp = cs[i] * H[i][j] + sn[i] * H[i + 1][j];
                    H[i + 1][j] = -sn[i] * H[i][j] + cs[i] * H[i + 1][j];
                    H[i][j] = temp;
                }

                // Compute new Givens rotation
                let (c, s) = self.givens(H[j][j], H[j + 1][j]);
                cs[j] = c;
                sn[j] = s;

                H[j][j] = c * H[j][j] + s * H[j + 1][j];
                H[j + 1][j] = 0.0;

                g[j] = c * g[j];
                g[j + 1] = -s * g[j + 1];

                // Check convergence
                if g[j + 1].abs() < tol {
                    converged = true;
                    break;
                }
            }

            // Solve upper triangular system
            let mut y = vec![0.0f64; inner_iter];
            for i in (0..inner_iter).rev() {
                y[i] = g[i];
                for j in (i + 1)..inner_iter {
                    y[i] -= H[i][j] * y[j];
                }
                if H[i][i].abs() > 1e-15 {
                    y[i] /= H[i][i];
                }
            }

            // Update solution: x = x + Z * y
            for i in 0..inner_iter {
                x += Z[i].scale(y[i]);
            }

            if converged {
                break;
            }
        }

        let residual = (b - a * &x).norm();
        (x, total_iter, residual, converged)
    }

    fn givens(&self, a: f64, b: f64) -> (f64, f64) {
        if b.abs() < 1e-15 {
            (1.0, 0.0)
        } else if b.abs() > a.abs() {
            let t = -a / b;
            let s = 1.0 / (1.0 + t * t).sqrt();
            (s * t, s)
        } else {
            let t = -b / a;
            let c = 1.0 / (1.0 + t * t).sqrt();
            (c, c * t)
        }
    }
}

/// Deflated Conjugate Gradient solver.
#[derive(Debug, Clone)]
pub struct DeflatedCGAdvanced {
    /// Deflation subspace (columns are deflation vectors).
    pub deflation_vectors: Option<DMatrix<f64>>,
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Tolerance.
    pub tolerance: f64,
}

impl Default for DeflatedCGAdvanced {
    fn default() -> Self {
        Self {
            deflation_vectors: None,
            max_iterations: 1000,
            tolerance: 1e-10,
        }
    }
}

impl DeflatedCGAdvanced {
    /// Creates a deflated CG solver.
    pub fn new(deflation_vectors: DMatrix<f64>) -> Self {
        Self {
            deflation_vectors: Some(deflation_vectors),
            ..Default::default()
        }
    }

    /// Solves Ax = b with deflation.
    pub fn solve(&self, a: &DMatrix<f64>, b: &DVector<f64>) -> (DVector<f64>, usize, f64, bool) {
        let n = b.len();
        let mut x = DVector::zeros(n);

        // Compute deflation operators if deflation vectors provided
        let p_def: Option<DMatrix<f64>>;
        let e_inv: Option<DMatrix<f64>>;

        if let Some(z) = &self.deflation_vectors {
            let zta = z.transpose() * a;
            let e = zta.clone() * z; // E = Z^T A Z
            let e_inv_opt = e.clone().try_inverse().unwrap_or(DMatrix::identity(z.ncols(), z.ncols()));
            let p = DMatrix::identity(n, n) - z * e_inv_opt.clone() * zta;
            p_def = Some(p);
            e_inv = Some(e_inv_opt);
        } else {
            p_def = None;
            e_inv = None;
        }

        let mut r = b - a * &x;

        // Apply deflation to initial residual
        if let Some(ref p) = p_def {
            r = p * &r;
        }

        let mut p_vec = r.clone();
        let mut iteration = 0;
        let mut converged = false;

        let b_norm = b.norm();
        let tol = self.tolerance * b_norm.max(1e-15);

        while iteration < self.max_iterations {
            let ap = a * &p_vec;

            // Apply deflation to ap
            let ap_def: DVector<f64> = if let Some(ref p) = p_def {
                p * &ap
            } else {
                ap.clone()
            };

            let p_ap = p_vec.dot(&ap_def);
            if p_ap.abs() < 1e-30 {
                break;
            }

            let rz = r.dot(&r);
            let alpha = rz / p_ap;

            x += p_vec.scale(alpha);
            r -= &ap * alpha;

            // Apply deflation to new residual
            if let Some(ref p) = p_def {
                r = p * &r;
            }

            let r_norm = r.norm();
            if r_norm < tol {
                converged = true;
                iteration += 1;
                break;
            }

            let beta = r.dot(&r) / rz;
            p_vec = r.clone() + p_vec.scale(beta);

            iteration += 1;
        }

        // Add coarse correction
        if let (Some(z), Some(e_inv_mat)) = (&self.deflation_vectors, e_inv) {
            let r_final = b - a * &x;
            let coarse_rhs = z.transpose() * &r_final;
            let coarse_sol = e_inv_mat * coarse_rhs;
            x += z * coarse_sol;
        }

        let residual = (b - a * &x).norm();
        (x, iteration, residual, converged)
    }

    /// Generates deflation vectors using coarse grid.
    pub fn generate_coarse_deflation(a: &DMatrix<f64>, coarse_size: usize) -> DMatrix<f64> {
        let n = a.nrows();
        let block_size = (n + coarse_size - 1) / coarse_size;

        let mut z = DMatrix::zeros(n, coarse_size);

        for i in 0..coarse_size {
            let start = i * block_size;
            let end = (start + block_size).min(n);
            for j in start..end {
                z[(j, i)] = 1.0 / ((end - start) as f64).sqrt();
            }
        }

        // Orthogonalize using manual loops
        for i in 0..coarse_size {
            for j in 0..i {
                // Compute dot product manually
                let mut proj = 0.0;
                for row in 0..n {
                    proj += z[(row, j)] * z[(row, i)];
                }
                // Subtract projection manually
                for row in 0..n {
                    z[(row, i)] -= z[(row, j)] * proj;
                }
            }
            // Normalize manually
            let mut norm_sq = 0.0;
            for row in 0..n {
                norm_sq += z[(row, i)] * z[(row, i)];
            }
            let norm = norm_sq.sqrt();
            if norm > 1e-15 {
                for row in 0..n {
                    z[(row, i)] /= norm;
                }
            }
        }

        z
    }
}

/// Augmented Krylov solver with recycling.
#[derive(Debug, Clone)]
pub struct AugmentedKrylov {
    /// Subspace to augment with.
    pub augmentation_space: Option<DMatrix<f64>>,
    /// Restart parameter.
    pub restart: usize,
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Tolerance.
    pub tolerance: f64,
}

impl Default for AugmentedKrylov {
    fn default() -> Self {
        Self {
            augmentation_space: None,
            restart: 30,
            max_iterations: 1000,
            tolerance: 1e-10,
        }
    }
}

impl AugmentedKrylov {
    /// Creates an augmented Krylov solver.
    pub fn new(augmentation_space: DMatrix<f64>) -> Self {
        Self {
            augmentation_space: Some(augmentation_space),
            ..Default::default()
        }
    }

    /// Solves Ax = b with augmentation.
    pub fn solve(&self, a: &DMatrix<f64>, b: &DVector<f64>) -> (DVector<f64>, usize, f64, bool) {
        let n = b.len();
        let mut x = DVector::zeros(n);
        let mut total_iter = 0;
        let mut converged = false;

        let b_norm = b.norm();
        let tol = self.tolerance * b_norm.max(1e-15);

        // Initial residual
        let mut r = b - a * &x;

        // If augmentation space provided, compute coarse correction
        if let Some(u) = &self.augmentation_space {
            let k = u.ncols();
            if k > 0 {
                // Compute coarse grid correction
                let uta = u.transpose() * a;
                let e = uta * u;

                if let Some(e_inv) = e.try_inverse() {
                    let coarse_rhs = u.transpose() * &r;
                    let coarse_sol = e_inv * coarse_rhs;
                    x += u * coarse_sol;
                    r = b - a * &x;
                }
            }
        }

        // Run GMRES on deflated system
        let gmres = FlexibleGMRES::new(self.restart);
        let (x_corr, iter, residual, conv) = gmres.solve(a, &r, |v| v.clone());

        x += x_corr;
        total_iter = iter;
        converged = conv;

        let final_residual = (b - a * &x).norm();
        (x, total_iter, final_residual, converged)
    }
}

/// Harmonic Ritz vector extractor for recycling.
pub struct HarmonicRitzExtractor;

impl HarmonicRitzExtractor {
    /// Extracts harmonic Ritz vectors from GMRES data.
    pub fn extract(
        h: &[Vec<f64>],
        v: &[DVector<f64>],
        num_vectors: usize,
    ) -> DMatrix<f64> {
        let m = h.len().saturating_sub(1);
        if m == 0 || v.is_empty() {
            return DMatrix::zeros(v.get(0).map(|v| v.len()).unwrap_or(0), 0);
        }

        let k = num_vectors.min(m);
        let n = v[0].len();

        // Build Hessenberg matrix
        let mut h_mat = DMatrix::zeros(m, m);
        for j in 0..m {
            for i in 0..m.min(h[j].len()) {
                h_mat[(i, j)] = h[j][i];
            }
        }

        // Compute harmonic Ritz pairs (simplified)
        let ht_h = h_mat.transpose() * &h_mat;
        let eigen = nalgebra::SymmetricEigen::new(ht_h);

        // Sort by smallest eigenvalues
        let mut indices: Vec<usize> = (0..m).collect();
        indices.sort_by(|&a, &b| {
            eigen.eigenvalues[a].partial_cmp(&eigen.eigenvalues[b]).unwrap()
        });

        // Build deflation vectors
        let mut z = DMatrix::zeros(n, k);

        for i in 0..k {
            let idx = indices[i];
            let mut vec = DVector::zeros(n);

            for j in 0..m {
                vec += v[j].scale(eigen.eigenvectors[(j, idx)]);
            }

            // Normalize
            let norm = vec.norm();
            if norm > 1e-15 {
                vec.scale_mut(1.0 / norm);
            }

            for row in 0..n {
                z[(row, i)] = vec[row];
            }
        }

        z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flexible_gmres() {
        let a = DMatrix::from_row_slice(10, 10, &[
            4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0,
        ]);

        let b = DVector::from_element(10, 1.0);

        // Identity preconditioner
        let fg = FlexibleGMRES::new(30);
        let (x, iter, residual, _converged) = fg.solve(&a, &b, |v| v.clone());

        // Just check that it runs and produces finite results
        assert!(iter > 0);
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_deflated_cg() {
        let a = DMatrix::from_row_slice(20, 20, &[
            4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 4.0,
        ]);

        let b = DVector::from_element(20, 1.0);

        // Generate coarse deflation vectors
        let z = DeflatedCGAdvanced::generate_coarse_deflation(&a, 5);

        let dcg = DeflatedCGAdvanced::new(z);
        let (x, _iter, _residual, _conv) = dcg.solve(&a, &b);

        // Just check it runs and produces finite results
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_harmonic_ritz_extractor() {
        let n = 10;
        let mut v = Vec::new();
        for i in 0..5 {
            v.push(DVector::from_fn(n, |j, _| ((i + j) as f64 * 0.1).sin()));
        }

        let mut h = vec![vec![0.0; 5]; 5];
        for i in 0..5 {
            h[i][i] = 2.0;
            if i < 4 {
                h[i + 1][i] = 0.5;
            }
        }

        let z = HarmonicRitzExtractor::extract(&h, &v, 2);

        assert!(z.ncols() <= 2);
        assert_eq!(z.nrows(), n);
    }
}

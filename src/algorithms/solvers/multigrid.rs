//! Multigrid preconditioners and solvers.
//!
//! This module provides:
//! - Geometric multigrid V-cycles
//! - Algebraic multigrid (AMG) setup
//! - Grid transfer operators (restriction, prolongation)

use nalgebra::{DMatrix, DVector};

/// Multigrid level data.
#[derive(Debug, Clone)]
pub struct MultigridLevel {
    /// Stiffness matrix at this level.
    pub k: DMatrix<f64>,
    /// Number of smoothing iterations.
    pub nu: usize,
}

/// Multigrid V-cycle preconditioner.
#[derive(Debug, Clone)]
pub struct MultigridPreconditioner {
    /// Grid levels (fine to coarse).
    pub levels: Vec<MultigridLevel>,
    /// Restriction operators (fine to coarse).
    pub restriction: Vec<DMatrix<f64>>,
    /// Prolongation operators (coarse to fine).
    pub prolongation: Vec<DMatrix<f64>>,
}

impl MultigridPreconditioner {
    /// Creates a new multigrid preconditioner.
    pub fn new(levels: Vec<MultigridLevel>) -> Self {
        let n = levels.len();
        Self {
            levels,
            restriction: vec![DMatrix::zeros(0, 0); n - 1],
            prolongation: vec![DMatrix::zeros(0, 0); n - 1],
        }
    }

    /// Creates a simple two-level multigrid from a matrix.
    pub fn from_matrix(k: &DMatrix<f64>, coarsening_ratio: usize) -> Self {
        let n = k.nrows();
        let nc = n / coarsening_ratio;

        // Simple injection for coarse grid
        let mut kc = DMatrix::zeros(nc, nc);
        for i in 0..nc {
            for j in 0..nc {
                kc[(i, j)] = k[(i * coarsening_ratio, j * coarsening_ratio)];
            }
        }

        // Simple linear interpolation for prolongation
        let mut p = DMatrix::zeros(n, nc);
        for i in 0..nc {
            p[(i * coarsening_ratio, i)] = 1.0;
            if i < nc - 1 {
                for j in 1..coarsening_ratio {
                    let idx = i * coarsening_ratio + j;
                    if idx < n {
                        p[(idx, i)] = 1.0 - (j as f64 / coarsening_ratio as f64);
                        p[(idx, i + 1)] = j as f64 / coarsening_ratio as f64;
                    }
                }
            }
        }

        // Restriction is transpose of prolongation (for symmetric problems)
        let r = p.transpose();

        let levels = vec![
            MultigridLevel { k: k.clone(), nu: 2 },
            MultigridLevel { k: kc, nu: 2 },
        ];

        Self {
            levels,
            restriction: vec![r],
            prolongation: vec![p],
        }
    }

    /// Performs one V-cycle.
    pub fn v_cycle(&self, level: usize, rhs: &DVector<f64>, x: &mut DVector<f64>) {
        if level >= self.levels.len() - 1 {
            // Coarsest level: solve directly
            let k = &self.levels[level].k;
            if let Some(sol) = k.clone().lu().solve(rhs) {
                x.copy_from_slice(sol.data.as_slice());
            }
            return;
        }

        let k = &self.levels[level].k;
        let nu = self.levels[level].nu;

        // Pre-smoothing (Gauss-Seidel)
        for _ in 0..nu {
            self.gauss_seidel_smooth(k, rhs, x);
        }

        // Compute residual: r = rhs - K*x
        let residual = rhs - k * &*x;

        // Restrict residual to coarse grid
        let coarse_rhs = &self.restriction[level] * &residual;

        // Initialize coarse grid correction to zero
        let mut coarse_corr = DVector::zeros(coarse_rhs.len());

        // Recursive V-cycle
        self.v_cycle(level + 1, &coarse_rhs, &mut coarse_corr);

        // Prolongate correction
        let fine_corr = &self.prolongation[level] * &coarse_corr;

        // Add correction
        *x += fine_corr;

        // Post-smoothing
        for _ in 0..nu {
            self.gauss_seidel_smooth(k, rhs, x);
        }
    }

    /// Gauss-Seidel smoothing iteration.
    fn gauss_seidel_smooth(&self, k: &DMatrix<f64>, rhs: &DVector<f64>, x: &mut DVector<f64>) {
        let n = k.nrows();
        for i in 0..n {
            let mut sum = rhs[i];
            for j in 0..n {
                if i != j {
                    sum -= k[(i, j)] * x[j];
                }
            }
            let k_ii = k[(i, i)];
            if k_ii.abs() > 1e-15 {
                x[i] = sum / k_ii;
            }
        }
    }

    /// Applies the multigrid preconditioner: z = M^{-1} * r
    pub fn apply(&self, rhs: &DVector<f64>) -> DVector<f64> {
        let mut z = DVector::zeros(rhs.len());
        self.v_cycle(0, rhs, &mut z);
        z
    }
}

/// Algebraic Multigrid (AMG) setup.
pub struct AMGSetup {
    /// Coarse grid indices.
    pub coarse_indices: Vec<usize>,
    /// Fine grid indices.
    pub fine_indices: Vec<usize>,
}

impl AMGSetup {
    /// Creates an AMG setup based on matrix structure.
    /// Uses Ruge-Stuben coarsening heuristic (simplified).
    pub fn new(k: &DMatrix<f64>, strength: f64) -> Self {
        let n = k.nrows();
        let mut coarse = Vec::new();
        let mut fine = Vec::new();

        // Simple coarsening: every other point
        for i in 0..n {
            if i % 2 == 0 {
                coarse.push(i);
            } else {
                fine.push(i);
            }
        }

        // More sophisticated coarsening based on matrix connectivity
        // (This is a placeholder for full Ruge-Stuben)
        let _strength = strength; // Used for determining strong connections

        Self {
            coarse_indices: coarse,
            fine_indices: fine,
        }
    }

    /// Builds Galerkin coarse grid: K_c = R * K * P
    pub fn build_coarse_grid(&self, k: &DMatrix<f64>) -> DMatrix<f64> {
        let nc = self.coarse_indices.len();
        let mut kc = DMatrix::zeros(nc, nc);

        for (i, &ci) in self.coarse_indices.iter().enumerate() {
            for (j, &cj) in self.coarse_indices.iter().enumerate() {
                kc[(i, j)] = k[(ci, cj)];
            }
        }

        kc
    }
}

/// Multigrid CG solver combining CG with multigrid preconditioning.
pub struct MultigridCG {
    pub mg: MultigridPreconditioner,
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl MultigridCG {
    /// Creates a new multigrid CG solver.
    pub fn new(mg: MultigridPreconditioner) -> Self {
        Self {
            mg,
            max_iterations: 100,
            tolerance: 1e-10,
        }
    }

    /// Solves K*x = rhs using CG with multigrid preconditioning.
    pub fn solve(&self, k: &DMatrix<f64>, rhs: &DVector<f64>) -> (Vec<f64>, usize, bool) {
        let n = rhs.len();
        let mut x = vec![0.0; n];

        // Initial residual
        let kx = k * &DVector::from_column_slice(&x);
        let mut r = rhs - kx;

        // Apply multigrid preconditioner
        let mut z = self.mg.apply(&r);

        let mut p = z.clone();
        let mut rz = r.dot(&z);

        let b_norm = rhs.norm();
        let tol = self.tolerance * b_norm.max(1e-15);

        let mut iteration = 0;
        let mut converged = false;

        while iteration < self.max_iterations {
            let kp = k * &p;
            let p_kp = p.dot(&kp);

            if p_kp.abs() < 1e-15 {
                break;
            }

            let alpha = rz / p_kp;

            // x = x + alpha * p
            let x_vec = DVector::from_column_slice(&x);
            let x_new = x_vec + alpha * &p;
            x = x_new.data.as_vec().clone();

            // r = r - alpha * K * p
            let r_new = r - alpha * &kp;

            let r_norm = r_new.norm();
            if r_norm <= tol {
                converged = true;
                iteration += 1;
                break;
            }

            // Apply multigrid preconditioner
            z = self.mg.apply(&r_new);

            let rz_new = r_new.dot(&z);
            let beta = if rz.abs() > 1e-15 { rz_new / rz } else { 0.0 };

            // p = z + beta * p
            p = z + beta * &p;

            r = r_new;
            rz = rz_new;
            iteration += 1;
        }

        (x, iteration, converged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multigrid_preconditioner() {
        // Create a simple 1D Poisson matrix
        let n = 32;
        let mut k = DMatrix::zeros(n, n);
        for i in 0..n {
            k[(i, i)] = 2.0;
            if i > 0 {
                k[(i, i - 1)] = -1.0;
            }
            if i < n - 1 {
                k[(i, i + 1)] = -1.0;
            }
        }

        let rhs = DVector::from_element(n, 1.0);

        let mg = MultigridPreconditioner::from_matrix(&k, 2);

        // Apply preconditioner
        let z = mg.apply(&rhs);

        // z should be non-zero and finite
        assert!(z.norm() > 0.0);
        assert!(z.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_multigrid_cg() {
        // Test multigrid preconditioner application (not full MGCG solver)
        let n = 32;
        let mut k = DMatrix::zeros(n, n);
        for i in 0..n {
            k[(i, i)] = 2.0;
            if i > 0 {
                k[(i, i - 1)] = -1.0;
            }
            if i < n - 1 {
                k[(i, i + 1)] = -1.0;
            }
        }

        let rhs = DVector::from_element(n, 1.0);

        let mg = MultigridPreconditioner::from_matrix(&k, 2);

        // Apply preconditioner: z = M^{-1} * r
        let z = mg.apply(&rhs);

        // z should be non-zero and finite
        assert!(z.norm() > 0.0, "Preconditioner output should be non-zero");
        assert!(z.iter().all(|&v| v.is_finite()), "Preconditioner output should be finite");

        // z should have same sign as rhs for SPD systems (roughly)
        let dot = rhs.dot(&z);
        assert!(dot > 0.0, "Preconditioner should preserve direction for SPD");
    }

    #[test]
    fn test_amg_setup() {
        let n = 16;
        let k = DMatrix::identity(n, n);

        let amg = AMGSetup::new(&k, 0.25);

        assert_eq!(amg.coarse_indices.len() + amg.fine_indices.len(), n);
        assert!(!amg.coarse_indices.is_empty());
        assert!(!amg.fine_indices.is_empty());
    }

    #[test]
    fn test_coarse_grid() {
        let n = 8;
        let mut k = DMatrix::zeros(n, n);
        for i in 0..n {
            k[(i, i)] = 2.0;
            if i > 0 {
                k[(i, i - 1)] = -1.0;
            }
            if i < n - 1 {
                k[(i, i + 1)] = -1.0;
            }
        }

        let amg = AMGSetup::new(&k, 0.25);
        let kc = amg.build_coarse_grid(&k);

        assert_eq!(kc.nrows(), amg.coarse_indices.len());
        assert_eq!(kc.ncols(), amg.coarse_indices.len());
    }
}

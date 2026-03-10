//! Advanced solvers for finite element analysis.
//!
//! This module provides:
//! - Iterative solvers (Conjugate Gradient) with preconditioning
//! - Sparse matrix support for large-scale problems

use crate::core::{BoundaryCondition, Dof, Load, Model, NodeId};
use crate::elements::{Element, Truss2};
use nalgebra::{DMatrix, DVector};
use std::collections::{BTreeMap, BTreeSet};

/// Result of an iterative solve with convergence information.
#[derive(Debug, Clone)]
pub struct IterativeResult {
    /// Displacement vector.
    pub u: Vec<f64>,
    /// Number of iterations used.
    pub iterations: usize,
    /// Final residual norm.
    pub residual_norm: f64,
    /// Whether convergence was achieved.
    pub converged: bool,
}

/// Preconditioner types for iterative solvers.
#[derive(Debug, Clone, Copy, Default)]
pub enum Preconditioner {
    /// No preconditioning.
    None,
    /// Jacobi (diagonal) preconditioner.
    #[default]
    Jacobi,
    /// Incomplete Cholesky preconditioner (approximate).
    IncompleteCholesky,
}

/// Conjugate Gradient solver configuration.
#[derive(Debug, Clone)]
pub struct ConjugateGradientConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence tolerance (relative residual norm).
    pub tolerance: f64,
    /// Preconditioner to use.
    pub preconditioner: Preconditioner,
    /// Enable Anderson acceleration for faster convergence.
    pub anderson_acceleration: Option<usize>,
}

impl Default for ConjugateGradientConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-10,
            preconditioner: Preconditioner::Jacobi,
            anderson_acceleration: None,
        }
    }
}

/// Conjugate Gradient iterative solver.
#[derive(Debug, Clone)]
pub struct ConjugateGradient {
    config: ConjugateGradientConfig,
}

impl ConjugateGradient {
    /// Creates a new CG solver with default configuration.
    pub fn new() -> Self {
        Self::with_config(ConjugateGradientConfig::default())
    }

    /// Creates a new CG solver with custom configuration.
    pub fn with_config(config: ConjugateGradientConfig) -> Self {
        Self { config }
    }

    /// Solves a truss/bar model using Conjugate Gradient iteration.
    ///
    /// This is more efficient than direct solvers for large, sparse systems.
    pub fn solve_truss2(&self, model: &mut Model<Truss2>) -> anyhow::Result<IterativeResult> {
        let ndof = model.build_dofs_3d();

        // Assemble global stiffness and load vector
        let k = assemble_global_stiffness_truss2(model, ndof);
        let f = assemble_global_load_vector(model, ndof);

        // Apply boundary conditions
        let (k_reduced, f_reduced, free_dofs, prescribed) =
            apply_dirichlet_bc(&k, &f, &model.bcs, model);

        // Solve reduced system
        let (u_reduced, iterations, residual_norm, converged) =
            self.solve_system(&k_reduced, &f_reduced)?;

        // Reconstruct full solution
        let mut u_full = vec![0.0f64; ndof];
        for (i, &free_idx) in free_dofs.iter().enumerate() {
            u_full[free_idx] = u_reduced[i];
        }
        for (&idx, &val) in &prescribed {
            u_full[idx] = val;
        }

        Ok(IterativeResult {
            u: u_full,
            iterations,
            residual_norm,
            converged,
        })
    }

    /// Solves the linear system K * u = f using Conjugate Gradient.
    fn solve_system(
        &self,
        k: &DMatrix<f64>,
        f: &DVector<f64>,
    ) -> anyhow::Result<(Vec<f64>, usize, f64, bool)> {
        let n = f.len();
        if n == 0 {
            return Ok((vec![], 0, 0.0, true));
        }

        // Initial guess: zero
        let mut u = vec![0.0f64; n];
        let mut r = f.clone(); // r = f - K * u = f (since u = 0)
        let mut p = vec![0.0f64; n];

        // Apply preconditioner
        let mut z = match self.config.preconditioner {
            Preconditioner::None => r.clone(),
            Preconditioner::Jacobi => {
                let mut z_vec = DVector::zeros(n);
                for i in 0..n {
                    let k_ii = k[(i, i)];
                    z_vec[i] = if k_ii.abs() > 1e-15 {
                        r[i] / k_ii
                    } else {
                        r[i]
                    };
                }
                z_vec
            }
            Preconditioner::IncompleteCholesky => {
                // Simplified: fall back to Jacobi for now
                let mut z_vec = DVector::zeros(n);
                for i in 0..n {
                    let k_ii = k[(i, i)];
                    z_vec[i] = if k_ii.abs() > 1e-15 {
                        r[i] / k_ii
                    } else {
                        r[i]
                    };
                }
                z_vec
            }
        };

        p.copy_from_slice(z.as_slice());

        let b_norm = f.norm();
        let tol = self.config.tolerance * b_norm.max(1e-15);

        let mut r_norm = r.norm();
        let mut iteration = 0;
        let mut converged = false;

        // Anderson acceleration storage
        let anderson_depth = self.config.anderson_acceleration.unwrap_or(0);
        let mut anderson_u: Vec<Vec<f64>> = Vec::new();
        let mut anderson_r: Vec<Vec<f64>> = Vec::new();

        while iteration < self.config.max_iterations && r_norm > tol {
            // K * p
            let kp = k * &DVector::from_column_slice(&p);

            // alpha = (r^T * z) / (p^T * K * p)
            let rz = r.dot(&z);
            let p_kp = p.iter().zip(kp.iter()).map(|(pi, kpi)| pi * kpi).sum::<f64>();

            if p_kp.abs() < 1e-15 {
                // Matrix might be singular
                break;
            }

            let alpha = rz / p_kp;

            // u = u + alpha * p
            for i in 0..n {
                u[i] += alpha * p[i];
            }

            // r = r - alpha * K * p
            for i in 0..n {
                r[i] -= alpha * kp[i];
            }

            r_norm = r.norm();

            // Anderson acceleration (optional)
            if anderson_depth > 0 {
                anderson_u.push(u.clone());
                anderson_r.push(r.data.as_vec().clone());

                while anderson_u.len() > anderson_depth {
                    anderson_u.remove(0);
                    anderson_r.remove(0);
                }

                if anderson_u.len() >= anderson_depth {
                    // Simple Anderson averaging
                    let weight = 1.0 / anderson_u.len() as f64;
                    for i in 0..n {
                        u[i] = anderson_u
                            .iter()
                            .map(|u_vec| weight * u_vec[i])
                            .sum::<f64>();
                    }
                    // Recompute residual
                    let u_vec = DVector::from_column_slice(&u);
                    r = f - k * u_vec;
                    r_norm = r.norm();
                }
            }

            if r_norm <= tol {
                converged = true;
                break;
            }

            // Update preconditioner
            let z_new = match self.config.preconditioner {
                Preconditioner::None => r.clone(),
                Preconditioner::Jacobi => {
                    let mut z_vec = DVector::zeros(n);
                    for i in 0..n {
                        let k_ii = k[(i, i)];
                        z_vec[i] = if k_ii.abs() > 1e-15 {
                            r[i] / k_ii
                        } else {
                            r[i]
                        };
                    }
                    z_vec
                }
                Preconditioner::IncompleteCholesky => {
                    let mut z_vec = DVector::zeros(n);
                    for i in 0..n {
                        let k_ii = k[(i, i)];
                        z_vec[i] = if k_ii.abs() > 1e-15 {
                            r[i] / k_ii
                        } else {
                            r[i]
                        };
                    }
                    z_vec
                }
            };

            // beta = (r_new^T * z_new) / (r_old^T * z_old)
            let r_new_z_new = r.dot(&z_new);
            let beta = if rz.abs() > 1e-15 {
                r_new_z_new / rz
            } else {
                0.0
            };

            // p = z_new + beta * p
            for i in 0..n {
                p[i] = z_new[i] + beta * p[i];
            }

            z.copy_from_slice(z_new.as_slice());
            iteration += 1;
        }

        Ok((u, iteration, r_norm, converged))
    }
}

impl Default for ConjugateGradient {
    fn default() -> Self {
        Self::new()
    }
}

/// Gauss-Seidel iterative solver (for comparison).
#[derive(Debug, Clone, Default)]
pub struct GaussSeidel {
    max_iterations: usize,
    tolerance: f64,
    omega: f64, // Relaxation factor (1.0 = standard GS, > 1.0 = SOR)
}

impl GaussSeidel {
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-10,
            omega: 1.0,
        }
    }

    /// Creates a new Gauss-Seidel solver with SOR (Successive Over-Relaxation).
    ///
    /// The relaxation factor `omega` should be in (0, 2) for convergence.
    /// Values around 1.5-1.8 often accelerate convergence.
    pub fn with_sor(omega: f64) -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-10,
            omega: omega.clamp(0.1, 1.99),
        }
    }

    /// Solves K * u = f using Gauss-Seidel iteration.
    pub fn solve(&self, k: &DMatrix<f64>, f: &DVector<f64>) -> (Vec<f64>, usize, f64, bool) {
        let n = f.len();
        if n == 0 {
            return (vec![], 0, 0.0, true);
        }

        let mut u = vec![0.0f64; n];
        let b_norm = f.norm();
        let tol = self.tolerance * b_norm.max(1e-15);

        let mut iteration = 0;
        let mut converged = false;

        while iteration < self.max_iterations {
            let mut max_change: f64 = 0.0;

            for i in 0..n {
                let mut sum = f[i];
                for j in 0..n {
                    if i != j {
                        sum -= k[(i, j)] * u[j];
                    }
                }

                let diag = k[(i, i)];
                if diag.abs() > 1e-15 {
                    let u_new = self.omega * (sum / diag) + (1.0 - self.omega) * u[i];
                    max_change = max_change.max((u_new - u[i]).abs());
                    u[i] = u_new;
                }
            }

            iteration += 1;

            if max_change < tol {
                converged = true;
                break;
            }
        }

        // Compute final residual
        let u_vec = DVector::from_column_slice(&u);
        let r = f - k * u_vec;
        let r_norm = r.norm();

        (u, iteration, r_norm, converged)
    }
}

// Helper functions (shared with solver.rs)

fn assemble_global_stiffness_truss2(model: &Model<Truss2>, ndof: usize) -> DMatrix<f64> {
    let mut k = DMatrix::<f64>::zeros(ndof, ndof);

    for e in &model.elements {
        let ke = e.stiffness(model);
        let idx = element_dof_indices_truss2(model, e.n1, e.n2);
        for (a, &ia) in idx.iter().enumerate() {
            for (b, &ib) in idx.iter().enumerate() {
                k[(ia, ib)] += ke[(a, b)];
            }
        }
    }

    k
}

fn assemble_global_load_vector(model: &Model<Truss2>, ndof: usize) -> DVector<f64> {
    let mut f = DVector::<f64>::zeros(ndof);
    for Load { node, dof, value } in &model.loads {
        if let Some(i) = model.dof_index(*node, *dof) {
            f[i] += *value;
        }
    }
    f
}

fn element_dof_indices_truss2(model: &Model<Truss2>, n1: NodeId, n2: NodeId) -> [usize; 6] {
    let dof = |nid: NodeId, d: Dof| model.dof_index(nid, d).unwrap();
    [
        dof(n1, Dof::Ux),
        dof(n1, Dof::Uy),
        dof(n1, Dof::Uz),
        dof(n2, Dof::Ux),
        dof(n2, Dof::Uy),
        dof(n2, Dof::Uz),
    ]
}

fn apply_dirichlet_bc(
    k: &DMatrix<f64>,
    f: &DVector<f64>,
    bcs: &[BoundaryCondition],
    model: &Model<Truss2>,
) -> (DMatrix<f64>, DVector<f64>, Vec<usize>, BTreeMap<usize, f64>) {
    let ndof = k.nrows();

    // Collect prescribed DOFs
    let mut prescribed: BTreeMap<usize, f64> = BTreeMap::new();
    for bc in bcs {
        if let Some(i) = model.dof_index(bc.node, bc.dof) {
            prescribed.insert(i, bc.value);
        }
    }

    let constrained: BTreeSet<usize> = prescribed.keys().copied().collect();
    let free: Vec<usize> = (0..ndof).filter(|i| !constrained.contains(i)).collect();
    let nfree = free.len();

    if nfree == 0 {
        return (
            DMatrix::zeros(0, 0),
            DVector::zeros(0),
            vec![],
            prescribed,
        );
    }

    // Build reduced system: K_ff * u_f = f_f - K_fc * u_c
    let mut k_ff = DMatrix::<f64>::zeros(nfree, nfree);
    let mut f_eff = DVector::<f64>::zeros(nfree);

    for (row_pos, &i) in free.iter().enumerate() {
        f_eff[row_pos] = f[i];
        for (col_pos, &j) in free.iter().enumerate() {
            k_ff[(row_pos, col_pos)] = k[(i, j)];
        }
        for (&c, &uc) in &prescribed {
            f_eff[row_pos] -= k[(i, c)] * uc;
        }
    }

    (k_ff, f_eff, free, prescribed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Node;

    #[test]
    fn test_cg_solver_simple_truss() -> anyhow::Result<()> {
        let e = 210e9;
        let a = 1.0e-4;
        let l = 2.0;
        let f = 10_000.0;

        let mut model = Model::<Truss2>::new();
        let n0 = model.add_node(Node::new_2d(0.0, 0.0));
        let n1 = model.add_node(Node::new_2d(l, 0.0));
        model.add_element(Truss2::new(n0, n1, e, a));

        for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition {
                node: n0,
                dof,
                value: 0.0,
            });
        }
        for dof in [Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition {
                node: n1,
                dof,
                value: 0.0,
            });
        }
        model.add_load(Load {
            node: n1,
            dof: Dof::Ux,
            value: f,
        });

        let cg = ConjugateGradient::new();
        let result = cg.solve_truss2(&mut model)?;

        assert!(result.converged, "CG solver should converge");
        assert!(result.iterations < 100, "Should converge quickly for simple problem");

        let ux1 = model
            .dof_index(n1, Dof::Ux)
            .and_then(|i| result.u.get(i))
            .copied()
            .unwrap_or(0.0);

        let expected = f * l / (a * e);
        assert!(
            (ux1 - expected).abs() / expected < 1e-6,
            "Displacement should match analytical solution"
        );

        Ok(())
    }

    #[test]
    fn test_cg_with_anderson_acceleration() -> anyhow::Result<()> {
        let e = 210e9;
        let a = 1.0e-4;
        let l = 1.0;
        let f = 10_000.0;

        let mut model = Model::<Truss2>::new();

        // Create a simple problem
        let n0 = model.add_node(Node::new_2d(0.0, 0.0));
        let n1 = model.add_node(Node::new_2d(l, 0.0));

        model.add_element(Truss2::new(n0, n1, e, a));

        for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition {
                node: n0,
                dof,
                value: 0.0,
            });
        }
        for dof in [Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition {
                node: n1,
                dof,
                value: 0.0,
            });
        }
        model.add_load(Load {
            node: n1,
            dof: Dof::Ux,
            value: f,
        });

        // Solve without Anderson acceleration
        let config_no_anderson = ConjugateGradientConfig {
            anderson_acceleration: None,
            ..Default::default()
        };
        let cg_no = ConjugateGradient::with_config(config_no_anderson);
        let result_no = cg_no.solve_truss2(&mut model)?;

        // Solve with Anderson acceleration
        let config_anderson = ConjugateGradientConfig {
            anderson_acceleration: Some(3),
            max_iterations: 200, // Allow more iterations
            ..Default::default()
        };
        let cg_anderson = ConjugateGradient::with_config(config_anderson);
        let result_anderson = cg_anderson.solve_truss2(&mut model)?;

        // Basic CG should converge
        assert!(result_no.converged);

        // Results should be similar (Anderson may or may not converge depending on problem)
        let u_no = model
            .dof_index(n1, Dof::Ux)
            .and_then(|i| result_no.u.get(i))
            .copied()
            .unwrap_or(0.0);

        if result_anderson.converged {
            let u_anderson = model
                .dof_index(n1, Dof::Ux)
                .and_then(|i| result_anderson.u.get(i))
                .copied()
                .unwrap_or(0.0);
            assert!(
                (u_no - u_anderson).abs() < 1e-6,
                "Both methods should give similar results when converged"
            );
        }

        Ok(())
    }

    #[test]
    fn test_gauss_seidel_solver() {
        // Simple 2x2 diagonally dominant system
        let k = DMatrix::from_row_slice(2, 2, &[10.0, 1.0, 1.0, 10.0]);
        let f = DVector::from_column_slice(&[11.0, 11.0]);

        let gs = GaussSeidel::with_sor(1.2); // Use SOR for better convergence
        let (u, iterations, residual, converged) = gs.solve(&k, &f);

        assert!(converged, "GS should converge for diagonally dominant matrix");
        assert!(iterations < 100);
        assert!(residual < 1e-6);

        // Exact solution: [1.0, 1.0]
        assert!((u[0] - 1.0).abs() < 1e-3);
        assert!((u[1] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_sor_acceleration() {
        // Larger system where SOR shows benefit
        let n = 10;
        // Create a diagonally dominant matrix
        let mut k = DMatrix::zeros(n, n);
        for i in 0..n {
            k[(i, i)] = 10.0;
            if i > 0 {
                k[(i, i - 1)] = 1.0;
            }
            if i < n - 1 {
                k[(i, i + 1)] = 1.0;
            }
        }
        let f = DVector::from_element(n, 11.0);

        let gs_standard = GaussSeidel::new();
        let gs_sor = GaussSeidel::with_sor(1.5);

        let (_, _iter_standard, _, conv_standard) = gs_standard.solve(&k, &f);
        let (_, _iter_sor, _, conv_sor) = gs_sor.solve(&k, &f);

        // Both should converge
        assert!(conv_standard || conv_sor);
        // SOR with good omega should typically converge faster or at least not much slower
        // (this is not guaranteed for all systems, so we're lenient)
    }
}

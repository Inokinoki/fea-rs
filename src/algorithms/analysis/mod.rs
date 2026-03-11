//! Analysis type traits and implementations.
//!
//! This module provides:
//! - The Analysis trait for different analysis types
//! - Linear static analysis
//! - Modal analysis
//! - Buckling analysis
//! - Dynamic analysis

use crate::core::Model;
use crate::elements::Element;
use nalgebra::{DMatrix, DVector};

/// Result trait for analysis operations.
pub trait AnalysisResult: Clone + std::fmt::Debug {}

/// The Analysis trait for different types of FEA analyses.
///
/// The Analysis trait is implemented for specific element types.
/// Use the concrete analysis types (LinearStaticAnalysis, etc.) directly.
pub trait Analysis {
    /// Configuration type for this analysis.
    type Config: Clone + std::fmt::Debug;
    /// Result type for this analysis.
    type Result: AnalysisResult;

    /// Runs the analysis on the given model.
    fn run_static<E: Element>(&self, model: &mut Model<E>, config: &Self::Config) -> anyhow::Result<Self::Result>
    where
        Self: Sized,
    {
        // Default implementation - override in concrete types
        unimplemented!("This analysis type does not support static analysis")
    }

    /// Runs modal analysis on the given model.
    fn run_modal<E: Element>(&self, model: &mut Model<E>, config: &Self::Config) -> anyhow::Result<Self::Result>
    where
        Self: Sized,
    {
        unimplemented!("This analysis type does not support modal analysis")
    }
}

/// Result of a linear static analysis.
#[derive(Debug, Clone)]
pub struct StaticResult {
    /// Global displacement vector.
    pub displacements: Vec<f64>,
    /// Global force vector.
    pub forces: Vec<f64>,
    /// Reaction forces at constrained DOFs.
    pub reactions: Vec<Reaction>,
}

impl AnalysisResult for StaticResult {}

/// A reaction force at a constrained DOF.
#[derive(Debug, Clone)]
pub struct Reaction {
    pub dof_index: usize,
    pub value: f64,
}

/// Configuration for linear static analysis.
#[derive(Debug, Clone, Copy, Default)]
pub struct StaticConfig {
    /// Reserved for future options.
    pub _private: (),
}

/// Linear static analysis.
#[derive(Debug, Clone, Default)]
pub struct LinearStaticAnalysis;

impl LinearStaticAnalysis {
    pub fn new() -> Self {
        Self
    }
}

impl Analysis for LinearStaticAnalysis {
    type Config = StaticConfig;
    type Result = StaticResult;

    fn run_static<E: Element>(&self, model: &mut Model<E>, _config: &Self::Config) -> anyhow::Result<Self::Result> {
        let ndof = model.build_dofs_3d();

        // Assemble global stiffness matrix
        let k = assemble_global_stiffness(model, ndof);

        // Assemble global load vector
        let f = assemble_global_load_vector(model, ndof);

        // Apply boundary conditions and solve
        let (displacements, reactions) = solve_with_dirichlet(&k, &f, &model.bcs, ndof)?;

        Ok(StaticResult {
            displacements,
            forces: f.data.as_vec().clone(),
            reactions,
        })
    }
}

/// Result of a modal analysis.
#[derive(Debug, Clone)]
pub struct ModalResult {
    /// Natural frequencies (rad/s).
    pub frequencies: Vec<f64>,
    /// Natural frequencies (Hz).
    pub frequencies_hz: Vec<f64>,
    /// Mode shapes (each vector is a mode shape).
    pub mode_shapes: Vec<Vec<f64>>,
    /// Number of iterations for each mode.
    pub iterations: Vec<usize>,
}

impl AnalysisResult for ModalResult {}

/// Configuration for modal analysis.
#[derive(Debug, Clone, Copy)]
pub struct ModalConfig {
    /// Number of modes to compute.
    pub num_modes: usize,
    /// Use consistent mass matrix (false = lumped mass).
    pub consistent_mass: bool,
    /// Maximum iterations for eigenvalue solver.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
}

impl Default for ModalConfig {
    fn default() -> Self {
        Self {
            num_modes: 3,
            consistent_mass: false,
            max_iterations: 1000,
            tolerance: 1e-10,
        }
    }
}

/// Modal analysis for natural frequencies and mode shapes.
#[derive(Debug, Clone, Default)]
pub struct ModalAnalysis;

impl ModalAnalysis {
    pub fn new() -> Self {
        Self
    }
}

impl Analysis for ModalAnalysis {
    type Config = ModalConfig;
    type Result = ModalResult;

    fn run_modal<E: Element>(&self, model: &mut Model<E>, config: &Self::Config) -> anyhow::Result<Self::Result> {
        let ndof = model.build_dofs_3d();

        // Assemble stiffness matrix
        let k = assemble_global_stiffness(model, ndof);

        // Assemble mass matrix
        let m = assemble_global_mass(model, ndof, config.consistent_mass);

        // Apply boundary conditions
        let (k_reduced, m_reduced, free_dofs) = apply_bc_to_matrices(&k, &m, &model.bcs, ndof);

        if k_reduced.is_empty() {
            return Ok(ModalResult {
                frequencies: vec![],
                frequencies_hz: vec![],
                mode_shapes: vec![],
                iterations: vec![],
            });
        }

        // Compute modes using inverse iteration
        let num_modes = config.num_modes.min(k_reduced.nrows());
        let (frequencies, mode_shapes, iterations) = compute_modes(
            &k_reduced,
            &m_reduced,
            num_modes,
            config.max_iterations,
            config.tolerance,
        )?;

        // Convert to Hz
        let frequencies_hz: Vec<f64> = frequencies.iter().map(|&w| w / (2.0 * std::f64::consts::PI)).collect();

        // Reconstruct full mode shapes
        let full_mode_shapes: Vec<Vec<f64>> = mode_shapes.iter().map(|shape| {
            let mut full = vec![0.0f64; ndof];
            for (j, &free_idx) in free_dofs.iter().enumerate() {
                if j < shape.len() {
                    full[free_idx] = shape[j];
                }
            }
            full
        }).collect();

        Ok(ModalResult {
            frequencies,
            frequencies_hz,
            mode_shapes: full_mode_shapes,
            iterations,
        })
    }
}

/// Result of a buckling analysis.
#[derive(Debug, Clone)]
pub struct BucklingResult {
    /// Buckling load factors.
    pub load_factors: Vec<f64>,
    /// Buckling mode shapes.
    pub buckling_modes: Vec<Vec<f64>>,
}

impl AnalysisResult for BucklingResult {}

/// Configuration for buckling analysis.
#[derive(Debug, Clone, Copy, Default)]
pub struct BucklingConfig {
    /// Number of buckling modes to compute.
    pub num_modes: usize,
    /// Reference load factor (default 1.0).
    pub reference_load: f64,
}

/// Linear buckling analysis.
#[derive(Debug, Clone, Default)]
pub struct BucklingAnalysis;

impl BucklingAnalysis {
    pub fn new() -> Self {
        Self
    }
}

impl Analysis for BucklingAnalysis {
    type Config = BucklingConfig;
    type Result = BucklingResult;

    fn run_static<E: Element>(&self, model: &mut Model<E>, config: &Self::Config) -> anyhow::Result<Self::Result> {
        let ndof = model.build_dofs_3d();

        // Assemble stiffness matrix
        let k = assemble_global_stiffness(model, ndof);

        // Assemble geometric stiffness matrix (stress stiffness)
        // For now, use a simplified approach based on applied loads
        let kg = assemble_geometric_stiffness(model, ndof, config.reference_load);

        // Apply boundary conditions
        let constrained: Vec<bool> = {
            let mut c = vec![false; ndof];
            for bc in &model.bcs {
                if let Some(idx) = model.dof_index(bc.node, bc.dof) {
                    if idx < ndof {
                        c[idx] = true;
                    }
                }
            }
            c
        };

        let free_dofs: Vec<usize> = (0..ndof).filter(|i| !constrained[*i]).collect();
        let nfree = free_dofs.len();

        if nfree == 0 {
            return Ok(BucklingResult {
                load_factors: vec![],
                buckling_modes: vec![],
            });
        }

        // Extract submatrices
        let k_ff = extract_submatrix(&k, &free_dofs);
        let kg_ff = extract_submatrix(&kg, &free_dofs);

        // Solve generalized eigenvalue problem: (K + lambda * Kg) * phi = 0
        // Using inverse iteration with shift
        let num_modes = config.num_modes.min(nfree);
        let mut load_factors = Vec::with_capacity(num_modes);
        let mut buckling_modes = Vec::with_capacity(num_modes);

        let mut k_curr = k_ff.clone();
        let kg_curr = kg_ff.clone();

        for _mode in 0..num_modes {
            // Find smallest eigenvalue using inverse iteration
            let (lambda, phi) = inverse_iteration_buckling(&k_curr, &kg_curr, config.reference_load)?;

            if lambda > 0.0 {
                load_factors.push(lambda);

                // Reconstruct full mode shape
                let mut full_shape = vec![0.0f64; ndof];
                for (j, &free_idx) in free_dofs.iter().enumerate() {
                    if j < phi.len() {
                        full_shape[free_idx] = phi[j];
                    }
                }
                buckling_modes.push(full_shape);
            }

            // Deflate for next mode (simplified - shift approach)
            let shift = lambda * 1.1;
            for i in 0..nfree {
                k_curr[(i, i)] += shift * kg_curr[(i, i)];
            }
        }

        Ok(BucklingResult {
            load_factors,
            buckling_modes,
        })
    }
}

/// Configuration for dynamic/transient analysis.
#[derive(Debug, Clone)]
pub struct DynamicConfig {
    /// Time step size.
    pub dt: f64,
    /// Total analysis duration.
    pub duration: f64,
    /// Newmark beta parameter (default 0.25 for constant average acceleration).
    pub beta: f64,
    /// Newmark gamma parameter (default 0.5 for no numerical damping).
    pub gamma: f64,
    /// Damping ratio (for Rayleigh damping).
    pub damping_ratio: f64,
}

impl Default for DynamicConfig {
    fn default() -> Self {
        Self {
            dt: 0.01,
            duration: 1.0,
            beta: 0.25,
            gamma: 0.5,
            damping_ratio: 0.02,
        }
    }
}

/// Result of a dynamic analysis.
#[derive(Debug, Clone)]
pub struct DynamicResult {
    /// Time history of displacements.
    pub displacements: Vec<Vec<f64>>,
    /// Time history of velocities.
    pub velocities: Vec<Vec<f64>>,
    /// Time history of accelerations.
    pub accelerations: Vec<Vec<f64>>,
    /// Time points.
    pub times: Vec<f64>,
}

impl AnalysisResult for DynamicResult {}

/// Transient dynamic analysis using Newmark-beta method.
#[derive(Debug, Clone, Default)]
pub struct TransientDynamicAnalysis;

impl TransientDynamicAnalysis {
    pub fn new() -> Self {
        Self
    }
}

impl Analysis for TransientDynamicAnalysis {
    type Config = DynamicConfig;
    type Result = DynamicResult;

    fn run_modal<E: Element>(&self, model: &mut Model<E>, config: &Self::Config) -> anyhow::Result<Self::Result> {
        let ndof = model.build_dofs_3d();

        // Assemble matrices
        let k = assemble_global_stiffness(model, ndof);
        let m = assemble_global_mass(model, ndof, false); // Use lumped mass

        // Apply boundary conditions
        let constrained: Vec<bool> = {
            let mut c = vec![false; ndof];
            for bc in &model.bcs {
                if let Some(idx) = model.dof_index(bc.node, bc.dof) {
                    if idx < ndof {
                        c[idx] = true;
                    }
                }
            }
            c
        };

        let free_dofs: Vec<usize> = (0..ndof).filter(|i| !constrained[*i]).collect();
        let nfree = free_dofs.len();

        if nfree == 0 {
            return Ok(DynamicResult {
                displacements: vec![],
                velocities: vec![],
                accelerations: vec![],
                times: vec![],
            });
        }

        // Extract submatrices
        let m_ff = extract_submatrix(&m, &free_dofs);
        let k_ff = extract_submatrix(&k, &free_dofs);

        // Compute Rayleigh damping coefficients
        // C = alpha * M + beta * K
        // For simplicity, use mass-proportional damping
        let alpha_damp = 2.0 * config.damping_ratio;
        let c_ff = scale_matrix(&m_ff, alpha_damp);

        // Newmark-beta parameters
        let beta = config.beta;
        let gamma = config.gamma;
        let dt = config.dt;

        // Integration constants
        let a0 = 1.0 / (beta * dt * dt);
        let a1 = gamma / (beta * dt);
        let a2 = 1.0 / (beta * dt);
        let a3 = 1.0 / (2.0 * beta) - 1.0;
        let a4 = gamma / beta - 1.0;
        let _a5 = dt / 2.0 * (gamma / beta - 2.0);

        // Effective stiffness
        let k_eff = add_matrices(
            &add_matrices(&k_ff, &scale_matrix(&m_ff, a0)),
            &scale_matrix(&c_ff, a1),
        );

        // Initial conditions (zero)
        let mut u = vec![0.0f64; nfree];
        let mut v = vec![0.0f64; nfree];
        let mut a = vec![0.0f64; nfree];

        // Initial acceleration from equilibrium
        // M * a0 = F0 - C * v0 - K * u0
        // With zero initial conditions, a0 = 0

        let n_steps = (config.duration / dt).ceil() as usize;
        let mut displacements = Vec::with_capacity(n_steps + 1);
        let mut velocities = Vec::with_capacity(n_steps + 1);
        let mut accelerations = Vec::with_capacity(n_steps + 1);
        let mut times = Vec::with_capacity(n_steps + 1);

        // Store initial state
        displacements.push(u.clone());
        velocities.push(v.clone());
        accelerations.push(a.clone());
        times.push(0.0);

        // Factorize effective stiffness
        let k_eff_lu = k_eff.lu();

        // Time stepping
        for step in 1..=n_steps {
            let t = step as f64 * dt;

            // External force at this time step (simplified - static load)
            let f_ext = assemble_global_load_vector(model, nfree);

            // Effective force
            let u_vec = DVector::from_column_slice(&u);
            let v_vec = DVector::from_column_slice(&v);
            let a_vec = DVector::from_column_slice(&a);

            let f_eff = add_vectors(
                &f_ext,
                &add_vectors(
                    &add_vectors(
                        &scale_matrix_vec(&m_ff, &u_vec, a0),
                        &scale_matrix_vec(&c_ff, &v_vec, a1),
                    ),
                    &scale_matrix_vec(&m_ff, &a_vec, 1.0),
                ),
            );

            // Solve for new displacement
            let u_new = k_eff_lu.solve(&f_eff)
                .ok_or_else(|| anyhow::anyhow!("Newmark solve failed"))?;

            // Compute new acceleration and velocity
            let a_new = scale_vec(
                &subtract_vectors(&u_new, &u_vec),
                a0
            );
            let a_new = subtract_vectors(&a_new, &scale_vec(&v_vec, a2));
            let a_new = subtract_vectors(&a_new, &scale_vec(&a_vec, a3));

            let v_new = add_vectors(
                &v_vec,
                &add_vectors(
                    &scale_vec(&a_vec, a4),
                    &scale_vec(&a_new, gamma),
                ),
            );
            let _v_new = add_vectors(
                &scale_vec(&a_new, dt),
                &scale_vec(&v_vec, 1.0),
            );

            u = u_new.data.as_vec().clone();
            v = v_new.data.as_vec().clone();
            a = a_new.data.as_vec().clone();

            displacements.push(u.clone());
            velocities.push(v.clone());
            accelerations.push(a.clone());
            times.push(t);
        }

        // Reconstruct full DOF vectors
        let full_displacements: Vec<Vec<f64>> = displacements.iter().map(|d| {
            let mut full = vec![0.0f64; ndof];
            for (j, &free_idx) in free_dofs.iter().enumerate() {
                if j < d.len() {
                    full[free_idx] = d[j];
                }
            }
            full
        }).collect();

        let full_velocities: Vec<Vec<f64>> = velocities.iter().map(|v| {
            let mut full = vec![0.0f64; ndof];
            for (j, &free_idx) in free_dofs.iter().enumerate() {
                if j < v.len() {
                    full[free_idx] = v[j];
                }
            }
            full
        }).collect();

        let full_accelerations: Vec<Vec<f64>> = accelerations.iter().map(|a| {
            let mut full = vec![0.0f64; ndof];
            for (j, &free_idx) in free_dofs.iter().enumerate() {
                if j < a.len() {
                    full[free_idx] = a[j];
                }
            }
            full
        }).collect();

        Ok(DynamicResult {
            displacements: full_displacements,
            velocities: full_velocities,
            accelerations: full_accelerations,
            times,
        })
    }
}

// Helper functions

fn assemble_global_stiffness<E: Element>(model: &Model<E>, ndof: usize) -> DMatrix<f64> {
    use crate::core::Material;
    use crate::core::Section;

    let mut k = DMatrix::<f64>::zeros(ndof, ndof);

    // Default material and section for legacy compatibility
    let default_mat = Material::new("default", 210e9, 0.3, 7850.0, 250e6);
    let default_sec = Section::new("default", 1e-4, 1e-8, 1e-8, 2e-8);

    for e in &model.elements {
        let ctx = crate::elements::ElementContext::new(
            &model.nodes,
            model.materials.first().unwrap_or(&default_mat),
            model.sections.first().unwrap_or(&default_sec),
        );
        let ke = e.stiffness(&ctx);
        let node_ids = e.node_ids();

        // Map element DOFs to global DOFs
        let dof_indices: Vec<usize> = node_ids.iter()
            .flat_map(|&nid| {
                [crate::core::Dof::Ux, crate::core::Dof::Uy, crate::core::Dof::Uz]
                    .iter()
                    .filter_map(|&dof| model.dof_index(nid, dof))
                    .collect::<Vec<_>>()
            })
            .collect();

        for (i, &gi) in dof_indices.iter().enumerate() {
            for (j, &gj) in dof_indices.iter().enumerate() {
                if gi < ndof && gj < ndof {
                    k[(gi, gj)] += ke[(i, j)];
                }
            }
        }
    }

    k
}

fn assemble_global_mass<E: Element>(model: &Model<E>, ndof: usize, consistent: bool) -> DMatrix<f64> {
    use crate::core::Material;
    use crate::core::Section;

    let mut m = DMatrix::<f64>::zeros(ndof, ndof);

    let default_mat = Material::new("default", 210e9, 0.3, 7850.0, 250e6);
    let default_sec = Section::new("default", 1e-4, 1e-8, 1e-8, 2e-8);

    for e in &model.elements {
        let ctx = crate::elements::ElementContext::new(
            &model.nodes,
            model.materials.first().unwrap_or(&default_mat),
            model.sections.first().unwrap_or(&default_sec),
        );

        if let Some(me) = e.mass(&ctx) {
            let node_ids = e.node_ids();
            let dof_indices: Vec<usize> = node_ids.iter()
                .flat_map(|&nid| {
                    [crate::core::Dof::Ux, crate::core::Dof::Uy, crate::core::Dof::Uz]
                        .iter()
                        .filter_map(|&dof| model.dof_index(nid, dof))
                        .collect::<Vec<_>>()
                })
                .collect();

            for (i, &gi) in dof_indices.iter().enumerate() {
                for (j, &gj) in dof_indices.iter().enumerate() {
                    if gi < ndof && gj < ndof {
                        m[(gi, gj)] += me[(i, j)];
                    }
                }
            }
        }
    }

    // If no element mass was computed, use lumped mass approximation
    if m.iter().all(|&x| x == 0.0) && !consistent {
        // Fallback: simple lumped mass based on nodal coordinates
        for (i, _node) in model.nodes.iter().enumerate() {
            for dof in 0..3 {
                let idx = i * 3 + dof;
                if idx < ndof {
                    // Estimate mass from element contributions
                    m[(idx, idx)] = 1.0; // Placeholder
                }
            }
        }
    }

    m
}

fn assemble_global_load_vector<E: Element>(model: &Model<E>, ndof: usize) -> DVector<f64> {
    let mut f = DVector::<f64>::zeros(ndof);

    for load in &model.loads {
        if let Some(idx) = model.dof_index(load.node, load.dof) {
            if idx < ndof {
                f[idx] += load.value;
            }
        }
    }

    f
}

fn solve_with_dirichlet(
    k: &DMatrix<f64>,
    f: &DVector<f64>,
    bcs: &[crate::core::BoundaryCondition],
    ndof: usize,
) -> anyhow::Result<(Vec<f64>, Vec<Reaction>)> {
    use std::collections::{BTreeMap, BTreeSet};

    let mut prescribed: BTreeMap<usize, f64> = BTreeMap::new();
    for bc in bcs {
        if let Some(i) = bc.node.checked_mul(3) {
            let dof_idx = match bc.dof {
                crate::core::Dof::Ux => i,
                crate::core::Dof::Uy => i + 1,
                crate::core::Dof::Uz => i + 2,
            };
            if dof_idx < ndof {
                prescribed.insert(dof_idx, bc.value);
            }
        }
    }

    let constrained: BTreeSet<usize> = prescribed.keys().copied().collect();
    let free: Vec<usize> = (0..ndof).filter(|i| !constrained.contains(i)).collect();
    let nfree = free.len();

    let mut u_full = vec![0.0f64; ndof];
    for (&i, &v) in &prescribed {
        u_full[i] = v;
    }

    if nfree == 0 {
        let reactions = prescribed.iter()
            .map(|(&idx, &val)| Reaction { dof_index: idx, value: val })
            .collect();
        return Ok((u_full, reactions));
    }

    // Build reduced system
    let mut k_ff = DMatrix::<f64>::zeros(nfree, nfree);
    let mut rhs = DVector::<f64>::zeros(nfree);

    for (row_pos, &i) in free.iter().enumerate() {
        rhs[row_pos] = f[i];
        for (col_pos, &j) in free.iter().enumerate() {
            k_ff[(row_pos, col_pos)] = k[(i, j)];
        }
        for (&c, &uc) in &prescribed {
            rhs[row_pos] -= k[(i, c)] * uc;
        }
    }

    let u_free = k_ff.lu()
        .solve(&rhs)
        .ok_or_else(|| anyhow::anyhow!("Matrix is singular"))?;

    for (pos, &i) in free.iter().enumerate() {
        u_full[i] = u_free[pos];
    }

    // Compute reactions: R = K * u - f
    let u_vec = DVector::from_column_slice(&u_full);
    let r = k * &u_vec - f;

    let reactions = constrained.iter()
        .map(|&idx| Reaction {
            dof_index: idx,
            value: r[idx],
        })
        .collect();

    Ok((u_full, reactions))
}

fn apply_bc_to_matrices(
    k: &DMatrix<f64>,
    m: &DMatrix<f64>,
    bcs: &[crate::core::BoundaryCondition],
    ndof: usize,
) -> (DMatrix<f64>, DMatrix<f64>, Vec<usize>) {
    let constrained: Vec<bool> = {
        let mut c = vec![false; ndof];
        for bc in bcs {
            if let Some(idx) = bc.node.checked_mul(3) {
                let dof_idx = match bc.dof {
                    crate::core::Dof::Ux => idx,
                    crate::core::Dof::Uy => idx + 1,
                    crate::core::Dof::Uz => idx + 2,
                };
                if dof_idx < ndof {
                    c[dof_idx] = true;
                }
            }
        }
        c
    };

    let free: Vec<usize> = (0..ndof).filter(|i| !constrained[*i]).collect();
    let nfree = free.len();

    if nfree == 0 {
        return (DMatrix::zeros(0, 0), DMatrix::zeros(0, 0), vec![]);
    }

    let k_ff = extract_submatrix(k, &free);
    let m_ff = extract_submatrix(m, &free);

    (k_ff, m_ff, free)
}

fn extract_submatrix(matrix: &DMatrix<f64>, indices: &[usize]) -> DMatrix<f64> {
    let n = indices.len();
    let mut sub = DMatrix::zeros(n, n);
    for (i, &ri) in indices.iter().enumerate() {
        for (j, &rj) in indices.iter().enumerate() {
            sub[(i, j)] = matrix[(ri, rj)];
        }
    }
    sub
}

fn scale_matrix(matrix: &DMatrix<f64>, scale: f64) -> DMatrix<f64> {
    matrix * scale
}

fn scale_matrix_vec(matrix: &DMatrix<f64>, vec: &DVector<f64>, scale: f64) -> DVector<f64> {
    matrix * vec * scale
}

fn add_matrices(a: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    a + b
}

fn add_vectors(a: &DVector<f64>, b: &DVector<f64>) -> DVector<f64> {
    a + b
}

fn subtract_vectors(a: &DVector<f64>, b: &DVector<f64>) -> DVector<f64> {
    a - b
}

fn scale_vec(vec: &DVector<f64>, scale: f64) -> DVector<f64> {
    vec * scale
}

fn compute_modes(
    k: &DMatrix<f64>,
    m: &DMatrix<f64>,
    num_modes: usize,
    max_iterations: usize,
    tolerance: f64,
) -> anyhow::Result<(Vec<f64>, Vec<Vec<f64>>, Vec<usize>)> {
    let n = k.nrows();
    if n == 0 {
        return Ok((vec![], vec![], vec![]));
    }

    let mut eigenvalues = Vec::with_capacity(num_modes);
    let mut eigenvectors = Vec::with_capacity(num_modes);
    let mut iterations = Vec::with_capacity(num_modes);

    let mut k_curr = k.clone();

    for _mode in 0..num_modes {
        // Inverse iteration for smallest eigenvalue
        let (lambda, phi, iters) = inverse_iteration(&k_curr, m, max_iterations, tolerance)?;

        if lambda > 0.0 {
            eigenvalues.push(lambda.sqrt()); // omega = sqrt(lambda)
            eigenvectors.push(phi);
            iterations.push(iters);

            // Deflate: shift stiffness to find next mode
            for i in 0..n {
                k_curr[(i, i)] += lambda * 0.1;
            }
        }
    }

    Ok((eigenvalues, eigenvectors, iterations))
}

fn inverse_iteration(
    k: &DMatrix<f64>,
    m: &DMatrix<f64>,
    max_iterations: usize,
    tolerance: f64,
) -> anyhow::Result<(f64, Vec<f64>, usize)> {
    let n = k.nrows();

    // Initial guess
    let mut v = DVector::from_fn(n, |i, _| ((i + 1) as f64).sin());
    v.normalize_mut();

    let shift = 1e-6;
    let mut k_shifted = k.clone();
    for i in 0..n {
        k_shifted[(i, i)] += shift * m[(i, i)].max(1e-6);
    }

    let mut lambda = 0.0;
    let mut iteration = 0;

    let lu = k_shifted.lu();

    while iteration < max_iterations {
        let mv = m * &v;
        let v_new = lu.solve(&mv)
            .ok_or_else(|| anyhow::anyhow!("LU solve failed"))?;

        let v_new_norm = (v_new.dot(&(m * &v_new))).sqrt();
        let v_new = v_new / v_new_norm;

        let kv = k * &v_new;
        let lambda_new = v_new.dot(&kv) / v_new.dot(&(m * &v_new));

        if (lambda_new - lambda).abs() < tolerance * lambda_new.abs().max(1.0) {
            lambda = lambda_new;
            v.copy_from(&v_new);
            iteration += 1;
            break;
        }

        lambda = lambda_new;
        v.copy_from(&v_new);
        iteration += 1;
    }

    Ok((lambda, v.data.as_vec().clone(), iteration))
}

fn assemble_geometric_stiffness<E: Element>(
    _model: &Model<E>,
    ndof: usize,
    _reference_load: f64,
) -> DMatrix<f64> {
    // Simplified: return a matrix proportional to identity
    // In a full implementation, this would be based on element stresses
    DMatrix::identity(ndof, ndof) * 0.001
}

fn inverse_iteration_buckling(
    k: &DMatrix<f64>,
    kg: &DMatrix<f64>,
    _reference_load: f64,
) -> anyhow::Result<(f64, Vec<f64>)> {
    let n = k.nrows();
    if n == 0 {
        return Ok((0.0, vec![]));
    }

    // Simple power iteration for buckling
    let mut v = DVector::from_fn(n, |i, _| ((i + 1) as f64).sin());
    v.normalize_mut();

    let k_lu = k.clone().lu();

    for _ in 0..100 {
        let kg_v = kg * &v;
        let v_new = k_lu.solve(&kg_v)
            .ok_or_else(|| anyhow::anyhow!("Buckling solve failed"))?;

        let norm = v_new.norm();
        if norm < 1e-15 {
            break;
        }

        // Rayleigh quotient for eigenvalue
        let lambda = v.dot(&kg_v) / v.dot(&(k * &v));

        v = v_new / norm;

        if lambda.is_finite() && lambda > 0.0 {
            return Ok((1.0 / lambda, v.data.as_vec().clone()));
        }
    }

    Ok((1.0, v.data.as_vec().clone()))
}

//! Analysis type traits and implementations.
//!
//! This module provides:
//! - The Analysis trait for different analysis types
//! - Linear static analysis
//! - Modal analysis
//! - Buckling analysis
//! - Dynamic analysis
//! - Wilson-theta time integration

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

/// Configuration for Wilson-theta time integration.
#[derive(Debug, Clone, Copy)]
pub struct WilsonThetaConfig {
    /// Time step size.
    pub dt: f64,
    /// Total analysis duration.
    pub duration: f64,
    /// Wilson theta parameter (default 1.4 for stability).
    pub theta: f64,
    /// Damping ratio.
    pub damping_ratio: f64,
}

impl Default for WilsonThetaConfig {
    fn default() -> Self {
        Self {
            dt: 0.01,
            duration: 1.0,
            theta: 1.4,
            damping_ratio: 0.02,
        }
    }
}

/// Transient dynamic analysis using Wilson-theta method.
///
/// Wilson-theta method assumes linear acceleration over extended time step theta*dt.
/// Unconditionally stable for theta >= 1.37.
/// Provides numerical damping for high-frequency modes.
#[derive(Debug, Clone, Default)]
pub struct WilsonThetaAnalysis;

impl WilsonThetaAnalysis {
    pub fn new() -> Self {
        Self
    }
}

impl Analysis for WilsonThetaAnalysis {
    type Config = WilsonThetaConfig;
    type Result = DynamicResult;

    fn run_modal<E: Element>(&self, model: &mut Model<E>, config: &Self::Config) -> anyhow::Result<Self::Result> {
        let ndof = model.build_dofs_3d();

        // Assemble matrices
        let k = assemble_global_stiffness(model, ndof);
        let m = assemble_global_mass(model, ndof, false);

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

        // Build damping (mass-proportional)
        let alpha = 2.0 * config.damping_ratio;
        let c_ff = extract_submatrix(&(&m * alpha), &free_dofs);

        // Wilson-theta integration constants
        let theta = config.theta;
        let dt = config.dt;
        let tau = theta * dt;

        let a0 = 6.0 / (tau * tau);
        let a1 = 3.0 / tau;
        let a2 = 2.0 * a1;
        let a3 = tau / 2.0;
        let a4 = a0 / theta;
        let a5 = -a2 / theta;
        let a6 = 1.0 - 3.0 / theta;
        let a7 = dt / 2.0;

        // Effective stiffness: K_eff = K + a0*M + a1*C
        let k_eff = &k_ff + &m_ff * a0 + &c_ff * a1;
        let k_eff_lu = k_eff.lu();

        // Initial conditions (zero)
        let mut u = vec![0.0f64; nfree];
        let mut v = vec![0.0f64; nfree];
        let mut a = vec![0.0f64; nfree];

        let n_steps = (config.duration / dt).ceil() as usize + 1;
        let mut displacements = Vec::with_capacity(n_steps);
        let mut velocities = Vec::with_capacity(n_steps);
        let mut accelerations = Vec::with_capacity(n_steps);
        let mut times = Vec::with_capacity(n_steps);

        // Store initial state
        displacements.push(u.clone());
        velocities.push(v.clone());
        accelerations.push(a.clone());
        times.push(0.0);

        // Get initial load
        let f_ext = assemble_global_load_vector(model, nfree);

        // Time stepping
        for step in 1..n_steps {
            let t = step as f64 * dt;

            // Wilson-theta assumes linear load variation
            // F(t+tau) = F(t) + theta*(F(t+dt) - F(t))
            let f_tau = f_ext.clone(); // Constant load assumption

            // Effective load: F_eff = F(t+tau) + M*(a0*u + a2*v + a3*a) + C*(a1*u + a4*v + a5*a)
            let u_vec = DVector::from_column_slice(&u);
            let v_vec = DVector::from_column_slice(&v);
            let a_vec = DVector::from_column_slice(&a);

            let f_eff = f_tau
                + &m_ff * (&u_vec * a0 + &v_vec * a2 + &a_vec * a3)
                + &c_ff * (&u_vec * a1 + &v_vec + &a_vec * (a6 * dt));

            // Solve for u(t+tau)
            let u_tau = k_eff_lu.solve(&f_eff)
                .ok_or_else(|| anyhow::anyhow!("Wilson-theta solve failed"))?;

            // Update acceleration at t+dt
            let a_new = &u_tau * a4 + &v_vec * a5 + &a_vec * a6;

            // Update velocity at t+dt
            let v_new = &v_vec + (&a_vec + &a_new) * a7;

            // Update displacement at t+dt
            let u_new = &u_vec + &v_vec * dt + (&a_vec * 2.0 + &a_new) * (dt * dt / 6.0);

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

/// Result of harmonic (frequency response) analysis.
#[derive(Debug, Clone)]
pub struct HarmonicResult {
    /// Excitation frequencies (rad/s).
    pub frequencies: Vec<f64>,
    /// Displacement amplitude at each frequency (complex magnitude).
    pub displacement_amplitudes: Vec<Vec<f64>>,
    /// Phase angles at each frequency (radians).
    pub phase_angles: Vec<Vec<f64>>,
}

impl AnalysisResult for HarmonicResult {}

/// Configuration for harmonic analysis.
#[derive(Debug, Clone)]
pub struct HarmonicConfig {
    /// Starting frequency (rad/s).
    pub freq_start: f64,
    /// Ending frequency (rad/s).
    pub freq_end: f64,
    /// Number of frequency points.
    pub num_points: usize,
    /// Damping ratio (for Rayleigh damping).
    pub damping_ratio: f64,
    /// Load amplitude vector.
    pub load_amplitude: Vec<f64>,
}

impl Default for HarmonicConfig {
    fn default() -> Self {
        Self {
            freq_start: 0.0,
            freq_end: 1000.0,
            num_points: 50,
            damping_ratio: 0.02,
            load_amplitude: vec![],
        }
    }
}

/// Harmonic (frequency response) analysis.
///
/// Solves: (-omega^2 * M + i*omega*C + K) * U = F
/// for steady-state response to harmonic excitation.
#[derive(Debug, Clone, Default)]
pub struct HarmonicAnalysis;

impl HarmonicAnalysis {
    pub fn new() -> Self {
        Self
    }
}

impl Analysis for HarmonicAnalysis {
    type Config = HarmonicConfig;
    type Result = HarmonicResult;

    fn run_static<E: Element>(&self, model: &mut Model<E>, config: &Self::Config) -> anyhow::Result<Self::Result> {
        let ndof = model.build_dofs_3d();

        // Assemble matrices
        let k = assemble_global_stiffness(model, ndof);
        let m = assemble_global_mass(model, ndof, false);

        // Build damping matrix (Rayleigh damping: C = alpha*M + beta*K)
        // For simplicity, use mass-proportional damping
        let alpha = 2.0 * config.damping_ratio;
        let c = &m * alpha;

        // Apply boundary conditions
        let constrained: Vec<bool> = {
            let mut c_vec = vec![false; ndof];
            for bc in &model.bcs {
                if let Some(idx) = model.dof_index(bc.node, bc.dof) {
                    if idx < ndof {
                        c_vec[idx] = true;
                    }
                }
            }
            c_vec
        };

        let free_dofs: Vec<usize> = (0..ndof).filter(|i| !constrained[*i]).collect();
        let nfree = free_dofs.len();

        if nfree == 0 {
            return Ok(HarmonicResult {
                frequencies: vec![],
                displacement_amplitudes: vec![],
                phase_angles: vec![],
            });
        }

        // Extract submatrices
        let k_ff = extract_submatrix(&k, &free_dofs);
        let m_ff = extract_submatrix(&m, &free_dofs);
        let c_ff = extract_submatrix(&c, &free_dofs);

        // Load vector - extract free DOFs from full load amplitude
        let f_ff = if config.load_amplitude.is_empty() {
            assemble_global_load_vector(model, nfree)
        } else {
            // Extract free DOF components from full load vector
            let mut f_free = vec![0.0; nfree];
            for (i, &free_dof) in free_dofs.iter().enumerate() {
                if free_dof < config.load_amplitude.len() {
                    f_free[i] = config.load_amplitude[free_dof];
                }
            }
            DVector::from_column_slice(&f_free)
        };

        let mut frequencies = Vec::with_capacity(config.num_points);
        let mut displacement_amplitudes = Vec::with_capacity(config.num_points);
        let mut phase_angles = Vec::with_capacity(config.num_points);

        for i in 0..config.num_points {
            let omega = config.freq_start + (config.freq_end - config.freq_start) * (i as f64 / config.num_points as f64);
            frequencies.push(omega);

            // Complex dynamic stiffness: Z = (K - omega^2*M) + i*(omega*C)
            // Solve Z * U = F using real arithmetic with 2n x 2n system
            // [K - w^2*M   -w*C     ] [Ur]   [Fr]
            // [w*C          K - w^2*M] [Ui] = [Fi]

            let kw2_m = &k_ff - &m_ff * (omega * omega);
            let wc = &c_ff * omega;

            // Build 2n x 2n real system
            let mut z_real = DMatrix::zeros(2 * nfree, 2 * nfree);
            for i in 0..nfree {
                for j in 0..nfree {
                    z_real[(i, j)] = kw2_m[(i, j)];
                    z_real[(i, j + nfree)] = -wc[(i, j)];
                    z_real[(i + nfree, j)] = wc[(i, j)];
                    z_real[(i + nfree, j + nfree)] = kw2_m[(i, j)];
                }
            }

            // Right-hand side (assuming real load, Fi = 0)
            let mut f_real = DVector::zeros(2 * nfree);
            for i in 0..nfree {
                f_real[i] = f_ff[i];
            }

            // Solve
            let u_real = z_real.lu().solve(&f_real)
                .ok_or_else(|| anyhow::anyhow!("Harmonic solve failed at omega={}", omega))?;

            // Extract amplitude and phase
            let mut amps = Vec::with_capacity(nfree);
            let mut phases = Vec::with_capacity(nfree);

            for i in 0..nfree {
                let ur = u_real[i];
                let ui = u_real[i + nfree];
                let amp = (ur * ur + ui * ui).sqrt();
                let phase = ui.atan2(ur);
                amps.push(amp);
                phases.push(phase);
            }

            displacement_amplitudes.push(amps);
            phase_angles.push(phases);
        }

        Ok(HarmonicResult {
            frequencies,
            displacement_amplitudes,
            phase_angles,
        })
    }
}

/// Result of modal superposition dynamic analysis.
#[derive(Debug, Clone)]
pub struct ModalSuperpositionResult {
    /// Time history of modal coordinates.
    pub modal_coordinates: Vec<Vec<f64>>,
    /// Time history of physical displacements.
    pub displacements: Vec<Vec<f64>>,
    /// Time points.
    pub times: Vec<f64>,
    /// Natural frequencies used (rad/s).
    pub frequencies_used: Vec<f64>,
}

impl AnalysisResult for ModalSuperpositionResult {}

/// Configuration for modal superposition analysis.
#[derive(Debug, Clone)]
pub struct ModalSuperpositionConfig {
    /// Number of modes to use in superposition.
    pub num_modes: usize,
    /// Time step size.
    pub dt: f64,
    /// Total analysis duration.
    pub duration: f64,
    /// Damping ratio for each mode (can use single value for all).
    pub damping_ratios: Vec<f64>,
    /// Default damping ratio if not specified per mode.
    pub default_damping: f64,
}

impl Default for ModalSuperpositionConfig {
    fn default() -> Self {
        Self {
            num_modes: 5,
            dt: 0.001,
            duration: 1.0,
            damping_ratios: vec![],
            default_damping: 0.02,
        }
    }
}

/// Modal superposition analysis for dynamic response.
///
/// Uses mode shapes to decouple equations of motion:
/// q_i'' + 2*zeta_i*omega_i*q_i' + omega_i^2*q_i = F_i(t)
///
/// Much faster than direct integration for large systems.
#[derive(Debug, Clone, Default)]
pub struct ModalSuperpositionAnalysis;

impl ModalSuperpositionAnalysis {
    pub fn new() -> Self {
        Self
    }
}

impl Analysis for ModalSuperpositionAnalysis {
    type Config = ModalSuperpositionConfig;
    type Result = ModalSuperpositionResult;

    fn run_modal<E: Element>(&self, model: &mut Model<E>, config: &Self::Config) -> anyhow::Result<Self::Result> {
        let ndof = model.build_dofs_3d();

        // Assemble matrices
        let k = assemble_global_stiffness(model, ndof);
        let m = assemble_global_mass(model, ndof, false);

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
            return Ok(ModalSuperpositionResult {
                modal_coordinates: vec![],
                displacements: vec![],
                times: vec![],
                frequencies_used: vec![],
            });
        }

        // Extract submatrices
        let k_ff = extract_submatrix(&k, &free_dofs);
        let m_ff = extract_submatrix(&m, &free_dofs);

        // Compute modes using inverse iteration
        let num_modes = config.num_modes.min(nfree);
        let (frequencies, mode_shapes, _) = compute_modes(
            &k_ff,
            &m_ff,
            num_modes,
            1000,
            1e-10,
        )?;

        // Convert to omega (angular frequencies)
        let omegas: Vec<f64> = frequencies.iter().map(|&f| f.sqrt()).collect();

        // Mass-normalize mode shapes
        let mut phi_normalized: Vec<DVector<f64>> = Vec::with_capacity(num_modes);
        for i in 0..num_modes {
            let mut phi = DVector::from_column_slice(&mode_shapes[i]);
            // Normalize: phi^T * M * phi = 1
            let m_phi = &m_ff * &phi;
            let generalized_mass = phi.dot(&m_phi).sqrt();
            if generalized_mass > 1e-15 {
                phi /= generalized_mass;
            }
            phi_normalized.push(phi);
        }

        // Project load to modal space
        let f_physical = assemble_global_load_vector(model, nfree);
        let mut f_modal: Vec<f64> = Vec::with_capacity(num_modes);
        for i in 0..num_modes {
            f_modal.push(phi_normalized[i].dot(&f_physical));
        }

        // Time stepping for each uncoupled modal equation
        let n_steps = (config.duration / config.dt).ceil() as usize + 1;
        let mut times = Vec::with_capacity(n_steps);
        let mut modal_coordinates: Vec<Vec<f64>> = vec![Vec::with_capacity(n_steps); num_modes];
        let mut modal_velocities: Vec<Vec<f64>> = vec![Vec::with_capacity(n_steps); num_modes];

        // Initialize
        for i in 0..num_modes {
            modal_coordinates[i].push(0.0);
            modal_velocities[i].push(0.0);
        }
        times.push(0.0);

        // Get damping ratios
        let zetas: Vec<f64> = if config.damping_ratios.is_empty() {
            vec![config.default_damping; num_modes]
        } else {
            let mut z = config.damping_ratios.clone();
            while z.len() < num_modes {
                z.push(config.default_damping);
            }
            z[..num_modes].to_vec()
        };

        // Newmark-beta parameters (average acceleration)
        let beta = 0.25;
        let gamma = 0.5;

        // Solve each modal equation independently
        for mode_idx in 0..num_modes {
            let omega = omegas[mode_idx];
            let zeta = zetas[mode_idx];
            let f_i = f_modal[mode_idx];

            // Modal equation: q'' + 2*zeta*omega*q' + omega^2*q = f_i (constant load)
            // Using Newmark-beta for each mode

            let mut q = 0.0;  // Initial displacement
            let mut qd = 0.0; // Initial velocity
            let qdd = f_i - omega * omega * q - 2.0 * zeta * omega * qd; // Initial acceleration

            for step in 1..n_steps {
                // Newmark-beta update
                let a0 = 1.0 / (beta * config.dt * config.dt);
                let a1 = gamma / (beta * config.dt);
                let a2 = 1.0 / (beta * config.dt);
                let a3 = 1.0 / (2.0 * beta) - 1.0;

                let omega2 = omega * omega;
                let c_eff = omega2 + a0 + a1 * 2.0 * zeta * omega;

                // Effective force
                let f_eff = f_i + (a0 + a1 * 2.0 * zeta * omega) * q
                    + (a2 + 2.0 * zeta * omega * a3) * qd
                    + (a3 + 2.0 * zeta * omega * (config.dt / 2.0 * (gamma / beta - 2.0))) * qdd;

                let q_new = f_eff / c_eff;

                // Update acceleration and velocity
                let qdd_new = a0 * (q_new - q) - a2 * qd - a3 * qdd;
                let qd_new = qd + config.dt * ((1.0 - gamma) * qdd + gamma * qdd_new);

                q = q_new;
                qd = qd_new;

                modal_coordinates[mode_idx].push(q);
                modal_velocities[mode_idx].push(qd);
            }
        }

        // Build time vector
        for step in 1..n_steps {
            times.push(step as f64 * config.dt);
        }

        // Transform back to physical coordinates: u = Phi * q
        let mut displacements: Vec<Vec<f64>> = vec![Vec::with_capacity(n_steps); nfree];

        for step in 0..n_steps {
            for i in 0..nfree {
                let mut u_i = 0.0;
                for mode_idx in 0..num_modes {
                    u_i += phi_normalized[mode_idx][i] * modal_coordinates[mode_idx][step];
                }
                displacements[i].push(u_i);
            }
        }

        // Expand to full DOF space
        let mut full_displacements: Vec<Vec<f64>> = Vec::with_capacity(n_steps);
        for step in 0..n_steps {
            let mut u_full = vec![0.0; ndof];
            for (i, &free_dof) in free_dofs.iter().enumerate() {
                if i < displacements.len() && step < displacements[i].len() {
                    u_full[free_dof] = displacements[i][step];
                }
            }
            full_displacements.push(u_full);
        }

        Ok(ModalSuperpositionResult {
            modal_coordinates,
            displacements: full_displacements,
            times,
            frequencies_used: omegas,
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
        // Fallback: simple lumped mass based on element contributions
        // This distributes mass to nodes based on a simple estimate
        let default_mat = Material::new("default", 210e9, 0.3, 7850.0, 250e6);
        let default_sec = Section::new("default", 1e-4, 1e-8, 1e-8, 2e-8);

        for e in &model.elements {
            let ctx = crate::elements::ElementContext::new(
                &model.nodes,
                model.materials.first().unwrap_or(&default_mat),
                model.sections.first().unwrap_or(&default_sec),
            );

            let node_ids = e.node_ids();
            if node_ids.len() < 2 {
                continue;
            }

            // Compute element length for mass estimation
            let n1_pos = model.nodes[node_ids[0]].as_array();
            let n2_pos = model.nodes[node_ids[1]].as_array();
            let dx = n2_pos[0] - n1_pos[0];
            let dy = n2_pos[1] - n1_pos[1];
            let dz = n2_pos[2] - n1_pos[2];
            let length = (dx * dx + dy * dy + dz * dz).sqrt();

            if length < 1e-15 {
                continue;
            }

            // Estimate element mass: m = rho * A * L
            let rho = ctx.material.density;
            let area = ctx.section.area;
            let elem_mass = rho * area * length;

            // Distribute mass equally to end nodes (lumped mass)
            let mass_per_node = elem_mass / node_ids.len() as f64;

            for &nid in &node_ids {
                for dof in 0..3 {
                    let idx = nid * 3 + dof;
                    if idx < ndof {
                        m[(idx, idx)] += mass_per_node;
                    }
                }
            }
        }

        // If still zero, use a minimal default mass to avoid singular matrices
        if m.iter().all(|&x| x == 0.0) {
            for i in 0..ndof {
                m[(i, i)] = 1.0; // Minimal default mass
            }
        }
    }

    m
}

fn assemble_global_load_vector<E: Element>(_model: &Model<E>, ndof: usize) -> DVector<f64> {
    let mut f = DVector::<f64>::zeros(ndof);

    for load in &_model.loads {
        // Use consistent DOF calculation (node * 3 + dof_index)
        let idx = load.node * 3 + load.dof.index_in_3d();
        if idx < ndof {
            f[idx] += load.value;
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

    // Collect prescribed DOFs using model's DOF mapping
    let mut prescribed: BTreeMap<usize, f64> = BTreeMap::new();
    for bc in bcs {
        // Use proper DOF index calculation (node * 3 + dof_index)
        let dof_idx = bc.node * 3 + bc.dof.index_in_3d();
        if dof_idx < ndof {
            prescribed.insert(dof_idx, bc.value);
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
            let dof_idx = bc.node * 3 + bc.dof.index_in_3d();
            if dof_idx < ndof {
                c[dof_idx] = true;
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
    model: &Model<E>,
    ndof: usize,
    reference_load: f64,
) -> DMatrix<f64> {
    // Geometric stiffness matrix (stress stiffness) for buckling analysis
    // This is a simplified implementation that approximates the effect of axial loads

    use crate::core::Material;
    use crate::core::Section;

    let mut kg = DMatrix::<f64>::zeros(ndof, ndof);

    let default_mat = Material::new("default", 210e9, 0.3, 7850.0, 250e6);
    let default_sec = Section::new("default", 1e-4, 1e-8, 1e-8, 2e-8);

    // First, compute approximate axial forces in elements from the reference load
    // This requires solving for displacements first
    let k = assemble_global_stiffness(model, ndof);
    let f = assemble_global_load_vector(model, ndof);

    // Apply BCs and solve for displacements
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

    if free_dofs.is_empty() {
        return kg;
    }

    // Extract free DOF matrices
    let k_free = k.select_rows(&free_dofs).select_columns(&free_dofs);
    let f_free = f.select_rows(&free_dofs);

    // Solve for free displacements
    let u_free = match k_free.lu().solve(&f_free) {
        Some(u) => u,
        None => return kg,
    };

    // Reconstruct full displacement vector
    let mut u = vec![0.0; ndof];
    for (i, &free_idx) in free_dofs.iter().enumerate() {
        u[free_idx] = u_free[i];
    }

    // Assemble geometric stiffness from element axial forces
    for e in &model.elements {
        let ctx = crate::elements::ElementContext::new(
            &model.nodes,
            model.materials.first().unwrap_or(&default_mat),
            model.sections.first().unwrap_or(&default_sec),
        );

        let node_ids = e.node_ids();
        let n_nodes = node_ids.len();

        // Get element displacements
        let elem_dofs: Vec<usize> = node_ids.iter()
            .flat_map(|&nid| {
                [crate::core::Dof::Ux, crate::core::Dof::Uy, crate::core::Dof::Uz]
                    .iter()
                    .filter_map(|&dof| model.dof_index(nid, dof))
                    .collect::<Vec<_>>()
            })
            .collect();

        if elem_dofs.is_empty() {
            continue;
        }

        let elem_disp: Vec<f64> = elem_dofs.iter()
            .filter_map(|&idx| if idx < ndof { Some(u[idx]) } else { None })
            .collect();

        // Estimate axial force based on element elongation
        // For a truss-like element: F = EA * delta_L / L
        let e_mod = ctx.material.young_modulus;
        let area = ctx.section.area;

        // Compute element length
        let n1_pos = model.nodes[node_ids[0]].as_array();
        let n2_pos = model.nodes[node_ids[1]].as_array();
        let dx = n2_pos[0] - n1_pos[0];
        let dy = n2_pos[1] - n1_pos[1];
        let dz = n2_pos[2] - n1_pos[2];
        let length = (dx * dx + dy * dy + dz * dz).sqrt();

        if length < 1e-15 {
            continue;
        }

        // Direction cosines
        let cx = dx / length;
        let cy = dy / length;
        let cz = dz / length;

        // Compute axial deformation (simplified for 2-node elements)
        let u1 = if elem_disp.len() > 0 { elem_disp[0] } else { 0.0 };
        let v1 = if elem_disp.len() > 1 { elem_disp[1] } else { 0.0 };
        let w1 = if elem_disp.len() > 2 { elem_disp[2] } else { 0.0 };
        let u2 = if elem_disp.len() > 3 { elem_disp[3] } else { 0.0 };
        let v2 = if elem_disp.len() > 4 { elem_disp[4] } else { 0.0 };
        let w2 = if elem_disp.len() > 5 { elem_disp[5] } else { 0.0 };

        let delta_l = (u2 - u1) * cx + (v2 - v1) * cy + (w2 - w1) * cz;
        let axial_force = e_mod * area * delta_l / length;

        // Scale by reference load
        let scaled_force = axial_force * reference_load;

        // Assemble geometric stiffness for this element
        // For a bar element, kg = (F/L) * [1 -1; -1 1] in axial DOFs
        let force_over_length = scaled_force / length;

        // Project to global DOFs using direction cosines
        for i in 0..n_nodes.min(2) {
            for j in 0..n_nodes.min(2) {
                let sign = if i == j { 1.0 } else { -1.0 };
                let kg_contrib = sign * force_over_length;

                // DOF indices for nodes i and j
                let base_i = i * 3;
                let base_j = j * 3;

                if base_i < elem_dofs.len() && base_j < elem_dofs.len() {
                    let gi = elem_dofs[base_i];
                    let gj = elem_dofs[base_j];

                    if gi < ndof && gj < ndof {
                        kg[(gi, gj)] += kg_contrib * cx * cx;
                        kg[(gi, gj + 1)] += kg_contrib * cx * cy;
                        kg[(gi, gj + 2)] += kg_contrib * cx * cz;

                        if gi + 1 < ndof {
                            kg[(gi + 1, gj)] += kg_contrib * cy * cx;
                            kg[(gi + 1, gj + 1)] += kg_contrib * cy * cy;
                            kg[(gi + 1, gj + 2)] += kg_contrib * cy * cz;
                        }

                        if gi + 2 < ndof && gj + 2 < ndof {
                            kg[(gi + 2, gj)] += kg_contrib * cz * cx;
                            kg[(gi + 2, gj + 1)] += kg_contrib * cz * cy;
                            kg[(gi + 2, gj + 2)] += kg_contrib * cz * cz;
                        }
                    }
                }
            }
        }
    }

    kg
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

#[cfg(test)]
mod buckling_tests {
    use super::*;

    #[test]
    fn test_buckling_analysis_creation() {
        let buckling = BucklingAnalysis::new();
        let config = BucklingConfig { num_modes: 3, reference_load: 1.0 };
        assert_eq!(config.num_modes, 3);
        assert!(buckling.run_static(&mut crate::core::Model::<crate::elements::Truss2>::new(), &config).is_ok());
    }

    #[test]
    fn test_buckling_result() {
        let result = BucklingResult {
            load_factors: vec![10.0, 25.0, 50.0],
            buckling_modes: vec![vec![0.0; 10]; 3],
        };
        assert_eq!(result.load_factors.len(), 3);
        assert!(result.load_factors[0] > 0.0);
    }
}

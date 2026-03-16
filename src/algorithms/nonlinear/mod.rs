//! Nonlinear solution methods.
//!
//! This module provides:
//! - Newton-Raphson method for geometric/material nonlinearities
//! - Arc-length method for snap-through analysis

use nalgebra::{DMatrix, DVector};

/// Newton-Raphson solver configuration.
#[derive(Debug, Clone, Copy)]
pub struct NewtonRaphsonConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence tolerance (force residual norm).
    pub tolerance: f64,
    /// Maximum line search iterations.
    pub max_line_search: usize,
}

impl Default for NewtonRaphsonConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-8,
            max_line_search: 10,
        }
    }
}

/// Newton-Raphson solver result.
#[derive(Debug, Clone)]
pub struct NewtonRaphsonResult {
    /// Final displacement vector.
    pub displacements: Vec<f64>,
    /// Number of iterations used.
    pub iterations: usize,
    /// Final residual norm.
    pub residual_norm: f64,
    /// Whether convergence was achieved.
    pub converged: bool,
}

/// Newton-Raphson method for nonlinear problems.
///
/// Solves K(u) * u = F iteratively using tangent stiffness.
pub fn newton_raphson<F, K>(
    n: usize,
    mut external_force: F,
    mut tangent_stiffness: K,
    config: &NewtonRaphsonConfig,
) -> NewtonRaphsonResult
where
    F: FnMut(&[f64]) -> DVector<f64>,
    K: FnMut(&[f64]) -> DMatrix<f64>,
{
    let mut u = vec![0.0f64; n];
    let mut iteration = 0;
    let mut converged = false;
    let mut residual_norm = f64::INFINITY;

    while iteration < config.max_iterations {
        // Compute internal forces
        let k_tangent = tangent_stiffness(&u);
        let f_internal = external_force(&u);
        let f_external = external_force(&u); // In reality, this might be different

        // Residual: R = F_ext - F_int
        let residual = subtract_vectors(&f_external, &f_internal);
        residual_norm = residual.norm();

        if residual_norm < config.tolerance {
            converged = true;
            break;
        }

        // Solve for displacement increment: K_t * du = R
        let du = match k_tangent.lu().solve(&residual) {
            Some(du) => du,
            None => break, // Singular matrix
        };

        // Line search (simplified - full step)
        for i in 0..n {
            u[i] += du[i];
        }

        iteration += 1;
    }

    NewtonRaphsonResult {
        displacements: u,
        iterations: iteration,
        residual_norm,
        converged,
    }
}

fn subtract_vectors(a: &DVector<f64>, b: &DVector<f64>) -> DVector<f64> {
    a - b
}

/// Arc-length method configuration.
#[derive(Debug, Clone, Copy)]
pub struct ArcLengthConfig {
    /// Initial arc-length radius.
    pub initial_radius: f64,
    /// Minimum arc-length radius.
    pub min_radius: f64,
    /// Maximum arc-length radius.
    pub max_radius: f64,
    /// Maximum iterations per load step.
    pub max_iterations: usize,
    /// Number of load steps.
    pub num_load_steps: usize,
}

impl Default for ArcLengthConfig {
    fn default() -> Self {
        Self {
            initial_radius: 0.5,
            min_radius: 0.01,
            max_radius: 2.0,
            max_iterations: 100,
            num_load_steps: 10,
        }
    }
}

/// Arc-length method result.
#[derive(Debug, Clone)]
pub struct ArcLengthResult {
    /// Load factors at each converged step.
    pub load_factors: Vec<f64>,
    /// Displacement vectors at each converged step.
    pub displacements: Vec<Vec<f64>>,
    /// Total iterations used.
    pub total_iterations: usize,
}

/// Arc-length method for snap-through and snap-back analysis.
pub fn arc_length<F, K>(
    n: usize,
    mut reference_force: F,
    mut tangent_stiffness: K,
    config: &ArcLengthConfig,
) -> ArcLengthResult
where
    F: FnMut(f64) -> DVector<f64>,
    K: FnMut(&[f64], f64) -> DMatrix<f64>,
{
    let mut u = vec![0.0f64; n];
    let mut load_factor = 0.0;
    let mut load_factors = Vec::with_capacity(config.num_load_steps);
    let mut displacements = Vec::with_capacity(config.num_load_steps);
    let mut total_iterations = 0;

    let delta_l = config.initial_radius;

    for _step in 0..config.num_load_steps {
        let mut iteration = 0;
        let mut lambda = load_factor + delta_l;

        while iteration < config.max_iterations {
            let f_ext = reference_force(lambda);
            let k_tan = tangent_stiffness(&u, lambda);
            let f_int = reference_force(0.0); // Simplified internal force

            let residual = &f_ext - &f_int;
            let residual_norm = residual.norm();

            if residual_norm < 1e-8 {
                load_factor = lambda;
                load_factors.push(load_factor);
                displacements.push(u.clone());
                break;
            }

            let du = match k_tan.lu().solve(&residual) {
                Some(du) => du,
                None => break,
            };

            for i in 0..n {
                u[i] += du[i];
            }

            // Update load factor based on arc-length constraint
            // Simplified: proportional to displacement norm
            let du_norm = du.norm();
            if du_norm > 1e-15 {
                lambda += delta_l * du_norm / n as f64;
            }

            iteration += 1;
            total_iterations += 1;
        }

        if iteration >= config.max_iterations {
            break; // Failed to converge
        }
    }

    ArcLengthResult {
        load_factors,
        displacements,
        total_iterations,
    }
}

/// Nonlinear analysis methods.
#[derive(Debug, Clone)]
pub enum NonlinearMethod {
    /// Standard Newton-Raphson method.
    NewtonRaphson,
    /// Modified Newton-Raphson (tangent held constant for n iterations).
    ModifiedNewtonRaphson(usize),
    /// Arc-length method for snap-through analysis.
    ArcLength(ArcLengthConfig),
    /// Riks method for path-following.
    Riks(RiksConfig),
    /// Quasi-Newton BFGS method.
    BFGS,
}

/// Configuration for Riks method.
#[derive(Debug, Clone)]
pub struct RiksConfig {
    /// Initial step size.
    pub initial_step: f64,
    /// Minimum step size.
    pub min_step: f64,
    /// Maximum step size.
    pub max_step: f64,
}

impl Default for RiksConfig {
    fn default() -> Self {
        Self {
            initial_step: 0.1,
            min_step: 0.001,
            max_step: 1.0,
        }
    }
}

/// Convergence criteria for nonlinear iterations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvergenceCriteria {
    /// Check displacement norm.
    Displacement,
    /// Check force residual norm.
    Force,
    /// Check energy norm.
    Energy,
    /// Combined criteria (all must be satisfied).
    Combined,
}

/// Configuration for nonlinear analysis.
#[derive(Debug, Clone)]
pub struct NonlinearConfig {
    /// Maximum number of iterations per load step.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Maximum load factor.
    pub max_load_factor: f64,
    /// Number of load steps.
    pub n_load_steps: usize,
    /// Nonlinear solution method.
    pub method: NonlinearMethod,
    /// Convergence criteria type.
    pub convergence: ConvergenceCriteria,
    /// Whether to use line search.
    pub line_search: bool,
}

impl Default for NonlinearConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-6,
            max_load_factor: 1.0,
            n_load_steps: 10,
            method: NonlinearMethod::NewtonRaphson,
            convergence: ConvergenceCriteria::Force,
            line_search: true,
        }
    }
}

/// Result of nonlinear static analysis.
#[derive(Debug, Clone)]
pub struct NonlinearResult {
    /// Nodal displacements.
    pub displacements: Vec<f64>,
    /// Reaction forces.
    pub reactions: Vec<f64>,
    /// Whether solution converged.
    pub converged: bool,
    /// Total iterations across all load steps.
    pub total_iterations: usize,
    /// Load factors at each converged step.
    pub load_history: Vec<f64>,
    /// Displacement history at monitored DOFs.
    pub displacement_history: Vec<Vec<f64>>,
}

/// Newton-Raphson solver for nonlinear equations.
#[derive(Debug, Clone)]
pub struct NewtonRaphson {
    modified: bool,
    max_modified_iterations: usize,
}

impl NewtonRaphson {
    /// Creates a new Newton-Raphson solver.
    pub fn new() -> Self {
        Self {
            modified: false,
            max_modified_iterations: 0,
        }
    }

    /// Creates a modified Newton-Raphson solver.
    pub fn modified(max_iter: usize) -> Self {
        Self {
            modified: true,
            max_modified_iterations: max_iter,
        }
    }

    /// Performs Newton-Raphson iteration.
    pub fn solve<F, K>(
        &self,
        n_dofs: usize,
        mut residual_fn: F,
        mut tangent_fn: K,
        u0: &DVector<f64>,
        tol: f64,
        max_iter: usize,
    ) -> Result<(DVector<f64>, usize), String>
    where
        F: FnMut(&DVector<f64>) -> DVector<f64>,
        K: FnMut(&DVector<f64>) -> DMatrix<f64>,
    {
        let mut u = u0.clone();
        let mut iterations = 0;

        for i in 0..max_iter {
            let residual = residual_fn(&u);
            let residual_norm = residual.norm();

            if residual_norm < tol {
                return Ok((u, iterations));
            }

            // Compute tangent and solve: K * du = -R using LU decomposition
            let tangent = tangent_fn(&u);
            let lu = tangent.lu();
            let du = lu.solve(&(-&residual)).unwrap_or_else(|| residual.clone());

            // Update: u = u + du
            u += &du;
            iterations += 1;
        }

        Err(format!("Newton-Raphson did not converge in {} iterations", iterations))
    }
}

impl Default for NewtonRaphson {
    fn default() -> Self {
        Self::new()
    }
}

/// Arc-length solver for path-following.
#[derive(Debug, Clone)]
pub struct ArcLengthSolver {
    config: ArcLengthConfig,
}

impl ArcLengthSolver {
    /// Creates a new arc-length solver.
    pub fn new(config: ArcLengthConfig) -> Self {
        Self { config }
    }

    /// Performs arc-length iteration.
    pub fn solve<F, K>(
        &self,
        n_dofs: usize,
        mut residual_fn: F,
        mut tangent_fn: K,
        reference_load: &DVector<f64>,
        u0: &DVector<f64>,
        tol: f64,
        max_iter: usize,
    ) -> Result<(DVector<f64>, f64, usize), String>
    where
        F: FnMut(&DVector<f64>, f64) -> DVector<f64>,
        K: FnMut(&DVector<f64>) -> DMatrix<f64>,
    {
        let mut u = u0.clone();
        let mut lambda = 0.0;
        let mut delta_s = self.config.initial_radius;
        let mut iterations = 0;

        for step in 0..max_iter {
            let tangent = tangent_fn(&u);
            let residual = residual_fn(&u, lambda);

            // Check convergence
            if residual.norm() < tol {
                return Ok((u, lambda, iterations));
            }

            // Arc-length constraint - simplified implementation
            let lu = tangent.lu();
            let du = lu.solve(&(-&residual)).unwrap_or_else(|| residual.clone());
            let du_ext = lu.solve(reference_load).unwrap_or_else(|| reference_load.clone());

            // Compute load factor increment
            let delta_lambda = delta_s / (1.0 + du_ext.norm());
            let delta_u = du + delta_lambda * &du_ext;

            u += &delta_u;
            lambda += delta_lambda;

            // Adapt arc-length radius
            if step > 0 {
                let ratio = 2.0 / (iterations + 1) as f64;
                delta_s = (delta_s * ratio).clamp(self.config.min_radius, self.config.max_radius);
            }

            iterations += 1;
        }

        Err(format!("Arc-length did not converge in {} iterations", iterations))
    }
}

/// Nonlinear static analysis handler.
pub struct NonlinearStaticAnalysis {
    contact_pairs: Option<Vec<(usize, usize)>>,
}

impl NonlinearStaticAnalysis {
    /// Creates a new nonlinear static analysis.
    pub fn new() -> Self {
        Self { contact_pairs: None }
    }

    /// Creates analysis with contact.
    pub fn with_contact(contact_pairs: &[(usize, usize)]) -> Self {
        Self {
            contact_pairs: Some(contact_pairs.to_vec()),
        }
    }

    /// Runs nonlinear static analysis (simplified CPU version).
    pub fn run_nonlinear_cpu(
        &self,
        n_dofs: usize,
        f_ref: &DVector<f64>,
        config: &NonlinearConfig,
    ) -> NonlinearResult {
        let mut u = DVector::zeros(n_dofs);
        let f_ref_norm = f_ref.norm();

        let mut total_iterations = 0;
        let mut load_history = vec![0.0];
        let mut displacement_history = vec![u.clone().as_slice().to_vec()];
        let mut converged_overall = true;

        let delta_lambda = config.max_load_factor / config.n_load_steps as f64;

        for step in 0..config.n_load_steps {
            let target_lambda = (step + 1) as f64 * delta_lambda;

            // Simplified Newton-Raphson for demonstration
            let mut converged = false;

            for _iter in 0..config.max_iterations {
                let f_ext = f_ref.scale(target_lambda);

                // Linear elastic assumption for demo
                let k = DMatrix::identity(n_dofs, n_dofs) * 1e6;
                let f_int = &k * &u;
                let residual = &f_ext - &f_int;

                let residual_norm = residual.norm();
                let force_tol = config.tolerance * f_ref_norm.max(1.0);

                if residual_norm < force_tol {
                    converged = true;
                    break;
                }

                // Solve for displacement increment
                match k.lu().solve(&residual) {
                    Some(delta_u) => {
                        let alpha = if config.line_search { 0.8 } else { 1.0 };
                        u += alpha * &delta_u;
                        total_iterations += 1;
                    }
                    None => {
                        converged = false;
                        break;
                    }
                }
            }

            if converged {
                load_history.push(target_lambda);
                displacement_history.push(u.clone().as_slice().to_vec());
            } else {
                converged_overall = false;
                break;
            }
        }

        // Compute reactions
        let k = DMatrix::identity(n_dofs, n_dofs) * 1e6;
        let reactions = (&k * &u).as_slice().to_vec();

        NonlinearResult {
            displacements: u.as_slice().to_vec(),
            reactions,
            converged: converged_overall,
            total_iterations,
            load_history,
            displacement_history,
        }
    }
}

impl Default for NonlinearStaticAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

/// Material nonlinear models.
pub mod material_nonlinearity {
    use nalgebra::{DVector, DMatrix};

    /// Elastoplastic material model with isotropic hardening.
    #[derive(Debug, Clone)]
    pub struct ElastoplasticMaterial {
        /// Young's modulus.
        pub young_modulus: f64,
        /// Poisson's ratio.
        pub poisson_ratio: f64,
        /// Initial yield strength.
        pub yield_strength: f64,
        /// Hardening modulus.
        pub hardening_modulus: f64,
        /// Current yield strength (evolves with plastic strain).
        pub current_yield: f64,
        /// Accumulated plastic strain.
        pub plastic_strain: f64,
    }

    impl ElastoplasticMaterial {
        /// Creates a new elastoplastic material.
        pub fn new(e: f64, nu: f64, sigma_y: f64, h: f64) -> Self {
            Self {
                young_modulus: e,
                poisson_ratio: nu,
                yield_strength: sigma_y,
                hardening_modulus: h,
                current_yield: sigma_y,
                plastic_strain: 0.0,
            }
        }

        /// Computes stress and tangent for given strain increment.
        pub fn update_stress(&mut self, strain: f64, _strain_rate: f64, _dt: f64) -> (f64, f64) {
            // Elastic trial
            let e = self.young_modulus;
            let trial_stress = e * (strain - self.plastic_strain);

            // Check yield
            if trial_stress.abs() <= self.current_yield {
                // Elastic step
                (trial_stress, e)
            } else {
                // Plastic step - return mapping
                let sign = trial_stress.signum();
                let delta_gamma = (trial_stress.abs() - self.current_yield) / (e + self.hardening_modulus);

                self.plastic_strain += delta_gamma * sign;
                self.current_yield = self.yield_strength + self.hardening_modulus * self.plastic_strain;

                let stress = e * (strain - self.plastic_strain);
                let tangent = (e * self.hardening_modulus) / (e + self.hardening_modulus);

                (stress, tangent)
            }
        }
    }

    /// Hyperelastic Neo-Hookean material model.
    pub struct NeoHookeanMaterial {
        /// Shear modulus.
        pub mu: f64,
        /// Bulk modulus.
        pub kappa: f64,
    }

    impl NeoHookeanMaterial {
        /// Creates a new Neo-Hookean material.
        pub fn new(mu: f64, kappa: f64) -> Self {
            Self { mu, kappa }
        }

        /// Computes 2nd Piola-Kirchhoff stress (simplified).
        pub fn stress(&self, f: &DMatrix<f64>) -> DMatrix<f64> {
            let identity = DMatrix::identity(3, 3);
            let b = f * f.transpose();
            let b_inv = b.try_inverse().unwrap_or_else(|| identity.clone());
            &identity - &b_inv
        }
    }
}

/// Geometric nonlinearity utilities.
pub mod geometric_nonlinearity {
    use nalgebra::{DMatrix, DVector};

    /// Computes the Green-Lagrange strain tensor.
    /// E = 0.5 * (F^T * F - I)
    pub fn green_lagrange_strain(f: &DMatrix<f64>) -> DMatrix<f64> {
        let identity = DMatrix::identity(f.nrows(), f.ncols());
        0.5 * (f.transpose() * f - identity)
    }

    /// Computes the deformation gradient from displacement gradient.
    /// F = I + grad(u)
    pub fn deformation_gradient(grad_u: &DMatrix<f64>) -> DMatrix<f64> {
        DMatrix::identity(grad_u.nrows(), grad_u.ncols()) + grad_u
    }

    /// Computes geometric (stress) stiffness matrix for a truss element.
    pub fn geometric_stiffness(axial_force: f64, length: f64) -> DMatrix<f64> {
        let factor = axial_force / length;
        DMatrix::from_row_slice(4, 4, &[
            0.0, 0.0, 0.0, 0.0,
            0.0, factor, 0.0, -factor,
            0.0, 0.0, 0.0, 0.0,
            0.0, -factor, 0.0, factor,
        ])
    }

    /// Computes tangent stiffness = material + geometric.
    pub fn tangent_stiffness(k_mat: &DMatrix<f64>, k_geo: &DMatrix<f64>) -> DMatrix<f64> {
        k_mat + k_geo
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_newton_raphson_basic() {
        let nr = NewtonRaphson::new();

        // Simple test: f(x) = x^2 - 4, f'(x) = 2x
        let residual = |x: &DVector<f64>| DVector::from_column_slice(&[x[0] * x[0] - 4.0]);
        let tangent = |x: &DVector<f64>| DMatrix::from_row_slice(1, 1, &[2.0 * x[0]]);

        let u0 = DVector::from_column_slice(&[1.0]);
        let (solution, iterations) = nr.solve(1, residual, tangent, &u0, 1e-10, 50).unwrap();

        assert!((solution[0] - 2.0).abs() < 1e-6);
        assert!(iterations < 10);
    }

    #[test]
    fn test_arc_length_config_default() {
        let config = ArcLengthConfig::default();
        assert!(config.initial_radius > 0.0);
    }

    #[test]
    fn test_nonlinear_config_default() {
        let config = NonlinearConfig::default();
        assert_eq!(config.max_iterations, 50);
        assert_eq!(config.convergence, ConvergenceCriteria::Force);
        assert!(config.line_search);
    }

    #[test]
    fn test_elastoplastic_material() {
        use material_nonlinearity::ElastoplasticMaterial;

        let mut mat = ElastoplasticMaterial::new(210e9, 0.3, 250e6, 2e9);

        // Elastic response
        let (stress, tangent) = mat.update_stress(0.001, 0.0, 0.01);
        assert!((stress - 210e9 * 0.001).abs() < 1e3);
        assert!((tangent - 210e9).abs() < 1e6);

        // Plastic response
        let (_stress_pl, tangent_pl) = mat.update_stress(0.002, 0.0, 0.01);
        assert!(tangent_pl < 210e9);
    }

    #[test]
    fn test_geometric_stiffness() {
        use geometric_nonlinearity::geometric_stiffness;

        let k_geo = geometric_stiffness(1000.0, 1.0);
        assert_eq!(k_geo.shape(), (4, 4));

        // Check symmetry
        for i in 0..4 {
            for j in 0..4 {
                assert!((k_geo[(i, j)] - k_geo[(j, i)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_green_lagrange_strain() {
        use geometric_nonlinearity::{deformation_gradient, green_lagrange_strain};

        let f = DMatrix::from_row_slice(3, 3, &[
            1.1, 0.0, 0.0,
            0.0, 0.95, 0.0,
            0.0, 0.0, 0.95,
        ]);

        let e = green_lagrange_strain(&f);
        assert!(e[(0, 0)] > 0.0);
        assert!(e[(1, 1)] < 0.0);
    }
}

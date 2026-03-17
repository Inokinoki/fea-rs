//! Explicit dynamic analysis for FEA.
#![allow(unused_variables)]
//!
//! This module provides:
//! - Central difference method
//! - Runge-Kutta explicit integrators
//! - Mass lumping for explicit methods
//! - Critical time step estimation
//! - Shock and impact analysis
//! - Energy conservation monitoring

use nalgebra::{DMatrix, DVector};

/// Explicit time integration method.
#[derive(Debug, Clone, Copy)]
pub enum ExplicitMethod {
    /// Central difference method (conditionally stable).
    CentralDifference,
    /// Forward Euler (first order).
    ForwardEuler,
    /// Runge-Kutta 4th order.
    RungeKutta4,
    /// Explicit Newmark (beta = 0).
    ExplicitNewmark,
}

/// Explicit dynamic analysis configuration.
#[derive(Debug, Clone)]
pub struct ExplicitConfig {
    /// Time integration method.
    pub method: ExplicitMethod,
    /// Time step size.
    pub time_step: f64,
    /// Total analysis time.
    pub total_time: f64,
    /// Damping ratio (Rayleigh).
    pub damping_alpha: f64,
    /// Damping ratio (Rayleigh).
    pub damping_beta: f64,
    /// Enable automatic time step.
    pub auto_time_step: bool,
    /// Output frequency (steps between outputs).
    pub output_frequency: usize,
}

impl Default for ExplicitConfig {
    fn default() -> Self {
        Self {
            method: ExplicitMethod::CentralDifference,
            time_step: 1e-4,
            total_time: 1.0,
            damping_alpha: 0.0,
            damping_beta: 0.0,
            auto_time_step: true,
            output_frequency: 10,
        }
    }
}

/// Result of explicit dynamic analysis.
#[derive(Debug, Clone)]
pub struct ExplicitDynamicResult {
    /// Displacement history (time x DOFs).
    pub displacement_history: Vec<Vec<f64>>,
    /// Velocity history.
    pub velocity_history: Vec<Vec<f64>>,
    /// Acceleration history.
    pub acceleration_history: Vec<Vec<f64>>,
    /// Time points.
    pub time_points: Vec<f64>,
    /// Kinetic energy history.
    pub kinetic_energy: Vec<f64>,
    /// Strain energy history.
    pub strain_energy: Vec<f64>,
    /// Total energy history.
    pub total_energy: Vec<f64>,
    /// Number of time steps.
    pub num_steps: usize,
    /// Final time.
    pub final_time: f64,
}

/// Explicit dynamic analyzer.
pub struct ExplicitDynamicAnalyzer {
    config: ExplicitConfig,
    /// Mass matrix (lumped for explicit).
    mass: DMatrix<f64>,
    /// Stiffness matrix.
    stiffness: DMatrix<f64>,
}

impl ExplicitDynamicAnalyzer {
    /// Creates a new explicit dynamic analyzer.
    pub fn new(mass: DMatrix<f64>, stiffness: DMatrix<f64>) -> Self {
        Self {
            mass,
            stiffness,
            config: ExplicitConfig::default(),
        }
    }

    /// Creates with custom configuration.
    pub fn with_config(mass: DMatrix<f64>, stiffness: DMatrix<f64>, config: ExplicitConfig) -> Self {
        Self { mass, stiffness, config }
    }

    /// Estimates critical time step for stability.
    pub fn estimate_critical_time_step(&self) -> f64 {
        // dt_critical = 2 / omega_max
        // For lumped mass: omega_max^2 = k_max / m_min

        let n = self.mass.nrows();
        let mut k_max = 0.0;
        let mut m_min = f64::INFINITY;

        for i in 0..n {
            let m_ii = self.mass[(i, i)];
            if m_ii > 0.0 && m_ii < m_min {
                m_min = m_ii;
            }

            let k_ii = self.stiffness[(i, i)];
            if k_ii > k_max {
                k_max = k_ii;
            }
        }

        if m_min > 0.0 && m_min < f64::INFINITY {
            let omega_max = (k_max / m_min).sqrt();
            2.0 / omega_max * 0.9 // Safety factor
        } else {
            1e-6 // Default
        }
    }

    /// Creates lumped mass matrix from consistent mass.
    pub fn lump_mass(consistent_mass: &DMatrix<f64>) -> DMatrix<f64> {
        let n = consistent_mass.nrows();
        let mut lumped = DMatrix::zeros(n, n);

        // HRZ lumping: scale diagonal to preserve total mass
        let mut total_mass = 0.0;
        let mut sum_diag = 0.0;

        for i in 0..n {
            total_mass += consistent_mass.column(i).sum();
            sum_diag += consistent_mass[(i, i)].abs();
        }

        let scale = if sum_diag > 0.0 { total_mass / sum_diag } else { 1.0 };

        for i in 0..n {
            lumped[(i, i)] = consistent_mass[(i, i)].abs() * scale;
        }

        lumped
    }

    /// Performs explicit dynamic analysis.
    pub fn analyze(
        &self,
        u0: &[f64],
        v0: &[f64],
        external_force: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    ) -> anyhow::Result<ExplicitDynamicResult> {
        let n = u0.len();
        let dt = if self.config.auto_time_step {
            self.estimate_critical_time_step()
        } else {
            self.config.time_step
        };

        let num_steps = (self.config.total_time / dt).ceil() as usize;

        // Initialize
        let mut u = Vec::from(u0);
        let mut v = Vec::from(v0);

        // Initial acceleration: a = M^{-1}(F - Ku)
        let ku = &self.stiffness * &DVector::from_column_slice(&u);
        let f0 = external_force(0.0, &u);
        let mut a = Vec::with_capacity(n);

        for i in 0..n {
            let m_ii = self.mass[(i, i)].max(1e-15);
            let f_ext = if i < f0.len() { f0[i] } else { 0.0 };
            a.push((f_ext - ku[i]) / m_ii);
        }

        // Storage
        let mut u_history = Vec::with_capacity(num_steps / self.config.output_frequency + 1);
        let mut v_history = Vec::with_capacity(num_steps / self.config.output_frequency + 1);
        let mut a_history = Vec::with_capacity(num_steps / self.config.output_frequency + 1);
        let mut t_history = Vec::with_capacity(num_steps / self.config.output_frequency + 1);
        let mut ke_history = Vec::new();
        let mut se_history = Vec::new();
        let mut te_history = Vec::new();

        // Store initial state
        u_history.push(u.clone());
        v_history.push(v.clone());
        a_history.push(a.clone());
        t_history.push(0.0);
        ke_history.push(self.compute_kinetic_energy(&v));
        se_history.push(self.compute_strain_energy(&u));
        te_history.push(ke_history[0] + se_history[0]);

        // Time integration
        match self.config.method {
            ExplicitMethod::CentralDifference => {
                self.central_difference_step(&mut u, &mut v, &mut a, dt, external_force,
                    &mut u_history, &mut v_history, &mut a_history, &mut t_history,
                    &mut ke_history, &mut se_history, &mut te_history, num_steps);
            }
            ExplicitMethod::RungeKutta4 => {
                self.rk4_step(&mut u, &mut v, &mut a, dt, external_force,
                    &mut u_history, &mut v_history, &mut a_history, &mut t_history,
                    &mut ke_history, &mut se_history, &mut te_history, num_steps);
            }
            ExplicitMethod::ForwardEuler => {
                self.forward_euler_step(&mut u, &mut v, &mut a, dt, external_force,
                    &mut u_history, &mut v_history, &mut a_history, &mut t_history,
                    &mut ke_history, &mut se_history, &mut te_history, num_steps);
            }
            ExplicitMethod::ExplicitNewmark => {
                self.explicit_newmark_step(&mut u, &mut v, &mut a, dt, external_force,
                    &mut u_history, &mut v_history, &mut a_history, &mut t_history,
                    &mut ke_history, &mut se_history, &mut te_history, num_steps);
            }
        }

        let final_time = t_history.last().copied().unwrap_or(0.0);

        Ok(ExplicitDynamicResult {
            displacement_history: u_history,
            velocity_history: v_history,
            acceleration_history: a_history,
            time_points: t_history,
            kinetic_energy: ke_history,
            strain_energy: se_history,
            total_energy: te_history,
            num_steps,
            final_time,
        })
    }

    /// Central difference method step.
    fn central_difference_step(
        &self,
        u: &mut Vec<f64>,
        v: &mut Vec<f64>,
        a: &mut Vec<f64>,
        dt: f64,
        external_force: &dyn Fn(f64, &[f64]) -> Vec<f64>,
        u_history: &mut Vec<Vec<f64>>,
        v_history: &mut Vec<Vec<f64>>,
        a_history: &mut Vec<Vec<f64>>,
        t_history: &mut Vec<f64>,
        ke_history: &mut Vec<f64>,
        se_history: &mut Vec<f64>,
        te_history: &mut Vec<f64>,
        num_steps: usize,
    ) {
        let n = u.len();
        let dt2 = dt * dt;
        let damping = self.config.damping_alpha > 0.0 || self.config.damping_beta > 0.0;

        for step in 1..=num_steps {
            let t = step as f64 * dt;

            // Update velocity (half step)
            for i in 0..n {
                let c = if damping {
                    self.config.damping_alpha * self.mass[(i, i)] + self.config.damping_beta * self.stiffness[(i, i)]
                } else {
                    0.0
                };
                v[i] += (a[i] - c * v[i] / self.mass[(i, i)].max(1e-15)) * dt / 2.0;
            }

            // Update displacement
            for i in 0..n {
                u[i] += v[i] * dt;
            }

            // Compute new acceleration
            let ku = &self.stiffness * &DVector::from_column_slice(u);
            let f_ext = external_force(t, u);

            for i in 0..n {
                let c = if damping {
                    self.config.damping_alpha * self.mass[(i, i)] + self.config.damping_beta * self.stiffness[(i, i)]
                } else {
                    0.0
                };
                let f_ext_i = if i < f_ext.len() { f_ext[i] } else { 0.0 };
                a[i] = (f_ext_i - ku[i] - c * v[i]) / self.mass[(i, i)].max(1e-15);
            }

            // Update velocity (second half step)
            for i in 0..n {
                let c = if damping {
                    self.config.damping_alpha * self.mass[(i, i)] + self.config.damping_beta * self.stiffness[(i, i)]
                } else {
                    0.0
                };
                v[i] += (a[i] - c * v[i] / self.mass[(i, i)].max(1e-15)) * dt / 2.0;
            }

            // Store results
            if step % self.config.output_frequency == 0 {
                u_history.push(u.clone());
                v_history.push(v.clone());
                a_history.push(a.clone());
                t_history.push(t);
                ke_history.push(self.compute_kinetic_energy(v));
                se_history.push(self.compute_strain_energy(u));
                te_history.push(ke_history.last().copied().unwrap_or(0.0) + se_history.last().copied().unwrap_or(0.0));
            }
        }
    }

    /// Forward Euler step.
    fn forward_euler_step(
        &self,
        u: &mut Vec<f64>,
        v: &mut Vec<f64>,
        a: &mut Vec<f64>,
        dt: f64,
        external_force: &dyn Fn(f64, &[f64]) -> Vec<f64>,
        u_history: &mut Vec<Vec<f64>>,
        v_history: &mut Vec<Vec<f64>>,
        a_history: &mut Vec<Vec<f64>>,
        t_history: &mut Vec<f64>,
        ke_history: &mut Vec<f64>,
        se_history: &mut Vec<f64>,
        te_history: &mut Vec<f64>,
        num_steps: usize,
    ) {
        let n = u.len();

        for step in 1..=num_steps {
            let t = step as f64 * dt;

            // Update velocity
            for i in 0..n {
                v[i] += a[i] * dt;
            }

            // Update displacement
            for i in 0..n {
                u[i] += v[i] * dt;
            }

            // Compute new acceleration
            let ku = &self.stiffness * &DVector::from_column_slice(u);
            let f_ext = external_force(t, u);

            for i in 0..n {
                let f_ext_i = if i < f_ext.len() { f_ext[i] } else { 0.0 };
                a[i] = (f_ext_i - ku[i]) / self.mass[(i, i)].max(1e-15);
            }

            if step % self.config.output_frequency == 0 {
                u_history.push(u.clone());
                v_history.push(v.clone());
                a_history.push(a.clone());
                t_history.push(t);
                ke_history.push(self.compute_kinetic_energy(v));
                se_history.push(self.compute_strain_energy(u));
                te_history.push(ke_history.last().copied().unwrap_or(0.0) + se_history.last().copied().unwrap_or(0.0));
            }
        }
    }

    /// Runge-Kutta 4th order step.
    fn rk4_step(
        &self,
        u: &mut Vec<f64>,
        v: &mut Vec<f64>,
        _a: &mut Vec<f64>,
        dt: f64,
        external_force: &dyn Fn(f64, &[f64]) -> Vec<f64>,
        u_history: &mut Vec<Vec<f64>>,
        v_history: &mut Vec<Vec<f64>>,
        a_history: &mut Vec<Vec<f64>>,
        t_history: &mut Vec<f64>,
        ke_history: &mut Vec<f64>,
        se_history: &mut Vec<f64>,
        te_history: &mut Vec<f64>,
        num_steps: usize,
    ) {
        let n = u.len();

        for step in 1..=num_steps {
            let t = step as f64 * dt;

            // RK4 for second order ODE: M*u'' + K*u = F
            // Convert to first order: y = [u, v], y' = [v, M^{-1}(F - Ku)]

            let compute_deriv = |u: &[f64], v: &[f64], t: f64| -> (Vec<f64>, Vec<f64>) {
                let ku = &self.stiffness * &DVector::from_column_slice(u);
                let f_ext = external_force(t, u);

                let u_dot = v.to_vec();
                let mut v_dot = Vec::with_capacity(n);

                for i in 0..n {
                    let f_ext_i = if i < f_ext.len() { f_ext[i] } else { 0.0 };
                    v_dot.push((f_ext_i - ku[i]) / self.mass[(i, i)].max(1e-15));
                }

                (u_dot, v_dot)
            };

            let (u_d1, v_d1) = compute_deriv(u, v, t - dt);
            let u2: Vec<f64> = u.iter().zip(u_d1.iter()).map(|(a, b)| a + b * dt / 2.0).collect();
            let v2: Vec<f64> = v.iter().zip(v_d1.iter()).map(|(a, b)| a + b * dt / 2.0).collect();

            let (u_d2, v_d2) = compute_deriv(&u2, &v2, t - dt / 2.0);
            let u3: Vec<f64> = u.iter().zip(u_d2.iter()).map(|(a, b)| a + b * dt / 2.0).collect();
            let v3: Vec<f64> = v.iter().zip(v_d2.iter()).map(|(a, b)| a + b * dt / 2.0).collect();

            let (u_d3, v_d3) = compute_deriv(&u3, &v3, t - dt / 2.0);
            let u4: Vec<f64> = u.iter().zip(u_d3.iter()).map(|(a, b)| a + b * dt).collect();
            let v4: Vec<f64> = v.iter().zip(v_d3.iter()).map(|(a, b)| a + b * dt).collect();

            let (u_d4, v_d4) = compute_deriv(&u4, &v4, t);

            for i in 0..n {
                u[i] += dt / 6.0 * (u_d1[i] + 2.0 * u_d2[i] + 2.0 * u_d3[i] + u_d4[i]);
                v[i] += dt / 6.0 * (v_d1[i] + 2.0 * v_d2[i] + 2.0 * v_d3[i] + v_d4[i]);
            }

            // Compute acceleration
            let ku = &self.stiffness * &DVector::from_column_slice(u);
            let f_ext = external_force(t, u);
            for i in 0..n {
                let f_ext_i = if i < f_ext.len() { f_ext[i] } else { 0.0 };
                _a[i] = (f_ext_i - ku[i]) / self.mass[(i, i)].max(1e-15);
            }

            if step % self.config.output_frequency == 0 {
                u_history.push(u.clone());
                v_history.push(v.clone());
                a_history.push(_a.clone());
                t_history.push(t);
                ke_history.push(self.compute_kinetic_energy(v));
                se_history.push(self.compute_strain_energy(u));
                te_history.push(ke_history.last().copied().unwrap_or(0.0) + se_history.last().copied().unwrap_or(0.0));
            }
        }
    }

    /// Explicit Newmark step.
    fn explicit_newmark_step(
        &self,
        u: &mut Vec<f64>,
        v: &mut Vec<f64>,
        a: &mut Vec<f64>,
        dt: f64,
        external_force: &dyn Fn(f64, &[f64]) -> Vec<f64>,
        u_history: &mut Vec<Vec<f64>>,
        v_history: &mut Vec<Vec<f64>>,
        a_history: &mut Vec<Vec<f64>>,
        t_history: &mut Vec<f64>,
        ke_history: &mut Vec<f64>,
        se_history: &mut Vec<f64>,
        te_history: &mut Vec<f64>,
        num_steps: usize,
    ) {
        let n = u.len();
        let dt2 = dt * dt;

        // Newmark parameters for explicit (beta = 0, gamma = 0.5)
        let beta = 0.0;
        let gamma = 0.5;

        for step in 1..=num_steps {
            let t = step as f64 * dt;

            // Predict displacement and velocity
            let u_pred: Vec<f64> = u.iter().zip(a.iter())
                .map(|(ui, ai)| ui + v[0] * dt + (0.5 - beta) * ai * dt2)
                .collect();

            let v_pred: Vec<f64> = v.iter().zip(a.iter())
                .map(|(vi, ai)| vi + (1.0 - gamma) * ai * dt)
                .collect();

            // Compute new acceleration
            let ku = &self.stiffness * &DVector::from_column_slice(&u_pred);
            let f_ext = external_force(t, &u_pred);

            for i in 0..n {
                let f_ext_i = if i < f_ext.len() { f_ext[i] } else { 0.0 };
                a[i] = (f_ext_i - ku[i]) / self.mass[(i, i)].max(1e-15);
            }

            // Correct velocity
            for i in 0..n {
                v[i] = v_pred[i] + gamma * a[i] * dt;
                u[i] = u_pred[i] + beta * a[i] * dt2;
            }

            if step % self.config.output_frequency == 0 {
                u_history.push(u.clone());
                v_history.push(v.clone());
                a_history.push(a.clone());
                t_history.push(t);
                ke_history.push(self.compute_kinetic_energy(v));
                se_history.push(self.compute_strain_energy(u));
                te_history.push(ke_history.last().copied().unwrap_or(0.0) + se_history.last().copied().unwrap_or(0.0));
            }
        }
    }

    /// Computes kinetic energy: KE = 0.5 * v^T * M * v
    fn compute_kinetic_energy(&self, v: &[f64]) -> f64 {
        let v_vec = DVector::from_column_slice(v);
        0.5 * v_vec.dot(&(&self.mass * &v_vec))
    }

    /// Computes strain energy: SE = 0.5 * u^T * K * u
    fn compute_strain_energy(&self, u: &[f64]) -> f64 {
        let u_vec = DVector::from_column_slice(u);
        0.5 * u_vec.dot(&(&self.stiffness * &u_vec))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explicit_config_default() {
        let config = ExplicitConfig::default();
        assert!(matches!(config.method, ExplicitMethod::CentralDifference));
        assert!((config.time_step - 1e-4).abs() < 1e-10);
    }

    #[test]
    fn test_mass_lumping() {
        let consistent = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]);
        let lumped = ExplicitDynamicAnalyzer::lump_mass(&consistent);

        assert!((lumped[(0, 0)] - 3.0).abs() < 1e-10);
        assert!((lumped[(1, 1)] - 3.0).abs() < 1e-10);
        assert!((lumped[(0, 1)] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_critical_time_step() {
        let mass = DMatrix::from_diagonal(&DVector::from_element(2, 1.0));
        let stiffness = DMatrix::from_diagonal(&DVector::from_element(2, 100.0));

        let analyzer = ExplicitDynamicAnalyzer::new(mass, stiffness);
        let dt = analyzer.estimate_critical_time_step();

        assert!(dt > 0.0);
        assert!(dt < 1.0);
    }

    #[test]
    fn test_energy_computation() {
        let mass = DMatrix::from_diagonal(&DVector::from_element(2, 1.0));
        let stiffness = DMatrix::from_diagonal(&DVector::from_element(2, 100.0));

        let analyzer = ExplicitDynamicAnalyzer::new(mass, stiffness);

        let v = vec![1.0, 1.0];
        let ke = analyzer.compute_kinetic_energy(&v);
        assert!((ke - 1.0).abs() < 1e-10);

        let u = vec![1.0, 1.0];
        let se = analyzer.compute_strain_energy(&u);
        assert!((se - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_explicit_dynamic_result() {
        let mass = DMatrix::from_diagonal(&DVector::from_element(2, 1.0));
        let stiffness = DMatrix::from_diagonal(&DVector::from_element(2, 100.0));

        let config = ExplicitConfig {
            method: ExplicitMethod::CentralDifference,
            time_step: 0.01,
            total_time: 0.1,
            ..Default::default()
        };

        let analyzer = ExplicitDynamicAnalyzer::with_config(mass, stiffness, config);
        let u0 = vec![0.0, 0.0];
        let v0 = vec![1.0, 1.0];

        let result = analyzer.analyze(&u0, &v0, &|_t, _u| vec![0.0, 0.0]).unwrap();

        assert!(!result.displacement_history.is_empty());
        assert!(!result.time_points.is_empty());
        assert!(result.num_steps > 0);
    }
}

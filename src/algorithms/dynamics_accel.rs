//! Dynamic analysis acceleration methods.
#![allow(unused_variables)]
//!
//! This module provides:
//! - Modal superposition acceleration
//! - Component mode synthesis
//! - Model order reduction
//! - Dynamic substructuring

use nalgebra::{DMatrix, DVector, SymmetricEigen};

/// Modal acceleration method for dynamic response.
#[derive(Debug, Clone)]
pub struct ModalAcceleration {
    /// Natural frequencies.
    pub frequencies: DVector<f64>,
    /// Mode shapes (columns).
    pub mode_shapes: DMatrix<f64>,
    /// Damping ratios.
    pub damping_ratios: DVector<f64>,
    /// Number of retained modes.
    pub num_modes: usize,
}

impl ModalAcceleration {
    /// Creates a new modal accelerator.
    pub fn new(frequencies: DVector<f64>, mode_shapes: DMatrix<f64>) -> Self {
        let num_modes = frequencies.len();
        let damping_ratios = DVector::from_element(num_modes, 0.02); // Default 2% damping

        Self {
            frequencies,
            mode_shapes,
            damping_ratios,
            num_modes,
        }
    }

    /// Sets damping ratios.
    pub fn with_damping(mut self, ratios: DVector<f64>) -> Self {
        self.damping_ratios = ratios;
        self
    }

    /// Truncates to specified number of modes.
    pub fn truncate(&mut self, num_modes: usize) {
        self.num_modes = num_modes.min(self.frequencies.len());
    }

    /// Computes frequency response function.
    pub fn frequency_response(
        &self,
        omega: f64,
        force_vector: &DVector<f64>,
    ) -> DVector<f64> {
        let n = self.mode_shapes.nrows();
        let mut response = DVector::zeros(n);

        for i in 0..self.num_modes {
            let phi_i = self.mode_shapes.column(i);
            let omega_i = self.frequencies[i];
            let zeta_i = self.damping_ratios[i];

            // Modal participation factor
            let gamma_i = phi_i.dot(force_vector);

            // Frequency response function for mode i
            let denom = (omega_i.powi(2) - omega.powi(2)).powi(2)
                + (2.0 * zeta_i * omega_i * omega).powi(2);

            if denom > 1e-30 {
                let real_part = (omega_i.powi(2) - omega.powi(2)) / denom;
                let imag_part = 2.0 * zeta_i * omega_i * omega / denom;

                let magnitude = (real_part.powi(2) + imag_part.powi(2)).sqrt();
                response += phi_i.scale(gamma_i * magnitude);
            }
        }

        response
    }

    /// Computes transient response using modal superposition.
    pub fn transient_response(
        &self,
        time: f64,
        initial_displacement: &DVector<f64>,
        initial_velocity: &DVector<f64>,
        force_fn: &dyn Fn(f64) -> DVector<f64>,
    ) -> DVector<f64> {
        let n = self.mode_shapes.nrows();
        let mut response = DVector::zeros(n);

        for i in 0..self.num_modes {
            let phi_i = self.mode_shapes.column(i);
            let omega_i = self.frequencies[i];
            let zeta_i = self.damping_ratios[i];
            let omega_d = omega_i * (1.0 - zeta_i.powi(2)).sqrt();

            // Initial modal coordinates
            let q0_i = phi_i.dot(initial_displacement);
            let qdot0_i = phi_i.dot(initial_velocity);

            // Free vibration response
            let exp_factor = (-zeta_i * omega_i * time).exp();
            let q_free = exp_factor * (
                q0_i * (omega_d * time).cos()
                + ((zeta_i * omega_i * q0_i + qdot0_i) / omega_d) * (omega_d * time).sin()
            );

            // Forced response (simplified - assumes constant force)
            let f_i = phi_i.dot(&force_fn(time));
            let q_forced = f_i / omega_i.powi(2);

            response += phi_i.scale(q_free + q_forced);
        }

        response
    }

    /// Computes effective modal mass for each mode.
    pub fn effective_modal_mass(&self, excitation_direction: &DVector<f64>) -> DVector<f64> {
        let mut masses = DVector::zeros(self.num_modes);

        for i in 0..self.num_modes {
            let phi_i = self.mode_shapes.column(i);
            let l_i = phi_i.dot(excitation_direction); // Modal participation
            masses[i] = l_i.powi(2);
        }

        masses
    }

    /// Computes modal mass participation ratio.
    pub fn mass_participation_ratio(
        &self,
        effective_masses: &DVector<f64>,
        threshold: f64,
    ) -> usize {
        let total_mass = effective_masses.sum();
        let mut cumulative = 0.0;

        for i in 0..self.num_modes {
            cumulative += effective_masses[i];
            if cumulative / total_mass >= threshold {
                return i + 1;
            }
        }

        self.num_modes
    }
}

/// Component Mode Synthesis (CMS) for substructuring.
#[derive(Debug, Clone)]
pub struct ComponentModeSynthesis {
    /// Number of boundary DOFs.
    pub num_boundary_dofs: usize,
    /// Number of internal modes retained.
    pub num_internal_modes: usize,
    /// Constraint modes (boundary to interior mapping).
    pub constraint_modes: Option<DMatrix<f64>>,
    /// Fixed-interface normal modes.
    pub normal_modes: Option<DMatrix<f64>>,
    /// Fixed-interface frequencies.
    pub fixed_frequencies: Option<DVector<f64>>,
}

impl ComponentModeSynthesis {
    /// Creates a new CMS reducer.
    pub fn new(num_boundary_dofs: usize, num_internal_modes: usize) -> Self {
        Self {
            num_boundary_dofs,
            num_internal_modes,
            constraint_modes: None,
            normal_modes: None,
            fixed_frequencies: None,
        }
    }

    /// Performs Craig-Bampton reduction.
    pub fn craig_bampton_reduction(
        &mut self,
        k: &DMatrix<f64>,
        m: &DMatrix<f64>,
        boundary_dofs: &[usize],
    ) -> (DMatrix<f64>, DMatrix<f64>) {
        let n_total = k.nrows();
        let n_internal = n_total - boundary_dofs.len();

        if n_internal == 0 {
            // No internal DOFs - return original matrices
            return (k.clone(), m.clone());
        }

        // Partition DOFs
        let mut internal_dofs = Vec::new();
        for i in 0..n_total {
            if !boundary_dofs.contains(&i) {
                internal_dofs.push(i);
            }
        }

        // Extract submatrices (simplified - would need proper indexing)
        let k_bb = self.extract_submatrix(k, boundary_dofs, boundary_dofs);
        let k_bi = self.extract_submatrix(k, boundary_dofs, &internal_dofs);
        let k_ii = self.extract_submatrix(k, &internal_dofs, &internal_dofs);

        let m_bb = self.extract_submatrix(m, boundary_dofs, boundary_dofs);
        let m_ii = self.extract_submatrix(m, &internal_dofs, &internal_dofs);

        // Constraint modes: Psi = -K_ii^{-1} * K_ib
        let constraint_modes = if let Some(k_ii_inv) = k_ii.clone().try_inverse() {
            let k_ib = self.extract_submatrix(k, &internal_dofs, boundary_dofs);
            k_ii_inv * k_ib * -1.0
        } else {
            DMatrix::zeros(n_internal, boundary_dofs.len())
        };

        self.constraint_modes = Some(constraint_modes.clone());

        // Fixed-interface modes (eigenvalue problem)
        let eigen = SymmetricEigen::new(k_ii.clone());
        let num_modes = self.num_internal_modes.min(eigen.eigenvalues.len());

        let mut phi = DMatrix::zeros(n_internal, num_modes);
        let mut frequencies = DVector::zeros(num_modes);

        for i in 0..num_modes {
            frequencies[i] = eigen.eigenvalues[i].sqrt();
            for j in 0..n_internal {
                phi[(j, i)] = eigen.eigenvectors[(j, i)];
            }
        }

        self.normal_modes = Some(phi.clone());
        self.fixed_frequencies = Some(frequencies.clone());

        // Build transformation matrix
        let n_reduced = boundary_dofs.len() + num_modes;
        let mut t = DMatrix::zeros(n_total, n_reduced);

        // Boundary DOFs
        for (i, &dof) in boundary_dofs.iter().enumerate() {
            t[(dof, i)] = 1.0;
        }

        // Internal DOFs - constraint modes + normal modes
        for (i_internal, &dof) in internal_dofs.iter().enumerate() {
            for (i_boundary, _) in boundary_dofs.iter().enumerate() {
                t[(dof, i_boundary)] = constraint_modes[(i_internal, i_boundary)];
            }
            for i_mode in 0..num_modes {
                t[(dof, boundary_dofs.len() + i_mode)] = phi[(i_internal, i_mode)];
            }
        }

        // Reduced matrices
        let k_reduced = t.transpose() * k * &t;
        let m_reduced = t.transpose() * m * &t;

        (k_reduced, m_reduced)
    }

    fn extract_submatrix(
        &self,
        matrix: &DMatrix<f64>,
        rows: &[usize],
        cols: &[usize],
    ) -> DMatrix<f64> {
        let mut sub = DMatrix::zeros(rows.len(), cols.len());
        for (i, &r) in rows.iter().enumerate() {
            for (j, &c) in cols.iter().enumerate() {
                sub[(i, j)] = matrix[(r, c)];
            }
        }
        sub
    }

    /// Applies Guyan reduction (static condensation).
    pub fn guyan_reduction(
        &self,
        k: &DMatrix<f64>,
        m: &DMatrix<f64>,
        master_dofs: &[usize],
    ) -> (DMatrix<f64>, DMatrix<f64>) {
        let n_total = k.nrows();
        let mut slave_dofs = Vec::new();

        for i in 0..n_total {
            if !master_dofs.contains(&i) {
                slave_dofs.push(i);
            }
        }

        let k_mm = self.extract_submatrix(k, master_dofs, master_dofs);
        let k_ms = self.extract_submatrix(k, master_dofs, &slave_dofs);
        let k_ss = self.extract_submatrix(k, &slave_dofs, &slave_dofs);

        let m_mm = self.extract_submatrix(m, master_dofs, master_dofs);
        let m_ss = self.extract_submatrix(m, &slave_dofs, &slave_dofs);

        // Guyan transformation
        let t_ms = if let Some(k_ss_inv) = k_ss.clone().try_inverse() {
            k_ss_inv * k_ms.transpose() * -1.0
        } else {
            DMatrix::zeros(slave_dofs.len(), master_dofs.len())
        };

        // Build transformation
        let mut t = DMatrix::zeros(n_total, master_dofs.len());
        for (i, &dof) in master_dofs.iter().enumerate() {
            t[(dof, i)] = 1.0;
        }
        for (i_s, &slave) in slave_dofs.iter().enumerate() {
            for (i_m, _) in master_dofs.iter().enumerate() {
                t[(slave, i_m)] = t_ms[(i_s, i_m)];
            }
        }

        let k_reduced = t.transpose() * k * &t;
        let m_reduced = t.transpose() * m * &t;

        (k_reduced, m_reduced)
    }
}

/// Proper Orthogonal Decomposition (POD) for model reduction.
#[derive(Debug, Clone)]
pub struct ProperOrthogonalDecomposition {
    /// POD modes (columns).
    pub modes: DMatrix<f64>,
    /// Singular values.
    pub singular_values: DVector<f64>,
    /// Energy captured by retained modes.
    pub energy_captured: f64,
}

impl ProperOrthogonalDecomposition {
    /// Creates POD from snapshot matrix.
    pub fn from_snapshots(snapshots: &DMatrix<f64>, energy_threshold: f64) -> Self {
        let (n_snapshots, _) = snapshots.shape();

        // Compute correlation matrix
        let correlation = snapshots.transpose() * snapshots;

        // Eigenvalue decomposition
        let eigen = SymmetricEigen::new(correlation);

        // Compute total energy
        let total_energy: f64 = eigen.eigenvalues.iter().map(|&e| e.max(0.0)).sum();

        // Determine number of modes to retain
        let mut cumulative_energy = 0.0;
        let mut num_modes = 0;

        for (i, &eigenvalue) in eigen.eigenvalues.iter().enumerate() {
            if eigenvalue > 0.0 {
                cumulative_energy += eigenvalue;
            }
            if cumulative_energy / total_energy >= energy_threshold {
                num_modes = i + 1;
                break;
            }
        }

        num_modes = num_modes.max(1).min(eigen.eigenvalues.len());

        // Compute POD modes
        let mut modes = DMatrix::zeros(n_snapshots, num_modes);
        let mut singular_values = DVector::zeros(num_modes);

        for i in 0..num_modes {
            let sigma = eigen.eigenvalues[i].sqrt().max(0.0);
            singular_values[i] = sigma;

            if sigma > 1e-10 {
                let snapshot_contribution = snapshots * eigen.eigenvectors.column(i);
                for j in 0..n_snapshots {
                    modes[(j, i)] = snapshot_contribution[j] / sigma;
                }
            }
        }

        let energy_captured = cumulative_energy / total_energy;

        Self {
            modes,
            singular_values,
            energy_captured,
        }
    }

    /// Projects system onto POD subspace.
    pub fn project_system(
        &self,
        k: &DMatrix<f64>,
        m: &DMatrix<f64>,
    ) -> (DMatrix<f64>, DMatrix<f64>) {
        let k_reduced = self.modes.transpose() * k * &self.modes;
        let m_reduced = self.modes.transpose() * m * &self.modes;

        (k_reduced, m_reduced)
    }

    /// Reconstructs full solution from reduced coordinates.
    pub fn reconstruct(&self, q_reduced: &DVector<f64>) -> DVector<f64> {
        &self.modes * q_reduced
    }
}

/// Krylov subspace model order reduction.
pub struct KrylovReduction {
    /// Projection matrix.
    pub v: DMatrix<f64>,
    /// Order of reduction.
    pub order: usize,
}

impl KrylovReduction {
    /// Creates Krylov reduction matrix.
    pub fn new(a: &DMatrix<f64>, b: &DVector<f64>, order: usize) -> Self {
        let n = a.nrows();
        let k = order.min(n);

        // Arnoldi process for Krylov subspace
        let mut v = vec![b.normalize()];
        let mut h = vec![vec![0.0f64; 0]; k + 1];

        for j in 0..k {
            let mut w = a * &v[j];

            // Gram-Schmidt orthogonalization
            let mut h_col = vec![0.0f64; j + 1];
            for i in 0..=j {
                h_col[i] = v[i].dot(&w);
                w -= v[i].scale(h_col[i]);
            }

            let h_next = w.norm();

            for (i, &h_val) in h_col.iter().enumerate() {
                h[i].push(h_val);
            }
            if j < k {
                h[j + 1].push(h_next);
            }

            if h_next > 1e-10 && j < k - 1 {
                v.push(w.scale(1.0 / h_next));
            } else {
                break;
            }
        }

        // Build projection matrix
        let actual_k = v.len();
        let mut v_matrix = DMatrix::zeros(n, actual_k);

        for (j, v_j) in v.iter().enumerate() {
            for i in 0..n {
                v_matrix[(i, j)] = v_j[i];
            }
        }

        Self {
            v: v_matrix,
            order: actual_k,
        }
    }

    /// Reduces system matrices.
    pub fn reduce(&self, a: &DMatrix<f64>, b: &DVector<f64>) -> (DMatrix<f64>, DVector<f64>) {
        let a_reduced = self.v.transpose() * a * &self.v;
        let b_reduced = self.v.transpose() * b;

        (a_reduced, b_reduced)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modal_acceleration() {
        // Simple 2-DOF system
        let frequencies = DVector::from_column_slice(&[10.0, 20.0]);
        let mode_shapes = DMatrix::from_column_slice(2, 2, &[
            1.0, 1.0,
            1.0, -1.0,
        ]);

        let modal = ModalAcceleration::new(frequencies, mode_shapes);

        let force = DVector::from_column_slice(&[1.0, 0.0]);
        let response = modal.frequency_response(15.0, &force);

        assert_eq!(response.len(), 2);
        assert!(response.iter().all(|v| v.is_finite()));

        // Test mass participation
        let direction = DVector::from_column_slice(&[1.0, 1.0]);
        let eff_masses = modal.effective_modal_mass(&direction);
        assert_eq!(eff_masses.len(), 2);
    }

    #[test]
    fn test_craig_bampton() {
        let mut cms = ComponentModeSynthesis::new(2, 3);

        // Simple 4-DOF system
        let k = DMatrix::from_row_slice(4, 4, &[
            2.0, -1.0, 0.0, 0.0,
            -1.0, 2.0, -1.0, 0.0,
            0.0, -1.0, 2.0, -1.0,
            0.0, 0.0, -1.0, 2.0,
        ]);
        let m = DMatrix::identity(4, 4);
        let boundary = vec![0, 3];

        let (k_r, m_r) = cms.craig_bampton_reduction(&k, &m, &boundary);

        assert!(k_r.nrows() > 0);
        assert_eq!(k_r.nrows(), m_r.nrows());
    }

    #[test]
    fn test_pod() {
        // Create snapshot matrix
        let snapshots = DMatrix::from_row_slice(10, 5, &[
            1.0, 1.1, 0.9, 1.0, 1.0,
            2.0, 2.1, 1.9, 2.0, 2.0,
            1.0, 1.1, 0.9, 1.0, 1.0,
            0.5, 0.55, 0.45, 0.5, 0.5,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ]);

        let pod = ProperOrthogonalDecomposition::from_snapshots(&snapshots, 0.99);

        assert!(pod.modes.ncols() > 0);
        assert!(pod.energy_captured >= 0.99 || pod.modes.ncols() == snapshots.ncols());
    }

    #[test]
    fn test_krylov_reduction() {
        let a = DMatrix::from_row_slice(5, 5, &[
            4.0, -1.0, 0.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0, 0.0,
            0.0, -1.0, 4.0, -1.0, 0.0,
            0.0, 0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, 0.0, -1.0, 4.0,
        ]);
        let b = DVector::from_element(5, 1.0);

        let krylov = KrylovReduction::new(&a, &b, 3);

        assert!(krylov.order > 0);
        assert!(krylov.order <= 3);

        let (a_r, b_r) = krylov.reduce(&a, &b);

        assert_eq!(a_r.nrows(), krylov.order);
        assert_eq!(b_r.len(), krylov.order);
    }
}

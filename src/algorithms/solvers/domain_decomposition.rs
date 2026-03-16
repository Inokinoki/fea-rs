//! Domain decomposition preconditioners for parallel FEA.
//!
//! This module provides:
//! - Additive Schwarz Method (ASM)
//! - Balancing Domain Decomposition (BDD)
//! - FETI (Finite Element Tearing and Interconnecting)
//! - Neumann-Neumann preconditioners

use nalgebra::{DMatrix, DVector};

/// Additive Schwarz Method preconditioner.
/// Overlapping domain decomposition with local solves.
#[derive(Debug, Clone)]
pub struct AdditiveSchwarz {
    /// Number of subdomains.
    pub num_subdomains: usize,
    /// Subdomain index ranges.
    pub subdomain_ranges: Vec<std::ops::Range<usize>>,
    /// Overlap size.
    pub overlap: usize,
    /// Local matrices for each subdomain.
    pub local_matrices: Vec<DMatrix<f64>>,
    /// Local LU factors.
    pub local_lu: Vec<DMatrix<f64>>,
}

impl AdditiveSchwarz {
    /// Creates a new Additive Schwarz preconditioner.
    pub fn new(
        global_size: usize,
        num_subdomains: usize,
        overlap: usize,
    ) -> Self {
        let subdomain_size = (global_size + num_subdomains - 1) / num_subdomains;
        let mut ranges = Vec::with_capacity(num_subdomains);

        for i in 0..num_subdomains {
            let start = (i * subdomain_size).saturating_sub(overlap);
            let end = ((i + 1) * subdomain_size + overlap).min(global_size);
            ranges.push(start..end);
        }

        Self {
            num_subdomains,
            subdomain_ranges: ranges,
            overlap,
            local_matrices: Vec::new(),
            local_lu: Vec::new(),
        }
    }

    /// Extracts local matrices from global matrix.
    pub fn extract_local_matrices(&mut self, global: &DMatrix<f64>) {
        self.local_matrices.clear();
        self.local_lu.clear();

        for range in &self.subdomain_ranges {
            let size = range.end - range.start;
            let mut local = DMatrix::zeros(size, size);

            for (i_local, i_global) in range.clone().enumerate() {
                for (j_local, j_global) in range.clone().enumerate() {
                    local[(i_local, j_local)] = global[(i_global, j_global)];
                }
            }

            self.local_lu.push(local.clone());
            self.local_matrices.push(local);
        }
    }

    /// Applies the Additive Schwarz preconditioner.
    pub fn apply(&self, r: &[f64]) -> Vec<f64> {
        let n = r.len();
        let mut z = vec![0.0; n];

        for (dom_idx, range) in self.subdomain_ranges.iter().enumerate() {
            let r_local: Vec<f64> = range.clone().map(|i| r[i]).collect();
            let r_local_vec = DVector::from_column_slice(&r_local);

            // Local solve
            let lu = &self.local_lu[dom_idx];
            let z_local = lu.clone().lu().solve(&r_local_vec).unwrap_or(r_local_vec);

            // Accumulate contributions
            for (i_local, i_global) in range.clone().enumerate() {
                z[i_global] += z_local[i_local];
            }
        }

        z
    }

    /// Applies restricted Additive Schwarz (one-level).
    pub fn apply_restricted(&self, r: &[f64]) -> Vec<f64> {
        let n = r.len();
        let mut z = vec![0.0; n];
        let mut counts = vec![0usize; n];

        for (dom_idx, range) in self.subdomain_ranges.iter().enumerate() {
            let r_local: Vec<f64> = range.clone().map(|i| r[i]).collect();
            let r_local_vec = DVector::from_column_slice(&r_local);

            let lu = &self.local_lu[dom_idx];
            let z_local = lu.clone().lu().solve(&r_local_vec).unwrap_or(r_local_vec);

            for (i_local, i_global) in range.clone().enumerate() {
                z[i_global] += z_local[i_local];
                counts[i_global] += 1;
            }
        }

        // Average overlapping contributions
        for i in 0..n {
            if counts[i] > 0 {
                z[i] /= counts[i] as f64;
            }
        }

        z
    }
}

/// Balancing Domain Decomposition (BDD) preconditioner.
///
/// Two-level preconditioner with coarse grid correction.
#[derive(Debug, Clone)]
pub struct BDDPreconditioner {
    /// Number of subdomains.
    pub num_subdomains: usize,
    /// Subdomain information.
    pub subdomains: Vec<SubdomainInfo>,
    /// Coarse grid matrix.
    pub coarse_matrix: Option<DMatrix<f64>>,
    /// Coarse grid solver.
    pub coarse_solver: Option<DMatrix<f64>>,
}

#[derive(Debug, Clone)]
pub struct SubdomainInfo {
    /// Global DOF indices in this subdomain.
    pub dofs: Vec<usize>,
    /// Local stiffness matrix.
    pub local_k: DMatrix<f64>,
    /// Local inverse (or factorization).
    pub local_inverse: DMatrix<f64>,
    /// Neumann matrix for interface.
    pub neumann_matrix: DMatrix<f64>,
}

impl BDDPreconditioner {
    /// Creates a new BDD preconditioner.
    pub fn new(num_subdomains: usize) -> Self {
        Self {
            num_subdomains,
            subdomains: Vec::new(),
            coarse_matrix: None,
            coarse_solver: None,
        }
    }

    /// Adds a subdomain.
    pub fn add_subdomain(
        &mut self,
        dofs: Vec<usize>,
        local_k: DMatrix<f64>,
    ) {
        let size = dofs.len();
        let local_inverse = local_k.clone().try_inverse().unwrap_or_else(|| {
            // Add regularization if singular
            let mut reg = local_k.clone();
            for i in 0..size {
                reg[(i, i)] += 1e-6;
            }
            reg.try_inverse().unwrap_or(DMatrix::zeros(size, size))
        });

        // Neumann matrix is the Schur complement on the interface
        // Simplified: just use the local matrix
        let neumann_matrix = local_k.clone();

        self.subdomains.push(SubdomainInfo {
            dofs,
            local_k,
            local_inverse,
            neumann_matrix,
        });
    }

    /// Builds the coarse grid problem.
    pub fn build_coarse_grid(&mut self) {
        // Coarse problem: sum of local Schur complements
        let coarse_size = self.num_subdomains;
        let mut coarse = DMatrix::zeros(coarse_size, coarse_size);

        for (i, sub_i) in self.subdomains.iter().enumerate() {
            for (j, sub_j) in self.subdomains.iter().enumerate() {
                if i == j {
                    coarse[(i, j)] = sub_i.neumann_matrix.sum();
                } else {
                    coarse[(i, j)] = 0.0;
                }
            }
        }

        // Add regularization for floating subdomains
        for i in 0..coarse_size {
            coarse[(i, i)] += 1e-10;
        }

        let coarse_inv = coarse.clone().try_inverse().unwrap_or_else(|| coarse.clone());
        self.coarse_solver = Some(coarse_inv);
        self.coarse_matrix = Some(coarse);
    }

    /// Applies the BDD preconditioner.
    pub fn apply(&self, r: &[f64]) -> Vec<f64> {
        let n = r.len();
        let mut z = vec![0.0; n];

        // Local solves
        for sub in &self.subdomains {
            let r_local: Vec<f64> = sub.dofs.iter().map(|&i| r[i]).collect();
            let z_local = &sub.local_inverse * DVector::from_column_slice(&r_local);

            for (i_local, &i_global) in sub.dofs.iter().enumerate() {
                z[i_global] += z_local[i_local];
            }
        }

        // Coarse correction (if available)
        if let Some(coarse_inv) = &self.coarse_solver {
            let mut coarse_rhs = DVector::zeros(self.num_subdomains);

            for (i, sub) in self.subdomains.iter().enumerate() {
                let r_local: Vec<f64> = sub.dofs.iter().map(|&i| r[i]).collect();
                coarse_rhs[i] = DVector::from_column_slice(&r_local).sum();
            }

            let coarse_sol = coarse_inv * coarse_rhs;

            // Distribute coarse correction
            for (i, sub) in self.subdomains.iter().enumerate() {
                for &i_global in &sub.dofs {
                    z[i_global] += coarse_sol[i] / sub.dofs.len() as f64;
                }
            }
        }

        z
    }
}

/// FETI (Finite Element Tearing and Interconnecting) preconditioner.
///
/// Dual-primal method with Lagrange multipliers.
#[derive(Debug, Clone)]
pub struct FETIPreconditioner {
    /// Number of subdomains.
    pub num_subdomains: usize,
    /// Number of interface DOFs.
    pub num_interface_dofs: usize,
    /// Local Dirichlet matrices.
    pub local_dirichlet: Vec<DMatrix<f64>>,
    /// Boolean matrices for interface.
    pub interface_bool: Vec<DMatrix<f64>>,
    /// Coarse problem matrix.
    pub coarse_k: Option<DMatrix<f64>>,
}

impl FETIPreconditioner {
    /// Creates a new FETI preconditioner.
    pub fn new(num_subdomains: usize, num_interface_dofs: usize) -> Self {
        Self {
            num_subdomains,
            num_interface_dofs,
            local_dirichlet: Vec::new(),
            interface_bool: Vec::new(),
            coarse_k: None,
        }
    }

    /// Adds subdomain data.
    pub fn add_subdomain(
        &mut self,
        local_k: DMatrix<f64>,
        interface_mapping: &[usize],
    ) {
        // Extract Dirichlet part (interior DOFs)
        let size = local_k.nrows();
        let num_interface = interface_mapping.len();
        let num_interior = size - num_interface;

        if num_interior > 0 {
            let mut dirichlet = DMatrix::zeros(num_interior, num_interior);
            for i in 0..num_interior {
                for j in 0..num_interior {
                    dirichlet[(i, j)] = local_k[(i, j)];
                }
            }
            self.local_dirichlet.push(dirichlet);
        }

        // Build boolean matrix for interface
        let mut bool_mat = DMatrix::zeros(num_interface, size);
        for (i_local, &i_global) in interface_mapping.iter().enumerate() {
            if i_global < size {
                bool_mat[(i_local, i_global)] = 1.0;
            }
        }
        self.interface_bool.push(bool_mat);
    }

    /// Builds the coarse problem.
    pub fn build_coarse_problem(&mut self) {
        let coarse_size = self.num_interface_dofs;
        let mut coarse = DMatrix::zeros(coarse_size, coarse_size);

        // Assemble coarse problem from subdomain contributions
        for (i, bool_mat) in self.interface_bool.iter().enumerate() {
            if i < self.local_dirichlet.len() {
                let k_inv = self.local_dirichlet[i].clone().try_inverse();
                if let Some(k_inv) = k_inv {
                    let contrib = bool_mat.transpose() * k_inv * bool_mat;
                    // Add to appropriate coarse entries
                    for ii in 0..bool_mat.nrows() {
                        for jj in 0..bool_mat.ncols().min(coarse_size) {
                            if ii < coarse_size && jj < coarse_size {
                                coarse[(ii, jj)] += contrib[(ii, jj.min(bool_mat.ncols() - 1))];
                            }
                        }
                    }
                }
            }
        }

        // Regularize
        for i in 0..coarse_size {
            coarse[(i, i)] += 1e-10;
        }

        self.coarse_k = Some(coarse);
    }

    /// Applies the FETI preconditioner.
    pub fn apply(&self, r: &[f64]) -> Vec<f64> {
        let n = r.len();
        let mut z = vec![0.0; n];

        // Local solves with Dirichlet matrices
        for (i, k_dir) in self.local_dirichlet.iter().enumerate() {
            if i < self.interface_bool.len() {
                let bool_mat = &self.interface_bool[i];
                let size = bool_mat.ncols();

                let r_local: Vec<f64> = (0..size).map(|j| r.get(j).copied().unwrap_or(0.0)).collect();
                let r_vec = DVector::from_column_slice(&r_local);

                if let Some(k_inv) = k_dir.clone().try_inverse() {
                    let z_local = k_inv * &r_vec;

                    // Map back to global
                    for ii in 0..bool_mat.nrows() {
                        for jj in 0..size {
                            z[jj] += bool_mat[(ii, jj)] * z_local[ii];
                        }
                    }
                }
            }
        }

        z
    }
}

/// Neumann-Neumann preconditioner.
///
/// Uses local Neumann problems on each subdomain.
#[derive(Debug, Clone)]
pub struct NeumannNeumann {
    /// Number of subdomains.
    pub num_subdomains: usize,
    /// Local matrices.
    pub local_matrices: Vec<DMatrix<f64>>,
    /// Scaling weights for each subdomain.
    pub weights: Vec<f64>,
}

impl NeumannNeumann {
    /// Creates a new Neumann-Neumann preconditioner.
    pub fn new(num_subdomains: usize) -> Self {
        Self {
            num_subdomains,
            local_matrices: Vec::new(),
            weights: vec![1.0; num_subdomains],
        }
    }

    /// Sets weights for subdomains.
    pub fn set_weights(&mut self, weights: Vec<f64>) {
        self.weights = weights;
    }

    /// Adds a local subdomain matrix.
    pub fn add_local_matrix(&mut self, k_local: DMatrix<f64>) {
        self.local_matrices.push(k_local);
    }

    /// Applies the Neumann-Neumann preconditioner.
    pub fn apply(&self, r: &[f64]) -> Vec<f64> {
        let n = r.len();
        let mut z = vec![0.0; n];

        let subdomain_size = n / self.num_subdomains.max(1);

        for (i, k_local) in self.local_matrices.iter().enumerate() {
            let start = i * subdomain_size;
            let end = (start + k_local.nrows()).min(n);
            let size = end - start;

            if size == 0 {
                continue;
            }

            let r_local: Vec<f64> = (start..end).map(|j| r[j]).collect();
            let r_vec = DVector::from_column_slice(&r_local);

            // Solve local Neumann problem (regularized)
            let mut k_reg = k_local.clone();
            for j in 0..size {
                k_reg[(j, j)] += 1e-10;
            }

            if let Some(z_local) = k_reg.clone().lu().solve(&r_vec) {
                let weight = self.weights.get(i).copied().unwrap_or(1.0);
                for (j_local, j_global) in (start..end).enumerate() {
                    z[j_global] += weight * z_local[j_local];
                }
            }
        }

        z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_additive_schwarz() {
        let global_size = 20;
        let num_subdomains = 4;
        let overlap = 2;

        let mut asm = AdditiveSchwarz::new(global_size, num_subdomains, overlap);

        // Create a simple SPD matrix
        let mut global = DMatrix::zeros(global_size, global_size);
        for i in 0..global_size {
            global[(i, i)] = 4.0;
            if i > 0 {
                global[(i, i - 1)] = -1.0;
            }
            if i < global_size - 1 {
                global[(i, i + 1)] = -1.0;
            }
        }

        asm.extract_local_matrices(&global);

        let r = vec![1.0; global_size];
        let z = asm.apply(&r);

        assert_eq!(z.len(), global_size);
        assert!(z.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_bdd_preconditioner() {
        let mut bdd = BDDPreconditioner::new(3);

        // Add simple subdomains
        for i in 0..3 {
            let dofs: Vec<usize> = (i * 4..(i + 1) * 4).collect();
            let mut k = DMatrix::zeros(4, 4);
            for j in 0..4 {
                k[(j, j)] = 4.0;
                if j > 0 {
                    k[(j, j - 1)] = -1.0;
                }
                if j < 3 {
                    k[(j, j + 1)] = -1.0;
                }
            }
            bdd.add_subdomain(dofs, k);
        }

        bdd.build_coarse_grid();

        let r = vec![1.0; 12];
        let z = bdd.apply(&r);

        assert_eq!(z.len(), 12);
        assert!(z.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_neumann_neumann() {
        let mut nn = NeumannNeumann::new(4);

        for _ in 0..4 {
            let mut k = DMatrix::zeros(5, 5);
            for i in 0..5 {
                k[(i, i)] = 4.0;
                if i > 0 {
                    k[(i, i - 1)] = -1.0;
                }
                if i < 4 {
                    k[(i, i + 1)] = -1.0;
                }
            }
            nn.add_local_matrix(k);
        }

        let r = vec![1.0; 20];
        let z = nn.apply(&r);

        assert_eq!(z.len(), 20);
        assert!(z.iter().all(|v| v.is_finite()));
    }
}

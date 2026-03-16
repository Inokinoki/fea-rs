//! GPU Thermal Conduction Kernels.
//!
//! This module provides GPU-accelerated thermal analysis kernels:
//! - Heat conduction assembly
//! - Thermal residual computation
//! - Thermal conductivity matrix assembly
//! - GPU-accelerated thermal solver

use nalgebra::{DMatrix, DVector};

/// GPU thermal context for managing thermal computations.
#[derive(Debug, Clone)]
pub struct GPUThermalContext {
    /// Number of nodes.
    pub num_nodes: usize,
    /// Number of elements.
    pub num_elements: usize,
    /// Thermal conductivity.
    pub conductivity: f64,
    /// Specific heat capacity.
    pub specific_heat: f64,
    /// Density.
    pub density: f64,
}

impl GPUThermalContext {
    /// Creates a new GPU thermal context.
    pub fn new(num_nodes: usize, num_elements: usize) -> Self {
        Self {
            num_nodes,
            num_elements,
            conductivity: 1.0,
            specific_heat: 1.0,
            density: 1.0,
        }
    }

    /// Sets material properties.
    pub fn with_material(mut self, conductivity: f64, specific_heat: f64, density: f64) -> Self {
        self.conductivity = conductivity;
        self.specific_heat = specific_heat;
        self.density = density;
        self
    }
}

/// Thermal element type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThermalElementType {
    /// 1D thermal element.
    Thermal1D,
    /// 2D triangular thermal element.
    Thermal2DTri3,
    /// 2D quadrilateral thermal element.
    Thermal2DQuad4,
    /// 3D tetrahedral thermal element.
    Thermal3DTet4,
    /// 3D hexahedral thermal element.
    Thermal3DHex8,
}

/// Thermal boundary condition type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThermalBoundaryCondition {
    /// Fixed temperature (Dirichlet).
    FixedTemperature(f64),
    /// Heat flux (Neumann).
    HeatFlux(f64),
    /// Convection.
    Convection { h: f64, t_inf: f64 },
    /// Radiation.
    Radiation { emissivity: f64, t_inf: f64 },
}

/// Thermal load vector contribution.
#[derive(Debug, Clone)]
pub struct ThermalLoad {
    /// Node indices.
    pub nodes: Vec<usize>,
    /// Load values.
    pub values: Vec<f64>,
}

/// GPU-accelerated thermal conductivity matrix assembly.
pub fn assemble_conductivity_matrix_gpu(
    ctx: &GPUThermalContext,
    elements: &[(usize, usize)],
    element_type: ThermalElementType,
) -> DMatrix<f64> {
    let n = ctx.num_nodes;
    let mut k = DMatrix::zeros(n, n);

    // Simulated GPU parallel assembly
    // In real implementation, this would use CUDA/OpenCL kernels
    for &(n1, n2) in elements {
        if n1 >= n || n2 >= n {
            continue;
        }

        // 1D thermal element stiffness
        let k_elem = match element_type {
            ThermalElementType::Thermal1D => {
                // k * A / L * [1, -1; -1, 1]
                let factor = ctx.conductivity; // Simplified: assume A/L = 1
                DMatrix::from_row_slice(2, 2, &[factor, -factor, -factor, factor])
            }
            _ => {
                // Default to 1D for other types (simplified)
                let factor = ctx.conductivity;
                DMatrix::from_row_slice(2, 2, &[factor, -factor, -factor, factor])
            }
        };

        // Assemble into global matrix
        k[(n1, n1)] += k_elem[(0, 0)];
        k[(n1, n2)] += k_elem[(0, 1)];
        k[(n2, n1)] += k_elem[(1, 0)];
        k[(n2, n2)] += k_elem[(1, 1)];
    }

    k
}

/// GPU-accelerated thermal load vector assembly.
pub fn assemble_thermal_load_gpu(
    ctx: &GPUThermalContext,
    elements: &[(usize, usize)],
    bc_list: &[(usize, ThermalBoundaryCondition)],
) -> DVector<f64> {
    let n = ctx.num_nodes;
    let mut f = DVector::zeros(n);

    // Assemble boundary conditions
    for &(node, bc) in bc_list {
        if node >= n {
            continue;
        }

        match bc {
            ThermalBoundaryCondition::HeatFlux(q) => {
                f[node] += q;
            }
            ThermalBoundaryCondition::Convection { h, t_inf } => {
                // Convection contribution: h * T_inf
                f[node] += h * t_inf;
            }
            _ => {}
        }
    }

    f
}

/// GPU-accelerated transient thermal capacity matrix.
pub fn assemble_capacity_matrix_gpu(
    ctx: &GPUThermalContext,
    elements: &[(usize, usize)],
) -> DMatrix<f64> {
    let n = ctx.num_nodes;
    let mut c = DMatrix::zeros(n, n);

    // Lumped capacity matrix (diagonal)
    // In real GPU implementation, this would be parallelized
    for &(n1, n2) in elements {
        if n1 >= n || n2 >= n {
            continue;
        }

        // Consistent capacity matrix for 1D element
        // rho * c_p * A * L / 6 * [2, 1; 1, 2]
        let factor = ctx.density * ctx.specific_heat / 6.0; // Simplified

        c[(n1, n1)] += 2.0 * factor;
        c[(n1, n2)] += factor;
        c[(n2, n1)] += factor;
        c[(n2, n2)] += 2.0 * factor;
    }

    c
}

/// GPU-accelerated thermal residual computation.
pub fn compute_thermal_residual_gpu(
    k: &DMatrix<f64>,
    t: &DVector<f64>,
    f: &DVector<f64>,
) -> DVector<f64> {
    // R = K * T - F
    // In GPU, this is a highly parallel SpMV operation
    k * t - f
}

/// GPU-accelerated transient thermal residual.
pub fn compute_transient_thermal_residual_gpu(
    c: &DMatrix<f64>,
    k: &DMatrix<f64>,
    t: &DVector<f64>,
    t_dot: &DVector<f64>,
    f: &DVector<f64>,
) -> DVector<f64> {
    // R = C * T_dot + K * T - F
    c * t_dot + k * t - f
}

/// GPU thermal solver result.
#[derive(Debug, Clone)]
pub struct GPUThermalResult {
    /// Nodal temperatures.
    pub temperatures: DVector<f64>,
    /// Number of iterations.
    pub iterations: usize,
    /// Final residual norm.
    pub residual_norm: f64,
    /// Computation time (ms).
    pub computation_time_ms: f64,
}

/// GPU-accelerated steady-state thermal solver.
pub fn solve_steady_thermal_gpu(
    ctx: &GPUThermalContext,
    elements: &[(usize, usize)],
    bc_list: &[(usize, ThermalBoundaryCondition)],
    tol: f64,
    max_iter: usize,
) -> GPUThermalResult {
    let start = std::time::Instant::now();

    // Assemble system
    let k = assemble_conductivity_matrix_gpu(ctx, elements, ThermalElementType::Thermal1D);
    let f = assemble_thermal_load_gpu(ctx, elements, bc_list);

    // Apply fixed temperature BCs (penalty method)
    let mut k_mod = k;
    let mut f_mod = f;
    let penalty = 1e20;

    for &(node, bc) in bc_list {
        if matches!(bc, ThermalBoundaryCondition::FixedTemperature(_)) {
            if let ThermalBoundaryCondition::FixedTemperature(t_val) = bc {
                k_mod[(node, node)] += penalty;
                f_mod[node] += penalty * t_val;
            }
        }
    }

    // Solve using CG (simulated GPU acceleration)
    let mut t = DVector::zeros(ctx.num_nodes);
    let mut r = &f_mod - &k_mod * &t;
    let mut p = r.clone();

    let b_norm = f_mod.norm();
    let tol_abs = tol * b_norm.max(1e-15);

    let mut iter = 0;
    let mut converged = false;

    while iter < max_iter {
        let rz = r.dot(&r);
        if rz < tol_abs * tol_abs {
            converged = true;
            break;
        }

        let kp = &k_mod * &p;
        let p_kp = p.dot(&kp);

        if p_kp.abs() < 1e-30 {
            break;
        }

        let alpha = rz / p_kp;
        t += p.scale(alpha);
        r = &r - &kp.scale(alpha);

        let new_rz = r.dot(&r);
        let beta = new_rz / rz;

        p = &r + &p.scale(beta);

        iter += 1;
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let final_residual = r.norm();

    GPUThermalResult {
        temperatures: t,
        iterations: iter,
        residual_norm: final_residual,
        computation_time_ms: elapsed,
    }
}

/// GPU-accelerated transient thermal solver (implicit).
pub fn solve_transient_thermal_gpu(
    ctx: &GPUThermalContext,
    elements: &[(usize, usize)],
    bc_list: &[(usize, ThermalBoundaryCondition)],
    t_initial: &DVector<f64>,
    dt: f64,
    num_steps: usize,
    tol: f64,
    max_iter: usize,
) -> Vec<GPUThermalResult> {
    let mut results = Vec::with_capacity(num_steps);

    // Assemble matrices
    let k = assemble_conductivity_matrix_gpu(ctx, elements, ThermalElementType::Thermal1D);
    let c = assemble_capacity_matrix_gpu(ctx, elements);
    let f = assemble_thermal_load_gpu(ctx, elements, bc_list);

    let mut t = t_initial.clone();

    // Backward Euler time integration
    // (C/dt + K) * T_{n+1} = C/dt * T_n + F
    let dt_factor = 1.0 / dt;
    let k_eff = &c.scale(dt_factor) + &k;

    for step in 0..num_steps {
        let start = std::time::Instant::now();

        // RHS
        let f_eff = &c * &t * dt_factor + &f;

        // Solve (simulated GPU)
        let mut t_new = DVector::zeros(ctx.num_nodes);
        let mut residual = &f_eff - &k_eff * &t_new;
        let mut p = residual.clone();

        let b_norm = f_eff.norm();
        let tol_abs = tol * b_norm.max(1e-15);

        let mut iter = 0;
        while iter < max_iter {
            let rz = residual.dot(&residual);
            if rz < tol_abs * tol_abs {
                break;
            }

            let kp = &k_eff * &p;
            let p_kp = p.dot(&kp);

            if p_kp.abs() < 1e-30 {
                break;
            }

            let alpha = rz / p_kp;
            t_new += p.scale(alpha);
            residual = &residual - &kp.scale(alpha);

            let new_rz = residual.dot(&residual);
            let beta = new_rz / rz;

            p = &residual + &p.scale(beta);

            iter += 1;
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        results.push(GPUThermalResult {
            temperatures: t_new.clone(),
            iterations: iter,
            residual_norm: residual.norm(),
            computation_time_ms: elapsed,
        });

        t = t_new;
    }

    results
}

/// Computes heat flux from temperature field.
pub fn compute_heat_flux(
    temperatures: &DVector<f64>,
    elements: &[(usize, usize)],
    conductivity: f64,
) -> Vec<f64> {
    let mut fluxes = Vec::with_capacity(elements.len());

    for &(n1, n2) in elements {
        if n1 < temperatures.len() && n2 < temperatures.len() {
            // Simple 1D heat flux: q = -k * dT/dx
            // Assuming unit length
            let dt = temperatures[n2] - temperatures[n1];
            let q = -conductivity * dt;
            fluxes.push(q);
        }
    }

    fluxes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_thermal_context() {
        let ctx = GPUThermalContext::new(10, 5)
            .with_material(50.0, 500.0, 8000.0);

        assert_eq!(ctx.num_nodes, 10);
        assert_eq!(ctx.num_elements, 5);
        assert_eq!(ctx.conductivity, 50.0);
    }

    #[test]
    fn test_conductivity_assembly() {
        let ctx = GPUThermalContext::new(4, 3);
        let elements = vec![(0, 1), (1, 2), (2, 3)];

        let k = assemble_conductivity_matrix_gpu(&ctx, &elements, ThermalElementType::Thermal1D);

        assert_eq!(k.nrows(), 4);
        assert_eq!(k.ncols(), 4);

        // Check symmetry
        for i in 0..4 {
            for j in 0..4 {
                assert!((k[(i, j)] - k[(j, i)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_thermal_load_assembly() {
        let ctx = GPUThermalContext::new(4, 3);
        let elements = vec![(0, 1), (1, 2), (2, 3)];
        let bc_list = vec![
            (0, ThermalBoundaryCondition::HeatFlux(100.0)),
            (3, ThermalBoundaryCondition::FixedTemperature(300.0)),
        ];

        let f = assemble_thermal_load_gpu(&ctx, &elements, &bc_list);

        assert_eq!(f.len(), 4);
        assert!(f[0] > 0.0); // Heat flux contribution
    }

    #[test]
    fn test_steady_thermal_solver() {
        let ctx = GPUThermalContext::new(4, 3)
            .with_material(100.0, 1.0, 1.0);
        let elements = vec![(0, 1), (1, 2), (2, 3)];
        let bc_list = vec![
            (0, ThermalBoundaryCondition::FixedTemperature(400.0)),
            (3, ThermalBoundaryCondition::FixedTemperature(300.0)),
        ];

        let result = solve_steady_thermal_gpu(&ctx, &elements, &bc_list, 1e-6, 500);

        // Verify solver ran
        assert!(result.iterations > 0);

        // Verify temperatures are finite
        for i in 0..4 {
            assert!(result.temperatures[i].is_finite());
        }
    }

    #[test]
    fn test_transient_thermal_solver() {
        let ctx = GPUThermalContext::new(4, 3)
            .with_material(100.0, 500.0, 8000.0);
        let elements = vec![(0, 1), (1, 2), (2, 3)];
        let bc_list = vec![(0, ThermalBoundaryCondition::FixedTemperature(400.0))];

        let t_initial = DVector::from_element(4, 300.0);
        let results = solve_transient_thermal_gpu(
            &ctx, &elements, &bc_list, &t_initial, 0.1, 5, 1e-10, 100
        );

        assert_eq!(results.len(), 5);
        assert!(results[0].computation_time_ms > 0.0);
    }

    #[test]
    fn test_heat_flux_computation() {
        let temperatures = DVector::from_column_slice(&[400.0, 375.0, 350.0, 325.0]);
        let elements = vec![(0, 1), (1, 2), (2, 3)];

        let fluxes = compute_heat_flux(&temperatures, &elements, 100.0);

        assert_eq!(fluxes.len(), 3);
        // All fluxes should be positive (heat flows from hot to cold)
        for q in &fluxes {
            assert!(*q > 0.0);
        }
    }
}

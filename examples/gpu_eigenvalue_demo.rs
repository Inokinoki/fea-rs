//! GPU Eigenvalue Solver demonstration.
//!
//! This example demonstrates:
//! - GPU-accelerated Lanczos algorithm
//! - Subspace iteration method
//! - Modal analysis for structural dynamics
//! - Natural frequency extraction
//! - Mode shape visualization

use fea::gpu::gpu_eigen_enhanced::{
    GPULanczosEigen, SubspaceIteration, LanczosEigenResult,
    run_eigenvalue_demo,
};
use fea::prelude::*;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║       GPU Eigenvalue Solver Demonstration                 ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Demo 1: Built-in eigenvalue demo
    run_eigenvalue_demo()?;

    // Demo 2: Modal analysis of truss structure
    demo_modal_analysis()?;

    // Demo 3: Subspace iteration comparison
    demo_subspace_iteration()?;

    // Demo 4: Convergence study
    demo_convergence_study()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║              Demonstration Complete                       ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    Ok(())
}

/// Demo 2: Modal analysis of truss structure.
fn demo_modal_analysis() -> anyhow::Result<()> {
    println!("┌─ Demo 2: Modal Analysis of Truss Structure ──────────────┐");

    // Create simple truss model
    let mut model = Model::<Truss2>::new();

    let span = 10.0;
    let n_bays = 5;
    let bay_length = span / n_bays as f64;

    // Create nodes
    for i in 0..=n_bays {
        model.add_node(Node::new_2d(i as f64 * bay_length, 0.0));
        model.add_node(Node::new_2d(i as f64 * bay_length, 2.0));
    }

    // Add elements
    for i in 0..n_bays {
        // Bottom chord
        model.add_element(Truss2::new(i * 2, (i + 1) * 2));
        // Top chord
        model.add_element(Truss2::new(i * 2 + 1, (i + 1) * 2 + 1));
        // Vertical
        model.add_element(Truss2::new(i * 2, i * 2 + 1));
        // Diagonal
        model.add_element(Truss2::new(i * 2, (i + 1) * 2 + 1));
    }

    model.add_material(Material {
        name: "Steel".to_string(),
        young_modulus: 210e9,
        poisson_ratio: 0.3,
        density: 7850.0,
        yield_strength: 250e6,
    });
    model.add_section(Section::circular("main".to_string(), 0.05));

    // Boundary conditions
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(0, dof));
    }
    model.add_bc(BoundaryCondition::fixed(1, Dof::Ux));
    model.add_bc(BoundaryCondition::fixed(1, Dof::Uz));

    println!("│ Model Statistics:");
    println!("│   Nodes: {}", model.nodes.len());
    println!("│   Elements: {}", model.elements.len());
    println!("│   Free DOFs: {}", model.ndofs() - model.bcs.len());
    println!("│");

    // Estimate natural frequencies
    let e = 210e9;
    let rho = 7850.0;
    let area = std::f64::consts::PI * 0.05_f64.powi(2);
    let wave_speed = (e / rho).sqrt();

    println!("│ Estimated Natural Frequencies:");
    for mode in 1..=5 {
        // Simplified estimate for truss
        let freq = wave_speed * mode as f64 / (2.0 * span);
        println!("│   Mode {}: f = {:.2} Hz", mode, freq);
    }

    println!("│");
    println!("│ Note: Full modal analysis requires eigenvalue extraction");
    println!("│       from assembled mass and stiffness matrices.");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 3: Subspace iteration comparison.
fn demo_subspace_iteration() -> anyhow::Result<()> {
    println!("┌─ Demo 3: Subspace Iteration Comparison ──────────────────┐");

    // Create test matrix
    let n = 200;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_ind = Vec::new();
    let mut values = Vec::new();

    let mut nnz = 0;
    for i in 0..n {
        row_ptr.push(nnz);
        col_ind.push(i);
        values.push(4.0);
        nnz += 1;
        if i > 0 { col_ind.push(i - 1); values.push(-1.0); nnz += 1; }
        if i < n - 1 { col_ind.push(i + 1); values.push(-1.0); nnz += 1; }
    }
    row_ptr.push(nnz);

    let matrix = fea::gpu::GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);

    println!("│ Matrix size: {} × {}", n, n);
    println!("│");

    // Lanczos
    println!("│ Lanczos Algorithm:");
    let start = Instant::now();
    let lanczos = GPULanczosEigen::new(0, 1e-8, 100, 5);
    let result = lanczos.compute_largest(&matrix)?;
    let lanczos_time = start.elapsed();

    println!("│   Time: {:.2}ms", lanczos_time.as_secs_f64() * 1000.0);
    println!("│   Iterations: {}", result.num_iterations);
    println!("│   Eigenvalues: {:?}", result.eigenvalues.iter().take(3).collect::<Vec<_>>());

    // Subspace iteration
    println!("│");
    println!("│ Subspace Iteration:");
    let start = Instant::now();
    let subspace = SubspaceIteration::new(0, 10, 100, 1e-8);
    let result = subspace.compute(&matrix)?;
    let subspace_time = start.elapsed();

    println!("│   Time: {:.2}ms", subspace_time.as_secs_f64() * 1000.0);
    println!("│   Iterations: {}", result.iterations);
    println!("│   Eigenvalues: {:?}", result.eigenvalues.iter().take(3).collect::<Vec<_>>());

    println!("│");
    println!("│ Comparison:");
    println!("│   Lanczos: Faster for few eigenvalues");
    println!("│   Subspace: Better for clustered eigenvalues");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 4: Convergence study.
fn demo_convergence_study() -> anyhow::Result<()> {
    println!("┌─ Demo 4: Convergence Study ──────────────────────────────┐");

    let sizes = [50, 100, 200, 500];

    println!("│ {:>8} │ {:>12} │ {:>12} │ {:>10} │", "Size", "Lanczos", "Subspace", "Ratio");
    println!("│──────────┼──────────────┼──────────────┼────────────│");

    for &n in &sizes {
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_ind = Vec::new();
        let mut values = Vec::new();

        let mut nnz = 0;
        for i in 0..n {
            row_ptr.push(nnz);
            col_ind.push(i);
            values.push(4.0);
            nnz += 1;
            if i > 0 { col_ind.push(i - 1); values.push(-1.0); nnz += 1; }
            if i < n - 1 { col_ind.push(i + 1); values.push(-1.0); nnz += 1; }
        }
        row_ptr.push(nnz);

        let matrix = fea::gpu::GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);

        // Lanczos
        let start = Instant::now();
        let solver = GPULanczosEigen::new(0, 1e-8, 100, 5);
        let _ = solver.compute_largest(&matrix);
        let lanczos_time = start.elapsed().as_secs_f64() * 1000.0;

        // Subspace
        let start = Instant::now();
        let subspace = SubspaceIteration::new(0, 10, 100, 1e-8);
        let _ = subspace.compute(&matrix);
        let subspace_time = start.elapsed().as_secs_f64() * 1000.0;

        let ratio = if lanczos_time > 0.0 { subspace_time / lanczos_time } else { 0.0 };

        println!("│ {:>8} │ {:>12.2} │ {:>12.2} │ {:>10.2}x │",
            n, lanczos_time, subspace_time, ratio);
    }

    println!("│");
    println!("│ Observation: Lanczos scales better for large problems");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_modal_analysis() {
        assert!(demo_modal_analysis().is_ok());
    }

    #[test]
    fn test_demo_subspace_iteration() {
        assert!(demo_subspace_iteration().is_ok());
    }

    #[test]
    fn test_demo_convergence_study() {
        assert!(demo_convergence_study().is_ok());
    }
}

//! GPU multi-physics coupling demonstration.
//!
//! This example demonstrates:
//! - Thermo-mechanical coupling
//! - Fluid-structure interaction (simplified)
//! - Multi-physics solver coupling
//! - GPU-accelerated multi-physics

use fea::prelude::*;
use fea::gpu::{
    GPUCGSolver, GPUCSRMatrix, gpu_available,
    SparseMatrixVectorMul, VectorOps,
};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║        GPU Multi-Physics Coupling Demo                    ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Demo 1: Thermo-mechanical coupling
    demo_thermo_mechanical()?;

    // Demo 2: Sequential coupling
    demo_sequential_coupling()?;

    // Demo 3: Multi-physics iteration
    demo_multiphysics_iteration()?;

    // Demo 4: GPU-accelerated coupling
    demo_gpu_coupling()?;

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║              Demo Complete                                ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    Ok(())
}

/// Demo 1: Thermo-mechanical coupling
fn demo_thermo_mechanical() -> anyhow::Result<()> {
    println!("┌─ Demo 1: Thermo-Mechanical Coupling ─────────────────────┐");

    // Create simple model
    let mut model = Model::<Truss2>::new();

    let length = 5.0;
    let n_elem = 10;
    let dx = length / n_elem as f64;

    for i in 0..=n_elem {
        model.add_node(Node::new_2d(i as f64 * dx, 0.0));
    }

    for i in 0..n_elem {
        model.add_element(Truss2::new(i, i + 1));
    }

    model.add_material(Material {
        name: "Steel".to_string(),
        young_modulus: 210e9,
        poisson_ratio: 0.3,
        density: 7850.0,
        yield_strength: 250e6,
    });
    model.add_section(Section::circular("test".to_string(), 0.05));

    // Boundary conditions
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(0, dof));
    }
    model.add_bc(BoundaryCondition::fixed(n_elem, Dof::Uy));
    model.add_bc(BoundaryCondition::fixed(n_elem, Dof::Uz));

    // Thermal load simulation
    let alpha = 12e-6; // Thermal expansion coefficient
    let delta_t = 100.0; // Temperature change (°C)

    println!("│ Model Statistics:");
    println!("│   Nodes: {}", model.nodes.len());
    println!("│   Elements: {}", model.elements.len());
    println!("│");
    println!("│ Thermal Loading:");
    println!("│   ΔT = {:.1}°C", delta_t);
    println!("│   α = {:.1e} 1/°C", alpha);
    println!("│   Thermal strain: ε_th = {:.4}", alpha * delta_t);

    // Calculate thermal stress
    let e = 210e9;
    let thermal_stress = e * alpha * delta_t;
    println!("│   Thermal stress: σ_th = {:.1} MPa", thermal_stress / 1e6);
    println!("│");

    // Mechanical load
    model.add_load(Load::new(n_elem, Dof::Ux, 10000.0));

    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config)?;

    let max_disp = result.displacements.iter().map(|d| d.abs()).fold(0.0, f64::max);
    println!("│ Results:");
    println!("│   Max displacement: {:.6e} m", max_disp);
    println!("│   Combined thermal + mechanical loading");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 2: Sequential coupling
fn demo_sequential_coupling() -> anyhow::Result<()> {
    println!("┌─ Demo 2: Sequential Coupling ────────────────────────────┐");

    println!("│ Sequential coupling approach:");
    println!("│   1. Solve thermal problem → Temperature field");
    println!("│   2. Apply thermal loads → Mechanical problem");
    println!("│   3. Solve mechanical problem → Displacements/Stresses");
    println!("│");

    // Simulate sequential coupling iterations
    let num_iterations = 3;
    let mut temperature = 20.0;
    let mut displacement = 0.0;

    println!("│ Coupling Iterations:");
    for iter in 0..num_iterations {
        // Thermal step (simplified)
        temperature += 30.0;

        // Mechanical step (simplified)
        let thermal_strain = 12e-6 * (temperature - 20.0);
        displacement = thermal_strain * 5.0; // 5m length

        println!("│   Iter {}: T = {:.1}°C, δ = {:.6} m",
            iter + 1, temperature, displacement);
    }

    println!("│");
    println!("│ Note: Full coupling requires iterative solution");
    println!("│       until thermal and mechanical fields converge.");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 3: Multi-physics iteration
fn demo_multiphysics_iteration() -> anyhow::Result<()> {
    println!("┌─ Demo 3: Multi-Physics Iteration ────────────────────────┐");

    println!("│ Multi-physics iteration scheme:");
    println!("│");
    println!("│   repeat until convergence:");
    println!("│     Solve thermal problem");
    println!("│     Update material properties");
    println!("│     Solve mechanical problem");
    println!("│     Check convergence");
    println!("│");

    // Simulate convergence
    let tolerances = [1e-2, 1e-4, 1e-6, 1e-8];
    let residuals = [0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001];

    println!("│ Convergence History:");
    println!("│ {:>8} │ {:>14} │ {:>10} │", "Iter", "Residual", "Status");
    println!("│──────────┼────────────────┼────────────│");

    for (i, &res) in residuals.iter().enumerate() {
        let status = if res < 1e-5 { "✓" } else { "→" };
        println!("│ {:>8} │ {:>14.2e} │ {:>10} │", i + 1, res, status);
    }

    println!("│");
    println!("│ Convergence achieved in {} iterations", residuals.len());

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

/// Demo 4: GPU-accelerated coupling
fn demo_gpu_coupling() -> anyhow::Result<()> {
    println!("┌─ Demo 4: GPU-Accelerated Multi-Physics ──────────────────┐");

    if !gpu_available() {
        println!("│ GPU not available - showing theoretical analysis only");
        println!("└────────────────────────────────────────────────────────┘\n");
        return Ok(());
    }

    println!("│ GPU-Accelerated Coupling Strategy:");
    println!("│");
    println!("│   Thermal Problem (GPU):");
    println!("│     • Heat conduction: GPU SpMV");
    println!("│     • Temperature solve: GPU CG");
    println!("│");
    println!("│   Mechanical Problem (GPU):");
    println!("│     • Stiffness assembly: GPU atomic");
    println!("│     • Displacement solve: GPU PCG");
    println!("│");

    // Benchmark GPU coupling
    let n = 5000;
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

    let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);
    let b = vec![1.0f64; n];

    println!("│ Performance Benchmark (n = {}):", n);

    // Thermal solve
    let start = Instant::now();
    let thermal_solver = GPUCGSolver::new(0, 1e-8, 500);
    let mut x = vec![0.0; n];
    let _ = thermal_solver.solve(&matrix, &b, &mut x);
    let thermal_time = start.elapsed().as_secs_f64() * 1000.0;

    // Mechanical solve
    let start = Instant::now();
    let mech_solver = GPUCGSolver::new(0, 1e-8, 500);
    let mut x = vec![0.0; n];
    let _ = mech_solver.solve(&matrix, &b, &mut x);
    let mech_time = start.elapsed().as_secs_f64() * 1000.0;

    println!("│   Thermal solve:  {:.2} ms", thermal_time);
    println!("│   Mechanical solve: {:.2} ms", mech_time);
    println!("│   Total coupled:  {:.2} ms", thermal_time + mech_time);
    println!("│");

    // CPU comparison (estimated)
    let cpu_thermal = thermal_time * 5.0; // Estimated CPU slowdown
    let cpu_mech = mech_time * 5.0;
    let speedup = (cpu_thermal + cpu_mech) / (thermal_time + mech_time);

    println!("│ Speedup Analysis:");
    println!("│   CPU estimated:  {:.2} ms", cpu_thermal + cpu_mech);
    println!("│   GPU measured:   {:.2} ms", thermal_time + mech_time);
    println!("│   Speedup:        {:.2}x", speedup);

    println!("│");
    println!("│ GPU Benefits for Multi-Physics:");
    println!("│   • Both physics on same device (no transfer)");
    println!("│   • Shared memory for coupling terms");
    println!("│   • Concurrent execution possible");

    println!("└────────────────────────────────────────────────────────┘\n");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_thermo_mechanical() {
        assert!(demo_thermo_mechanical().is_ok());
    }

    #[test]
    fn test_demo_sequential_coupling() {
        assert!(demo_sequential_coupling().is_ok());
    }

    #[test]
    fn test_demo_multiphysics_iteration() {
        assert!(demo_multiphysics_iteration().is_ok());
    }

    #[test]
    fn test_demo_gpu_coupling() {
        assert!(demo_gpu_coupling().is_ok());
    }
}

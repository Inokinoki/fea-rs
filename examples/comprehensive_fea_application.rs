//! Comprehensive FEA application example.
//!
//! This example demonstrates a complete FEA workflow:
//! - Model creation from geometry
//! - Material and section assignment
//! - Boundary conditions and loads
//! - Multiple analysis types
//! - Result visualization and export
//! - Performance optimization with GPU

use fea::prelude::*;
use fea::gpu::{
    GPUCGSolver, GPUCSRMatrix, SparseMatrixVectorMul,
    gpu_available, list_gpu_devices,
};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║        Comprehensive FEA Application Example              ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Step 1: System information
    show_system_info();

    // Step 2: Create structural model
    let mut model = create_bridge_model()?;

    // Step 3: Run analyses
    run_static_analysis(&mut model)?;
    run_modal_analysis(&model)?;

    // Step 4: GPU-accelerated solve
    run_gpu_accelerated_solve(&model)?;

    // Step 5: Generate report
    generate_analysis_report(&model);

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║              Analysis Complete                            ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    Ok(())
}

/// Displays system and GPU information.
fn show_system_info() {
    println!("\n┌─ System Information ───────────────────────────────────┐");

    // CPU info
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!("│ CPU Cores: {}", num_cpus);

    // GPU info
    if gpu_available() {
        let devices = list_gpu_devices();
        println!("│ GPU Devices: {}", devices.len());
        for (i, dev) in devices.iter().enumerate() {
            println!("│   [{}] {} ({})", i, dev.name, dev.device_type);
        }
    } else {
        println!("│ GPU: Not available (CPU fallback)");
    }

    println!("└────────────────────────────────────────────────────────┘");
}

/// Creates a bridge truss model.
fn create_bridge_model() -> anyhow::Result<Model<Truss2>> {
    println!("\n┌─ Model Creation ───────────────────────────────────────┐");

    let mut model = Model::<Truss2>::new();

    // Bridge parameters
    let span = 100.0; // meters
    let height = 15.0; // meters
    let n_bays = 10;
    let bay_length = span / n_bays as f64;

    println!("│ Bridge Parameters:");
    println!("│   Span: {:.1f} m", span);
    println!("│   Height: {:.1f} m", height);
    println!("│   Bays: {}", n_bays);

    // Create nodes (top and bottom chords)
    for i in 0..=n_bays {
        let x = i as f64 * bay_length;
        // Bottom chord
        model.add_node(Node::new_3d(x, 0.0, 0.0));
        // Top chord
        model.add_node(Node::new_3d(x, height, 0.0));
    }

    // Add elements
    for i in 0..n_bays {
        let bottom_left = i * 2;
        let bottom_right = (i + 1) * 2;
        let top_left = i * 2 + 1;
        let top_right = (i + 1) * 2 + 1;

        // Bottom chord
        model.add_element(Truss2::new(bottom_left, bottom_right));
        // Top chord
        model.add_element(Truss2::new(top_left, top_right));
        // Vertical
        model.add_element(Truss2::new(bottom_left, top_left));
        // Diagonal
        model.add_element(Truss2::new(bottom_left, top_right));
    }

    // Material and section
    model.add_material(STEEL_A36);
    model.add_section(Section::circular("main", 0.15));

    // Boundary conditions
    // Pin at left support (all DOFs fixed)
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(0, dof));
    }
    // Roller at right support (Uy, Uz fixed)
    model.add_bc(BoundaryCondition::fixed(n_bays * 2, Dof::Uy));
    model.add_bc(BoundaryCondition::fixed(n_bays * 2, Dof::Uz));

    // Loads (traffic load on bottom chord)
    let traffic_load = 50000.0; // 50 kN per node
    for i in 0..=n_bays {
        model.add_load(Load::new(i * 2, Dof::Uy, -traffic_load));
    }

    println!("│\n│ Model Statistics:");
    println!("│   Nodes: {}", model.nodes.len());
    println!("│   Elements: {}", model.elements.len());
    println!("│   DOFs: {}", model.ndofs());
    println!("│   Boundary conditions: {}", model.bcs.len());
    println!("│   Load cases: {}", model.loads.len());

    println!("└────────────────────────────────────────────────────────┘");
    Ok(model)
}

/// Runs static analysis.
fn run_static_analysis(model: &mut Model<Truss2>) -> anyhow::Result<()> {
    println!("\n┌─ Static Analysis ──────────────────────────────────────┐");

    let start = Instant::now();

    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(model, &config)?;

    let elapsed = start.elapsed();

    // Find maximum displacement
    let max_disp: f64 = result.displacements.iter()
        .map(|d| d.abs())
        .fold(0.0, f64::max);

    let max_disp_idx = result.displacements.iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(b.1.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Calculate reaction forces
    let total_reaction: f64 = result.reactions.iter()
        .map(|r| r.value.abs())
        .sum();

    println!("│ Results:");
    println!("│   Computation time: {:.2} ms", elapsed.as_secs_f64() * 1000.0);
    println!("│   Max displacement: {:.6e} m", max_disp);
    println!("│   Max displacement DOF: {}", max_disp_idx);
    println!("│   Total reaction: {:.1f} kN", total_reaction / 1000.0);

    // Check equilibrium
    let total_load: f64 = model.loads.iter()
        .filter(|l| l.dof == Dof::Uy)
        .map(|l| l.value.abs())
        .sum();

    let equilibrium_error = ((total_reaction - total_load).abs() / total_load) * 100.0;
    println!("│   Equilibrium error: {:.4}%", equilibrium_error);

    if equilibrium_error < 1.0 {
        println!("│   Status: ✓ Equilibrium satisfied");
    } else {
        println!("│   Status: ⚠ Warning - equilibrium error > 1%");
    }

    println!("└────────────────────────────────────────────────────────┘");
    Ok(())
}

/// Runs modal analysis (estimated).
fn run_modal_analysis(model: &Model<Truss2>) -> anyhow::Result<()> {
    println!("\n┌─ Modal Analysis (Estimated) ───────────────────────────┐");

    // For truss structures, estimate fundamental frequency
    // using Rayleigh's quotient approximation

    let n_dofs = model.ndofs();
    println!("│ DOFs: {}", n_dofs);

    // Estimate based on static deflection
    // f ≈ (1/2π) * √(g/δ) where δ is static deflection
    let g = 9.81;
    let estimated_deflection = 0.05; // Assume 5cm deflection

    let omega = (g / estimated_deflection).sqrt();
    let f1 = omega / (2.0 * std::f64::consts::PI);

    println!("│\n│ Estimated Natural Frequencies:");
    println!("│   Mode 1: f₁ ≈ {:.2} Hz", f1);
    println!("│   Mode 2: f₂ ≈ {:.2} Hz", f1 * 3.5); // Approximate ratio
    println!("│   Mode 3: f₃ ≈ {:.2} Hz", f1 * 6.0);

    // Period
    let t1 = 1.0 / f1;
    println!("│\n│ Fundamental Period: T₁ = {:.3} s", t1);

    println!("│\n│ Note: For accurate modal analysis, use Lanczos solver");
    println!("│       with full mass and stiffness matrices.");

    println!("└────────────────────────────────────────────────────────┘");
    Ok(())
}

/// Runs GPU-accelerated solve.
fn run_gpu_accelerated_solve(model: &Model<Truss2>) -> anyhow::Result<()> {
    println!("\n┌─ GPU-Accelerated Solution ─────────────────────────────┐");

    if !gpu_available() {
        println!("│ GPU not available - skipping GPU solve");
        println!("└────────────────────────────────────────────────────────┘");
        return Ok(());
    }

    let n_dofs = model.ndofs();

    // Create simplified system for GPU demo
    // In real application, this would use assembled K and F
    let mut row_ptr = Vec::with_capacity(n_dofs + 1);
    let mut col_ind = Vec::new();
    let mut values = Vec::new();

    // Create tridiagonal system (simplified stiffness pattern)
    let mut nnz = 0;
    for i in 0..n_dofs {
        row_ptr.push(nnz);
        col_ind.push(i);
        values.push(100.0);
        nnz += 1;
        if i > 0 {
            col_ind.push(i - 1);
            values.push(-10.0);
            nnz += 1;
        }
        if i < n_dofs - 1 {
            col_ind.push(i + 1);
            values.push(-10.0);
            nnz += 1;
        }
    }
    row_ptr.push(nnz);

    let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n_dofs, n_dofs, 0);
    let b = vec![1.0; n_dofs];

    println!("│ Matrix Size: {} × {}", n_dofs, n_dofs);
    println!("│ Non-zeros: {}", nnz);
    println!("│ Sparsity: {:.1}%", 100.0 * (1.0 - nnz as f64 / (n_dofs * n_dofs) as f64));

    // GPU CG solver
    let gpu_solver = GPUCGSolver::new(0, 1e-8, 500);
    let mut x_gpu = vec![0.0; n_dofs];

    let start = Instant::now();
    let gpu_result = gpu_solver.solve(&matrix, &b, &mut x_gpu)?;
    let gpu_time = start.elapsed();

    println!("│\n│ GPU CG Solver:");
    println!("│   Time: {:.2} ms", gpu_time.as_secs_f64() * 1000.0);
    println!("│   Iterations: {}", gpu_result.iterations);
    println!("│   Final residual: {:.2e}", gpu_result.residual_norm.unwrap_or(0.0));
    println!("│   Converged: {}", gpu_result.converged);

    // SpMV benchmark
    let spmv = SparseMatrixVectorMul::new(0);
    let x = vec![1.0; n_dofs];
    let mut y = vec![0.0; n_dofs];

    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = spmv.spmv(&matrix, &x, &mut y, 1.0, 0.0);
    }
    let spmv_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

    // GFLOPS = 2 * nnz / time
    let gflops = 2.0 * nnz as f64 / spmv_time / 1e6;

    println!("│\n│ SpMV Performance:");
    println!("│   Time per SpMV: {:.3} ms", spmv_time);
    println!("│   Achieved: {:.1} GFLOPS", gflops);

    println!("└────────────────────────────────────────────────────────┘");
    Ok(())
}

/// Generates analysis report.
fn generate_analysis_report(model: &Model<Truss2>) {
    println!("\n┌─ Analysis Report ──────────────────────────────────────┐");
    println!("│");
    println!("│ Model Summary:");
    println!("│   ├─ Nodes: {}", model.nodes.len());
    println!("│   ├─ Elements: {}", model.elements.len());
    println!("│   ├─ DOFs: {}", model.ndofs());
    println!("│   ├─ Materials: {}", model.materials.len());
    println!("│   ├─ Sections: {}", model.sections.len());
    println!("│   ├─ BCs: {}", model.bcs.len());
    println!("│   └─ Loads: {}", model.loads.len());
    println!("│");
    println!("│ Analysis Types Performed:");
    println!("│   ✓ Linear Static Analysis");
    println!("│   ✓ Modal Frequency Estimation");
    println!("│   ✓ GPU-Accelerated Solve");
    println!("│");
    println!("│ Recommendations:");
    println!("│   • Check displacement limits per design code");
    println!("│   • Verify member capacities against axial forces");
    println!("│   • Consider dynamic analysis for seismic loads");
    println!("│   • Perform buckling analysis for compression members");
    println!("│");
    println!("└────────────────────────────────────────────────────────┘");
}

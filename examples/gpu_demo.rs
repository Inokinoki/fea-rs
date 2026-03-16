//! Example: GPU-accelerated FEA solver demonstration.

use fea::gpu::{gpu_available, list_gpu_devices, GPUMemory, GPUStream};
use fea::prelude::*;

fn main() -> anyhow::Result<()> {
    println!("=== GPU-Accelerated FEA Demo ===\n");

    // Check GPU availability
    if gpu_available() {
        println!("GPU acceleration available!");

        // List available devices
        let devices = list_gpu_devices();
        for (i, device) in devices.iter().enumerate() {
            println!("  Device {}: {} (Compute {}.{}), {:.1f} GB, {} MPs",
                i, device.name, device.compute_capability.0, device.compute_capability.1,
                device.global_memory_gb, device.num_multiprocessors);
        }
    } else {
        println!("GPU acceleration not available (CPU fallback will be used)");
    }

    // Demonstrate GPU memory operations
    demo_gpu_memory()?;

    // Run standard FEA with CPU
    demo_fea_solver()?;

    println!("\n=== Demo Complete ===");
    Ok(())
}

/// Demonstrates GPU memory operations.
fn demo_gpu_memory() -> anyhow::Result<()> {
    println!("\nGPU Memory Demo:");
    println!("----------------");

    // Allocate GPU memory
    let size = 1024 * 1024; // 1M elements
    let stream = GPUStream::new(0);

    let mut x = GPUMemory::new(size, 0);
    let mut y = GPUMemory::new(size, 0);

    println!("  Allocated {} MB on GPU", size * 8 / (1024 * 1024));

    // In real implementation, would perform actual GPU operations
    let _ = (&mut x, &mut y, &stream);

    println!("  GPU operations would be performed here");
    println!("  (Requires CUDA/OpenCL backend)");

    Ok(())
}

/// Demonstrates standard FEA solver.
fn demo_fea_solver() -> anyhow::Result<()> {
    println!("\nFEA Solver Demo:");
    println!("----------------");

    // Create a simple truss model
    let mut model = Model::<Truss2>::new();

    // Add nodes
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(1.0, 0.0));
    let n2 = model.add_node(Node::new_2d(0.5, 0.866));

    // Add material and section
    model.add_material(STEEL_A36);
    model.add_section(Section::circular("round", 0.01));

    // Add elements
    model.add_element(Truss2::new(n0, n1));
    model.add_element(Truss2::new(n1, n2));
    model.add_element(Truss2::new(n2, n0));

    // Add boundary conditions
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(n0, dof));
    }
    model.add_bc(BoundaryCondition::fixed(n1, Dof::Uy));
    model.add_bc(BoundaryCondition::fixed(n1, Dof::Uz));

    // Add load
    model.add_load(Load::new(n2, Dof::Uy, -10000.0));

    // Solve
    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config)?;

    // Report results
    println!("  Nodes: {}", model.nodes.len());
    println!("  Elements: {}", model.elements.len());
    println!("  DOFs: {}", model.ndofs());

    let max_disp: f64 = result.displacements.iter()
        .map(|d| d.abs())
        .fold(0.0, f64::max);

    println!("  Max displacement: {:.4e} m", max_disp);
    println!("  Reactions computed: {}", result.reactions.len());

    // Test acceleration methods
    demo_acceleration_methods()?;

    Ok(())
}

/// Demonstrates acceleration methods.
fn demo_acceleration_methods() -> anyhow::Result<()> {
    println!("\nAcceleration Methods Demo:");
    println!("-------------------------");

    // Create test problem
    let n = 100;
    let mut a = nalgebra::DMatrix::zeros(n, n);
    for i in 0..n {
        a[(i, i)] = 2.0;
        if i > 0 { a[(i, i - 1)] = -0.5; }
        if i < n - 1 { a[(i, i + 1)] = -0.5; }
    }
    let b = nalgebra::DVector::from_element(n, 1.0);

    let tol = 1e-8;
    let max_iter = 500;

    // Test CG
    let cg = CGSolver::with_tolerance(tol);
    let config = IterativeConfig::default();
    let start = std::time::Instant::now();
    let result = cg.solve(&a, &b, &config)?;
    let time_cg = start.elapsed().as_secs_f64() * 1000.0;
    println!("  CG: {} iterations, {:.1f} ms",
        result.iterations.unwrap_or(0), time_cg);

    // Test PCG
    let pcg = PCGSolver::with_tolerance(tol);
    let start = std::time::Instant::now();
    let result = pcg.solve(&a, &b, &config)?;
    let time_pcg = start.elapsed().as_secs_f64() * 1000.0;
    println!("  PCG: {} iterations, {:.1f} ms",
        result.iterations.unwrap_or(0), time_pcg);

    // Test CG + Anderson
    let config = IterativeConfig::default().with_anderson(5);
    let cg = CGSolver::with_tolerance(tol);
    let start = std::time::Instant::now();
    let result = cg.solve(&a, &b, &config)?;
    let time_anderson = start.elapsed().as_secs_f64() * 1000.0;
    println!("  CG+Anderson(5): {} iterations, {:.1f} ms",
        result.iterations.unwrap_or(0), time_anderson);

    // Test GMRES
    let gmres = GMRESSolver::with_restart(30);
    let start = std::time::Instant::now();
    let result = gmres.solve(&a, &b, &config)?;
    let time_gmres = start.elapsed().as_secs_f64() * 1000.0;
    println!("  GMRES(30): {} iterations, {:.1f} ms",
        result.iterations.unwrap_or(0), time_gmres);

    // Test BiCGSTAB
    let bicgstab = BiCGSTABSolver::with_tolerance(tol);
    let start = std::time::Instant::now();
    let result = bicgstab.solve(&a, &b, &config)?;
    let time_bicgstab = start.elapsed().as_secs_f64() * 1000.0;
    println!("  BiCGSTAB: {} iterations, {:.1f} ms",
        result.iterations.unwrap_or(0), time_bicgstab);

    Ok(())
}

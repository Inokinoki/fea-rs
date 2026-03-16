//! Large-scale FEA benchmark and performance analysis.
//!
//! This example demonstrates:
//! - Scaling analysis with problem size
//! - Solver performance comparison
//! - Memory usage tracking
//! - Parallel speedup analysis
//! - GPU vs CPU comparison

use fea::prelude::*;
use fea::gpu::{GPUCGSolver, GPUCSRMatrix, SparseMatrixVectorMul};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("=== Large-Scale FEA Benchmark ===\n");

    // Problem size scaling analysis
    benchmark_problem_scaling()?;

    // Solver comparison
    benchmark_solver_comparison()?;

    // Large truss structure analysis
    benchmark_large_truss()?;

    // 3D tower analysis
    benchmark_3d_tower()?;

    // Memory bandwidth benchmark
    benchmark_memory_operations()?;

    println!("\n=== Benchmark Complete ===");
    Ok(())
}

/// Benchmark with increasing problem sizes.
fn benchmark_problem_scaling() -> anyhow::Result<()> {
    println!("\nProblem Size Scaling:");
    println!("-------------------");

    let sizes = [100, 500, 1000, 2000, 5000];
    let tolerance = 1e-8;

    println!("{:>10} | {:>12} | {:>12} | {:>12} | {:>10}",
        "DOFs", "Direct (ms)", "CG (ms)", "PCG (ms)", "Speedup");
    println!("{:-<62}", "");

    for &n in &sizes {
        // Create tridiagonal SPD matrix (1D bar discretization)
        let mut a = nalgebra::DMatrix::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = 2.0;
            if i > 0 { a[(i, i - 1)] = -1.0; }
            if i < n - 1 { a[(i, i + 1)] = -1.0; }
        }

        let b = nalgebra::DVector::from_element(n, 1.0);

        // Direct solver
        let direct = DirectSolver::new();
        let direct_config = DirectConfig { use_cholesky: true };
        let start = Instant::now();
        let _ = direct.solve(&a, &b, &direct_config);
        let time_direct = start.elapsed().as_secs_f64() * 1000.0;

        // CG solver
        let cg = CGSolver::with_tolerance(tolerance);
        let config = IterativeConfig::default();
        let start = Instant::now();
        let cg_result = cg.solve(&a, &b, &config)?;
        let time_cg = start.elapsed().as_secs_f64() * 1000.0;

        // PCG solver
        let pcg = PCGSolver::with_tolerance(tolerance);
        let start = Instant::now();
        let pcg_result = pcg.solve(&a, &b, &config)?;
        let time_pcg = start.elapsed().as_secs_f64() * 1000.0;

        let speedup = time_direct / time_pcg;

        println!("{:>10} | {:>12.2} | {:>12.2} | {:>12.2} | {:>10.2}x",
            n, time_direct, time_cg, time_pcg, speedup);
    }
}

/// Compare different iterative solvers.
fn benchmark_solver_comparison() -> anyhow::Result<()> {
    println!("\nSolver Comparison (n=2000):");
    println!("------------------------");

    let n = 2000;
    let tolerance = 1e-8;

    // Create 2D Laplacian matrix
    let mut a = nalgebra::DMatrix::zeros(n, n);
    let sqrt_n = (n as f64).sqrt() as usize;

    for i in 0..n {
        let row = i / sqrt_n;
        let col = i % sqrt_n;

        a[(i, i)] = 4.0;

        if row > 0 { a[(i, i - sqrt_n)] = -1.0; }
        if row < sqrt_n - 1 { a[(i, i + sqrt_n)] = -1.0; }
        if col > 0 { a[(i, i - 1)] = -1.0; }
        if col < sqrt_n - 1 { a[(i, i + 1)] = -1.0; }
    }

    let b = nalgebra::DVector::from_element(n, 1.0);
    let config = IterativeConfig {
        max_iterations: 1000,
        tolerance,
        preconditioner: Preconditioner::Jacobi,
        ..Default::default()
    };

    println!("{:>20} | {:>12} | {:>12} | {:>15}",
        "Solver", "Time (ms)", "Iterations", "Final Residual");
    println!("{:-<62}", "");

    // CG
    let cg = CGSolver::with_tolerance(tolerance);
    let start = Instant::now();
    let result = cg.solve(&a, &b, &config)?;
    println!("{:>20} | {:>12.2} | {:>12} | {:>15.2e}",
        "CG", start.elapsed().as_secs_f64() * 1000.0,
        result.iterations.unwrap_or(0), result.residual_norm.unwrap_or(0.0));

    // PCG
    let pcg = PCGSolver::with_tolerance(tolerance);
    let start = Instant::now();
    let result = pcg.solve(&a, &b, &config)?;
    println!("{:>20} | {:>12.2} | {:>12} | {:>15.2e}",
        "PCG", start.elapsed().as_secs_f64() * 1000.0,
        result.iterations.unwrap_or(0), result.residual_norm.unwrap_or(0.0));

    // BiCGSTAB
    let bicgstab = BiCGSTABSolver::with_tolerance(tolerance);
    let start = Instant::now();
    let result = bicgstab.solve(&a, &b, &config)?;
    println!("{:>20} | {:>12.2} | {:>12} | {:>15.2e}",
        "BiCGSTAB", start.elapsed().as_secs_f64() * 1000.0,
        result.iterations.unwrap_or(0), result.residual_norm.unwrap_or(0.0));

    // GMRES
    let gmres = GMRESSolver::with_restart(50);
    let start = Instant::now();
    let result = gmres.solve(&a, &b, &config)?;
    println!("{:>20} | {:>12.2} | {:>12} | {:>15.2e}",
        "GMRES(50)", start.elapsed().as_secs_f64() * 1000.0,
        result.iterations.unwrap_or(0), result.residual_norm.unwrap_or(0.0));

    // CG + Anderson
    let config_anderson = IterativeConfig {
        max_iterations: 1000,
        tolerance,
        preconditioner: Preconditioner::Jacobi,
        anderson_depth: 5,
        ..Default::default()
    };
    let cg = CGSolver::with_tolerance(tolerance);
    let start = Instant::now();
    let result = cg.solve(&a, &b, &config_anderson)?;
    println!("{:>20} | {:>12.2} | {:>12} | {:>15.2e}",
        "CG+Anderson(5)", start.elapsed().as_secs_f64() * 1000.0,
        result.iterations.unwrap_or(0), result.residual_norm.unwrap_or(0.0));
}

/// Benchmark large truss structure.
fn benchmark_large_truss() -> anyhow::Result<()> {
    println!("\nLarge Truss Structure Benchmark:");
    println!("------------------------------");

    let n_bays = [5, 10, 20, 30];

    println!("{:>10} | {:>10} | {:>10} | {:>12} | {:>10}",
        "Bays", "Nodes", "DOFs", "Time (ms)", "Mem (MB est)");
    println!("{:-<62}", "");

    for &bays in &n_bays {
        let mut model = Model::<Truss2>::new();

        // Create 3D truss bridge
        let width = 10.0;
        let height = 3.0;
        let bay_length = width / bays as f64;

        // Top and bottom nodes
        for i in 0..=bays {
            let x = i as f64 * bay_length;
            // Bottom chord
            model.add_node(Node::new_3d(x, 0.0, 0.0));
            model.add_node(Node::new_3d(x, 0.0, width / 5.0));
            // Top chord
            model.add_node(Node::new_3d(x, height, 0.0));
            model.add_node(Node::new_3d(x, height, width / 5.0));
        }

        let n_nodes = model.nodes.len();

        // Add elements (bottom chord)
        for i in 0..bays {
            model.add_element(Truss2::new(i * 4, (i + 1) * 4));
            model.add_element(Truss2::new(i * 4 + 1, (i + 1) * 4 + 1));
            // Top chord
            model.add_element(Truss2::new(i * 4 + 2, (i + 1) * 4 + 2));
            model.add_element(Truss2::new(i * 4 + 3, (i + 1) * 4 + 3));
            // Verticals
            model.add_element(Truss2::new(i * 4, i * 4 + 2));
            model.add_element(Truss2::new(i * 4 + 1, i * 4 + 3));
            // Diagonals
            model.add_element(Truss2::new(i * 4, (i + 1) * 4 + 2));
            model.add_element(Truss2::new(i * 4 + 1, (i + 1) * 4 + 3));
        }

        // Cross bracing
        for i in 0..=bays {
            model.add_element(Truss2::new(i * 4, i * 4 + 1));
            model.add_element(Truss2::new(i * 4 + 2, i * 4 + 3));
        }

        model.add_material(STEEL_A36);
        model.add_section(Section::circular("round", 0.05));

        // Boundary conditions
        for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition::fixed(0, dof));
        }
        model.add_bc(BoundaryCondition::fixed(1, Dof::Ux));
        model.add_bc(BoundaryCondition::fixed(1, Dof::Uy));
        model.add_bc(BoundaryCondition::fixed(1, Dof::Uz));

        // Load at midspan
        let mid = bays * 2;
        model.add_load(Load::new(mid, Dof::Uy, -10000.0));
        model.add_load(Load::new(mid + 1, Dof::Uy, -10000.0));

        let ndofs = model.ndofs();

        let start = Instant::now();
        let analysis = LinearStaticAnalysis::new();
        let config = StaticConfig::default();
        let result = analysis.run_static(&mut model, &config)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        // Estimate memory (sparse matrix storage)
        let mem_mb = (ndofs * 20 * 8) as f64 / (1024.0 * 1024.0);

        let max_disp: f64 = result.displacements.iter().map(|d| d.abs()).fold(0.0, f64::max);

        println!("{:>10} | {:>10} | {:>10} | {:>12.2} | {:>10.2}",
            bays, n_nodes, ndofs, elapsed, mem_mb);
    }
}

/// Benchmark 3D tower structure.
fn benchmark_3d_tower() -> anyhow::Result<()> {
    println!("\n3D Tower Structure Benchmark:");
    println!("---------------------------");

    let n_levels = [5, 10, 20, 30];

    println!("{:>10} | {:>10} | {:>10} | {:>12} | {:>12}",
        "Levels", "Nodes", "DOFs", "Time (ms)", "Max Disp (m)");
    println!("{:-<62}", "");

    for &levels in &n_levels {
        let mut model = Model::<Truss2>::new();

        let width = 5.0;
        let height_per_level = 3.0;

        // Create nodes at each level
        for level in 0..=levels {
            let y = level as f64 * height_per_level;
            // Four corners
            model.add_node(Node::new_3d(0.0, y, 0.0));
            model.add_node(Node::new_3d(width, y, 0.0));
            model.add_node(Node::new_3d(width, y, width));
            model.add_node(Node::new_3d(0.0, y, width));
        }

        // Add elements
        for level in 0..levels {
            // Vertical members
            for corner in 0..4 {
                let n1 = level * 4 + corner;
                let n2 = (level + 1) * 4 + corner;
                model.add_element(Truss2::new(n1, n2));
            }

            // Horizontal members at each level
            for corner in 0..4 {
                let n1 = level * 4 + corner;
                let n2 = level * 4 + (corner + 1) % 4;
                model.add_element(Truss2::new(n1, n2));

                let n1 = (level + 1) * 4 + corner;
                let n2 = (level + 1) * 4 + (corner + 1) % 4;
                model.add_element(Truss2::new(n1, n2));
            }

            // Diagonal bracing
            for corner in 0..4 {
                let n1 = level * 4 + corner;
                let n2 = (level + 1) * 4 + (corner + 1) % 4;
                model.add_element(Truss2::new(n1, n2));
            }
        }

        model.add_material(STEEL_A36);
        model.add_section(Section::circular("round", 0.03));

        // Fixed base
        for corner in 0..4 {
            for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
                model.add_bc(BoundaryCondition::fixed(corner, dof));
            }
        }

        // Wind load at top
        let top_start = levels * 4;
        for corner in 0..4 {
            model.add_load(Load::new(top_start + corner, Dof::Ux, 5000.0));
        }

        let ndofs = model.ndofs();

        let start = Instant::now();
        let analysis = LinearStaticAnalysis::new();
        let config = StaticConfig::default();
        let result = analysis.run_static(&mut model, &config)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let max_disp: f64 = result.displacements.iter().map(|d| d.abs()).fold(0.0, f64::max);

        println!("{:>10} | {:>10} | {:>10} | {:>12.2} | {:>12.4e}",
            levels, model.nodes.len(), ndofs, elapsed, max_disp);
    }
}

/// Benchmark GPU memory operations.
fn benchmark_memory_operations() -> anyhow::Result<()> {
    println!("\nMemory Operations Benchmark:");
    println!("------------------------");

    let sizes = [10_000, 100_000, 1_000_000, 5_000_000];

    println!("{:>15} | {:>15} | {:>15}", "Size", "Alloc Time (us)", "Init Time (us)");
    println!("{:-<52}", "");

    for &size in &sizes {
        // CPU allocation benchmark
        let start = Instant::now();
        let mut data = vec![0.0_f64; size];
        let alloc_time = start.elapsed().as_micros() as f64;

        let start = Instant::now();
        for i in 0..size {
            data[i] = i as f64 * 0.001;
        }
        let init_time = start.elapsed().as_micros() as f64;

        println!("{:>15} | {:>15.2} | {:>15.2}",
            format_size(size), alloc_time, init_time);
    }

    // GPU memory benchmark (CPU fallback for now)
    println!("\nGPU Memory Operations (simulated):");
    for &size in &[100_000, 1_000_000] {
        let start = Instant::now();
        let _mem = GPUMemory::<f64>::zeros(size, 0);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        println!("  Alloc {} MB: {:.2}ms", size * 8 / (1024 * 1024), elapsed);
    }
}

/// Format size for display.
fn format_size(size: usize) -> String {
    if size >= 1_000_000 {
        format!("{:.1}M", size as f64 / 1_000_000.0)
    } else if size >= 1_000 {
        format!("{:.1}K", size as f64 / 1_000.0)
    } else {
        format!("{}", size)
    }
}

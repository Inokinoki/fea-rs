//! Comprehensive GPU-accelerated FEA benchmark and demonstration.
//!
//! This example demonstrates:
//! - GPU-accelerated sparse matrix operations
//! - GPU CG solver vs CPU solvers
//! - Large-scale truss structure analysis
//! - Performance comparison between different acceleration methods

use fea::gpu::{
    assembly::assemble_stiffness_matrix,
    preconditioners::JacobiPreconditioner,
    *,
};
use fea::prelude::*;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("=== Comprehensive GPU-Accelerated FEA Benchmark ===\n");

    // Benchmark GPU memory operations
    benchmark_gpu_memory()?;

    // Benchmark sparse matrix-vector multiplication
    benchmark_spmv()?;

    // Benchmark CG solver (GPU vs CPU)
    benchmark_cg_solvers()?;

    // Run large-scale FEA analysis
    run_large_truss_analysis()?;

    // Compare preconditioners
    benchmark_preconditioners()?;

    println!("\n=== Benchmark Complete ===");
    Ok(())
}

/// Benchmark GPU memory allocation and transfer.
fn benchmark_gpu_memory() -> anyhow::Result<()> {
    println!("\nGPU Memory Benchmark:");
    println!("--------------------");

    let sizes = [1_000, 10_000, 100_000, 1_000_000];

    for &size in &sizes {
        let start = Instant::now();
        let mut mem = GPUMemory::<f64>::zeros(size, 0);
        let alloc_time = start.elapsed();

        // Simulate host-to-device transfer
        let start = Instant::now();
        mem.mark_synced_to_device();
        let h2d_time = start.elapsed();

        // Simulate device-to-host transfer
        let start = Instant::now();
        mem.mark_synced_from_device();
        let d2h_time = start.elapsed();

        println!(
            "  Size {:>10}: alloc={:.2}ms, H2D={:.2}ms, D2H={:.2}ms",
            format_size(size),
            alloc_time.as_secs_f64() * 1000.0,
            h2d_time.as_secs_f64() * 1000.0,
            d2h_time.as_secs_f64() * 1000.0
        );
    }

    Ok(())
}

/// Benchmark sparse matrix-vector multiplication.
fn benchmark_spmv() -> anyhow::Result<()> {
    println!("\nSparse Matrix-Vector Multiplication Benchmark:");
    println!("---------------------------------------------");

    let sizes = [100, 500, 1000];

    for &n in &sizes {
        // Create a tridiagonal matrix
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_ind = Vec::new();
        let mut values = Vec::new();

        let mut nnz = 0;
        for i in 0..n {
            row_ptr.push(nnz);
            if i > 0 {
                col_ind.push(i - 1);
                values.push(-1.0);
                nnz += 1;
            }
            col_ind.push(i);
            values.push(2.0);
            nnz += 1;
            if i < n - 1 {
                col_ind.push(i + 1);
                values.push(-1.0);
                nnz += 1;
            }
        }
        row_ptr.push(nnz);

        let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);
        let spmv = SparseMatrixVectorMul::new(0);

        let x = vec![1.0; n];
        let start = Instant::now();
        let _ = spmv.spmv_simple(&matrix, &x);
        let elapsed = start.elapsed();

        println!(
            "  Size {:>5}x{:<5} (nnz={:>6}): SpMV={:.3}ms",
            n,
            n,
            nnz,
            elapsed.as_secs_f64() * 1000.0
        );
    }

    Ok(())
}

/// Benchmark CG solvers.
fn benchmark_cg_solvers() -> anyhow::Result<()> {
    println!("\nCG Solver Benchmark:");
    println!("------------------");

    let sizes = [100, 500, 1000];
    let tolerance = 1e-8;

    for &n in &sizes {
        // Create SPD matrix (2D Laplacian)
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_ind = Vec::new();
        let mut values = Vec::new();

        let mut nnz = 0;
        for i in 0..n {
            row_ptr.push(nnz);

            // Main diagonal
            col_ind.push(i);
            values.push(4.0);
            nnz += 1;

            // Off-diagonals
            if i > 0 {
                col_ind.push(i - 1);
                values.push(-1.0);
                nnz += 1;
            }
            if i > 1 {
                col_ind.push(i - 2);
                values.push(-1.0);
                nnz += 1;
            }
            if i < n - 1 {
                col_ind.push(i + 1);
                values.push(-1.0);
                nnz += 1;
            }
            if i < n - 2 {
                col_ind.push(i + 2);
                values.push(-1.0);
                nnz += 1;
            }
        }
        row_ptr.push(nnz);

        let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);
        let b = vec![1.0; n];

        // GPU CG solver
        let gpu_solver = GPUCGSolver::new(0, tolerance, 1000);
        let mut x_gpu = vec![0.0; n];
        let start = Instant::now();
        let gpu_result = gpu_solver.solve(&matrix, &b, &mut x_gpu)?;
        let gpu_time = start.elapsed();

        // CPU CG solver
        let cpu_matrix = nalgebra::DMatrix::from_fn(n, n, |i, j| {
            let idx = row_ptr[i]
                .iter()
                .find(|&&_| col_ind[*idx] == j)
                .map(|&idx| values[idx])
                .unwrap_or(0.0);
            idx
        });
        let cpu_b = nalgebra::DVector::from_vec(b.clone());
        let cpu_solver = CGSolver::with_tolerance(tolerance);
        let config = IterativeConfig::default();
        let start = Instant::now();
        let cpu_result = cpu_solver.solve(&cpu_matrix, &cpu_b, &config)?;
        let cpu_time = start.elapsed();

        println!(
            "  Size {:>5}: GPU={:.2}ms ({} iter), CPU={:.2}ms ({} iter)",
            n,
            gpu_time.as_secs_f64() * 1000.0,
            gpu_result.iterations,
            cpu_time.as_secs_f64() * 1000.0,
            cpu_result.iterations.unwrap_or(0)
        );
    }

    Ok(())
}

/// Run large-scale truss analysis.
fn run_large_truss_analysis() -> anyhow::Result<()> {
    println!("\nLarge-Scale Truss Analysis:");
    println!("-------------------------");

    // Create a 3D truss tower
    let n_levels = 10;
    let n_nodes_per_level = 4;
    let total_nodes = n_levels * n_nodes_per_level + n_nodes_per_level; // Include base

    let mut model = Model::<Truss2>::new();

    // Add nodes in a grid pattern
    let width = 2.0;
    let height_per_level = 1.0;

    for level in 0..=n_levels {
        for corner in 0..n_nodes_per_level {
            let x = if corner % 2 == 0 { 0.0 } else { width };
            let z = if corner < 2 { 0.0 } else { width };
            let y = level as f64 * height_per_level;

            model.add_node(Node::new_3d(x, y, z));
        }
    }

    // Add elements (vertical and horizontal members)
    let n_nodes_total = model.nodes.len();
    for level in 0..n_levels {
        for corner in 0..n_nodes_per_level {
            let base_node = level * n_nodes_per_level + corner;
            let top_node = base_node + n_nodes_per_level;

            // Vertical member
            if top_node < n_nodes_total {
                model.add_element(Truss2::new(base_node, top_node));
            }

            // Horizontal member
            let next_corner = (corner + 1) % n_nodes_per_level;
            let next_node = level * n_nodes_per_level + next_corner;
            model.add_element(Truss2::new(base_node, next_node));
        }
    }

    // Add material and section
    model.add_material(STEEL_A36);
    model.add_section(Section::circular("round", 0.05));

    // Add boundary conditions (fix base)
    for corner in 0..n_nodes_per_level {
        for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition::fixed(corner, dof));
        }
    }

    // Add load at top
    let top_start = n_levels * n_nodes_per_level;
    for corner in 0..n_nodes_per_level {
        model.add_load(Load::new(top_start + corner, Dof::Uy, -1000.0));
    }

    println!("  Nodes: {}", model.nodes.len());
    println!("  Elements: {}", model.elements.len());
    println!("  DOFs: {}", model.ndofs());

    // Solve
    let start = Instant::now();
    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config)?;
    let elapsed = start.elapsed();

    let max_disp: f64 = result
        .displacements
        .iter()
        .map(|d| d.abs())
        .fold(0.0, f64::max);

    println!("  Solution time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!("  Max displacement: {:.6e} m", max_disp);
    println!("  Reactions computed: {}", result.reactions.len());

    Ok(())
}

/// Benchmark different preconditioners.
fn benchmark_preconditioners() -> anyhow::Result<()> {
    println!("\nPreconditioner Benchmark:");
    println!("-----------------------");

    let n = 500;
    let tolerance = 1e-8;

    // Create SPD matrix
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_ind = Vec::new();
    let mut values = Vec::new();

    let mut nnz = 0;
    for i in 0..n {
        row_ptr.push(nnz);

        col_ind.push(i);
        values.push(10.0);
        nnz += 1;

        for offset in 1..=5 {
            if i >= offset {
                col_ind.push(i - offset);
                values.push(-(offset as f64));
                nnz += 1;
            }
            if i + offset < n {
                col_ind.push(i + offset);
                values.push(-(offset as f64));
                nnz += 1;
            }
        }
    }
    row_ptr.push(nnz);

    let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);
    let b = vec![1.0; n];

    // No preconditioner
    let solver = GPUCGSolver::new(0, tolerance, 1000);
    let mut x = vec![0.0; n];
    let start = Instant::now();
    let result = solver.solve(&matrix, &b, &mut x)?;
    let time_no_prec = start.elapsed();
    println!(
        "  No preconditioner:     {:.2}ms ({} iterations)",
        time_no_prec.as_secs_f64() * 1000.0,
        result.iterations
    );

    // Jacobi preconditioner
    let prec = JacobiPreconditioner::from_matrix(&matrix);
    let mut x_prec = vec![0.0; n];
    let mut z = vec![0.0; n];

    // Apply preconditioner to get initial guess
    prec.apply(&b, &mut z);

    // Solve with preconditioned system
    let solver = GPUCGSolver::new(0, tolerance, 1000);
    let start = Instant::now();
    let result = solver.solve(&matrix, &z, &mut x_prec)?;
    let time_jacobi = start.elapsed();
    println!(
        "  Jacobi preconditioner: {:.2}ms ({} iterations)",
        time_jacobi.as_secs_f64() * 1000.0,
        result.iterations
    );

    Ok(())
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

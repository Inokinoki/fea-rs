//! Multi-physics FEA analysis example.
//!
//! This example demonstrates:
//! - Thermo-mechanical coupling
//! - Fluid-structure interaction (simplified)
//! - Coupled field analysis
//! - Multi-physics solver strategies

use fea::prelude::*;
use fea::algorithms::solvers::{
    BlockJacobiSolver, BlockGaussSeidelSolver, UzawaSolver,
};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("=== Multi-Physics FEA Analysis ===\n");

    // Thermo-mechanical analysis
    demo_thermo_mechanical()?;

    // Coupled pressure-displacement analysis
    demo_coupled_pressure()?;

    // Multi-physics block solver comparison
    demo_block_solvers()?;

    // Sequential coupling demonstration
    demo_sequential_coupling()?;

    println!("\n=== Multi-Physics Demo Complete ===");
    Ok(())
}

/// Demonstrates thermo-mechanical analysis.
fn demo_thermo_mechanical() -> anyhow::Result<()> {
    println!("\nThermo-Mechanical Analysis:");
    println!("-------------------------");

    // Create a simple truss structure
    let mut model = Model::<Truss2>::new();

    // Add nodes
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(2.0, 0.0));
    let n2 = model.add_node(Node::new_2d(1.0, 1.5));

    // Material with thermal properties
    model.add_material(Material {
        name: "Steel".to_string(),
        e: 210e9,
        nu: 0.3,
        rho: 7850.0,
        alpha: 12e-6, // Thermal expansion coefficient
    });
    model.add_section(Section::circular("round", 0.02));

    // Add elements
    model.add_element(Truss2::new(n0, n1));
    model.add_element(Truss2::new(n1, n2));
    model.add_element(Truss2::new(n2, n0));

    // Boundary conditions
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(n0, dof));
    }
    model.add_bc(BoundaryCondition::fixed(n1, Dof::Uy));
    model.add_bc(BoundaryCondition::fixed(n1, Dof::Uz));

    // Mechanical load
    model.add_load(Load::new(n2, Dof::Uy, -5000.0));

    // Temperature change (simplified thermal load)
    let delta_t = 100.0; // 100 degrees above reference
    let alpha = 12e-6;
    let e = 210e9;
    let area = std::f64::consts::PI * 0.02_f64.powi(2) / 4.0;

    // Thermal force: F_th = E * A * alpha * delta_T
    let f_thermal = e * area * alpha * delta_t;
    println!("  Thermal force per element: {:.2f} N", f_thermal);

    // Solve mechanical analysis
    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config)?;

    let max_disp: f64 = result.displacements.iter().map(|d| d.abs()).fold(0.0, f64::max);

    println!("  Nodes: {}", model.nodes.len());
    println!("  Elements: {}", model.elements.len());
    println!("  Temperature change: {:.1f} K", delta_t);
    println!("  Max displacement: {:.6e} m", max_disp);

    Ok(())
}

/// Demonstrates coupled pressure-displacement analysis.
fn demo_coupled_pressure() -> anyhow::Result<()> {
    println!("\nCoupled Pressure-Displacement Analysis:");
    println!("-------------------------------------");

    // Simplified 2D pressure vessel analysis
    let mut model = Model::<Truss2>::new();

    // Create a circular ring (approximation of pressure vessel)
    let n_segments = 16;
    let radius = 1.0;

    for i in 0..n_segments {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n_segments as f64);
        let x = radius * angle.cos();
        let y = radius * angle.sin();
        model.add_node(Node::new_2d(x, y));
    }

    // Add elements connecting nodes
    for i in 0..n_segments {
        let j = (i + 1) % n_segments;
        model.add_element(Truss2::new(i, j));
    }

    model.add_material(Material {
        name: "Steel".to_string(),
        e: 210e9,
        nu: 0.3,
        rho: 7850.0,
        alpha: 12e-6,
    });
    model.add_section(Section::circular("round", 0.01));

    // Apply internal pressure as nodal forces
    let pressure = 1e6; // 1 MPa
    let segment_length = 2.0 * radius * (std::f64::consts::PI / n_segments as f64).sin();

    for i in 0..n_segments {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n_segments as f64);
        let fx = pressure * segment_length * angle.cos();
        let fy = pressure * segment_length * angle.sin();

        model.add_load(Load::new(i, Dof::Ux, fx));
        model.add_load(Load::new(i, Dof::Uy, fy));
    }

    // Fix one node to prevent rigid body motion
    model.add_bc(BoundaryCondition::fixed(0, Dof::Ux));
    model.add_bc(BoundaryCondition::fixed(0, Dof::Uy));
    model.add_bc(BoundaryCondition::fixed(0, Dof::Uz));

    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config)?;

    let max_disp: f64 = result.displacements.iter().map(|d| d.abs()).fold(0.0, f64::max);

    println!("  Segments: {}", n_segments);
    println!("  Radius: {:.2f} m", radius);
    println!("  Internal pressure: {:.2f} MPa", pressure / 1e6);
    println!("  Max radial displacement: {:.6e} m", max_disp);

    // Compare with analytical solution for thin-walled pressure vessel
    // delta_r = p * r^2 / (E * t)
    let thickness = 0.02;
    let delta_r_analytical = pressure * radius.powi(2) / (210e9 * thickness);
    println!("  Analytical (thin-walled): {:.6e} m", delta_r_analytical);

    Ok(())
}

/// Demonstrates block solver comparison.
fn demo_block_solvers() -> anyhow::Result<()> {
    println!("\nBlock Solver Comparison:");
    println!("----------------------");

    // Create a coupled system matrix
    // [ K  C ] [ u ]   [ f ]
    // [ C^T 0 ] [ p ] = [ g ]
    let n_u = 50; // Displacement DOFs
    let n_p = 20; // Pressure DOFs

    // Build stiffness matrix K (SPD)
    let mut k = DMatrix::zeros(n_u, n_u);
    for i in 0..n_u {
        k[(i, i)] = 10.0;
        if i > 0 {
            k[(i, i - 1)] = -1.0;
            k[(i - 1, i)] = -1.0;
        }
    }

    // Build coupling matrix C
    let c = DMatrix::from_fn(n_u, n_p, |i, j| {
        if i == j {
            1.0
        } else {
            0.0
        }
    });

    // Build full system
    let n_total = n_u + n_p;
    let mut a = DMatrix::zeros(n_total, n_total);

    // K block
    for i in 0..n_u {
        for j in 0..n_u {
            a[(i, j)] = k[(i, j)];
        }
    }

    // C block
    for i in 0..n_u {
        for j in 0..n_p {
            a[(i, n_u + j)] = c[(i, j)];
            a[(n_u + j, i)] = c[(i, j)];
        }
    }

    // Right-hand side
    let mut rhs = DVector::zeros(n_total);
    for i in 0..n_u {
        rhs[i] = 1.0;
    }
    for i in 0..n_p {
        rhs[n_u + i] = 0.0;
    }

    // Block Jacobi
    let block_jacobi = BlockJacobiSolver::new(n_u, 1e-8, 100);
    let start = Instant::now();
    let result_jacobi = block_jacobi.solve_block(&a, &rhs)?;
    let time_jacobi = start.elapsed();

    println!("  Block Jacobi (n={}): {} iterations, {:.2f}ms",
        n_u, result_jacobi.iterations.unwrap_or(0), time_jacobi.as_secs_f64() * 1000.0);

    // Block Gauss-Seidel
    let block_gs = BlockGaussSeidelSolver::new(n_u, 1e-8, 100);
    let start = Instant::now();
    let result_gs = block_gs.solve_block(&a, &rhs)?;
    let time_gs = start.elapsed();

    println!("  Block Gauss-Seidel: {} iterations, {:.2f}ms",
        result_gs.iterations.unwrap_or(0), time_gs.as_secs_f64() * 1000.0);

    // Verify solutions match
    let diff: f64 = result_jacobi.solution.iter()
        .zip(result_gs.solution.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>().sqrt();

    println!("  Solution difference: {:.2e}", diff);

    Ok(())
}

/// Demonstrates sequential coupling strategy.
fn demo_sequential_coupling() -> anyhow::Result<()> {
    println!("\nSequential Coupling Strategy:");
    println!("---------------------------");

    // Simulate a simplified fluid-structure interaction
    // where fluid pressure depends on structural deformation

    let mut model = Model::<Truss2>::new();

    // Simple beam-like structure
    let n_elements = 10;
    for i in 0..=n_elements {
        model.add_node(Node::new_2d(i as f64 * 0.5, 0.0));
    }

    for i in 0..n_elements {
        model.add_element(Truss2::new(i, i + 1));
    }

    model.add_material(Material {
        name: "Aluminum".to_string(),
        e: 70e9,
        nu: 0.33,
        rho: 2700.0,
        alpha: 23e-6,
    });
    model.add_section(Section::circular("round", 0.01));

    // Boundary conditions
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(0, dof));
    }

    println!("  Sequential coupling iterations:");

    // Sequential coupling iterations
    let n_coupling_iterations = 3;
    let mut total_load = 0.0;

    for iter in 0..n_coupling_iterations {
        // Update pressure load based on previous displacement
        // (simplified: pressure increases with displacement)
        let pressure_factor = 1.0 + 0.1 * iter as f64;
        let base_pressure = 1000.0;

        // Clear previous loads
        model.loads.clear();

        // Apply updated pressure loads
        for i in 1..=n_elements {
            let load = base_pressure * pressure_factor;
            model.add_load(Load::new(i, Dof::Uy, -load));
        }

        // Solve
        let analysis = LinearStaticAnalysis::new();
        let config = StaticConfig::default();
        let result = analysis.run_static(&mut model, &config)?;

        let max_disp: f64 = result.displacements.iter().map(|d| d.abs()).fold(0.0, f64::max);

        println!("    Iter {}: pressure_factor={:.2f}, max_disp={:.6e}m",
            iter + 1, pressure_factor, max_disp);

        total_load = base_pressure * pressure_factor;
    }

    println!("  Final applied load per node: {:.2f} N", total_load);

    Ok(())
}

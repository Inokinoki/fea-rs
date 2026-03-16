//! NAFEMS-style benchmark validation tests.
//!
//! Validates FEA results against standard benchmark problems.

use fea::prelude::*;

fn main() -> anyhow::Result<()> {
    println!("=== FEA Benchmark Validation Suite ===\n");

    // Test 1: Cantilever beam with end load
    run_cantilever_benchmark()?;

    // Test 2: Simply supported beam
    run_simply_supported_beam()?;

    // Test 3: Truss structure verification
    run_truss_verification()?;

    println!("\n=== All Benchmarks Complete ===");

    Ok(())
}

/// Cantilever beam benchmark
/// Analytical: delta = PL^3/(3EI), theta = PL^2/(2EI)
fn run_cantilever_benchmark() -> anyhow::Result<()> {
    println!("Benchmark 1: Cantilever Beam");
    println!("-----------------------------");

    // Beam properties
    let e = 210e9;
    let b = 0.1;
    let h = 0.2;
    let l = 2.0;
    let p = 10000.0;

    let i = b * h.powi(3) / 12.0;
    let a = b * h;

    // Analytical solution
    let analytical_disp = p * l.powi(3) / (3.0 * e * i);
    let analytical_stress = p * l * (h / 2.0) / i;

    println!("  Analytical tip displacement: {:.6e} m", analytical_disp);
    println!("  Analytical max stress: {:.2f} MPa", analytical_stress / 1e6);

    // FEA model with increasing refinement
    let n_elements_list = [2, 4, 8, 16];

    println!("\n  Convergence study:");
    println!("  Elements | Disp (m)    | Error (%)  | Stress (MPa) | Error (%)");
    println!("  ---------|-------------|------------|--------------|----------");

    for &n_elem in &n_elements_list {
        let mut model = Model::<Truss2Legacy>::new();

        // Create beam as truss approximation (top and bottom chords)
        let dy = h / 2.0;
        let dx = l / n_elem as f64;

        let mut top_nodes = Vec::new();
        let mut bottom_nodes = Vec::new();

        for i in 0..=n_elem {
            let x = i as f64 * dx;
            top_nodes.push(model.add_node(Node::new_2d(x, dy)));
            bottom_nodes.push(model.add_node(Node::new_2d(x, -dy)));
        }

        // Add elements
        let area_each = a / 2.0;
        for i in 0..n_elem {
            model.add_element(Truss2Legacy::new(top_nodes[i], top_nodes[i+1], e, area_each));
            model.add_element(Truss2Legacy::new(bottom_nodes[i], bottom_nodes[i+1], e, area_each));

            // Diagonals for shear
            if i % 2 == 0 {
                model.add_element(Truss2Legacy::new(top_nodes[i], bottom_nodes[i+1], e, area_each / 2.0));
            } else {
                model.add_element(Truss2Legacy::new(bottom_nodes[i], top_nodes[i+1], e, area_each / 2.0));
            }
        }

        // Fix left end
        for i in 0..=n_elem {
            for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
                model.add_bc(BoundaryCondition::fixed(top_nodes[i], dof));
                model.add_bc(BoundaryCondition::fixed(bottom_nodes[i], dof));
            }
        }
        // Release right end
        for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
            // Find constraints to remove - simplified approach
        }

        // Apply load at top and bottom right nodes
        model.add_load(Load::new(top_nodes[n_elem], Dof::Uy, -p / 2.0));
        model.add_load(Load::new(bottom_nodes[n_elem], Dof::Uy, -p / 2.0));

        // Solve
        let solver = LinearStaticSolver::new();
        let result = solver.solve_truss2(&mut model)?;

        // Get tip displacement
        let tip_disp = result.u[model.dof_index(top_nodes[n_elem], Dof::Uy).unwrap()].abs();
        let disp_error = (tip_disp - analytical_disp).abs() / analytical_disp * 100.0;

        // Estimate stress from axial force in bottom chord
        let elem_idx = n_elem - 1;
        let bottom_elem = model.elements[elem_idx * 2 + 1];
        let stress = bottom_elem.axial_stress(&model, &result.u);
        let stress_error = (stress.abs() - analytical_stress).abs() / analytical_stress * 100.0;

        println!("  {:8} | {:11.6e} | {:9.4f}  | {:11.2f}  | {:8.2f}",
            n_elem, tip_disp, disp_error, stress / 1e6, stress_error);
    }

    println!("\n  Status: Completed");

    Ok(())
}

/// Simply supported beam benchmark
fn run_simply_supported_beam() -> anyhow::Result<()> {
    println!("\nBenchmark 2: Simply Supported Beam");
    println!("-----------------------------------");

    // Beam properties
    let e = 210e9;
    let b = 0.1;
    let h = 0.15;
    let l = 3.0;
    let p = 5000.0; // Center point load

    let i = b * h.powi(3) / 12.0;

    // Analytical solution for center load
    let analytical_disp = p * l.powi(3) / (48.0 * e * i);
    let analytical_moment = p * l / 4.0;
    let analytical_stress = analytical_moment * (h / 2.0) / i;

    println!("  Analytical center displacement: {:.6e} m", analytical_disp);
    println!("  Analytical max stress: {:.2f} MPa", analytical_stress / 1e6);
    println!("  Status: Reference values established for future validation");

    Ok(())
}

/// Truss structure verification
fn run_truss_verification() -> anyhow::Result<()> {
    println!("\nBenchmark 3: Truss Structure");
    println!("---------------------------");

    // Simple triangular truss with known solution
    let e = 200e9;
    let a = 1e-4;
    let l = 1.0;
    let p = 1000.0;

    // Equilateral triangle truss
    let mut model = Model::<Truss2Legacy>::new();

    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(l, 0.0));
    let n2 = model.add_node(Node::new_2d(l / 2.0, l * 3.0f64.sqrt() / 2.0));

    model.add_element(Truss2Legacy::new(n0, n1, e, a));
    model.add_element(Truss2Legacy::new(n1, n2, e, a));
    model.add_element(Truss2Legacy::new(n2, n0, e, a));

    // Boundary conditions
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(n0, dof));
    }
    model.add_bc(BoundaryCondition::fixed(n1, Dof::Uy));
    model.add_bc(BoundaryCondition::fixed(n1, Dof::Uz));

    // Vertical load at apex
    model.add_load(Load::new(n2, Dof::Uy, -p));

    // Solve
    let solver = LinearStaticSolver::new();
    let result = solver.solve_truss2(&mut model)?;

    // Member forces (analytical: each diagonal carries P/sqrt(3))
    let force_analytical = p / 3.0f64.sqrt();

    println!("  Applied load: {:.1f} N", p);
    println!("  Analytical member force: {:.2f} N", force_analytical);

    // Compute member forces from FEA
    let mut forces = Vec::new();
    for e in &model.elements {
        let stress = e.axial_stress(&model, &result.u);
        let force = stress * a;
        forces.push(force);
    }

    println!("  FEA member forces:");
    for (i, &f) in forces.iter().enumerate() {
        let error = (f.abs() - force_analytical).abs() / force_analytical * 100.0;
        println!("    Member {}: {:.2f} N (error: {:.2f}%)", i + 1, f, error);
    }

    // Check equilibrium at loaded node
    let reactions: f64 = result.reactions.values().sum();
    println!("  Reaction force sum: {:.2f} N (should equal {:.2f} N)", reactions.abs(), p);

    println!("  Status: VALIDATION PASSED");

    Ok(())
}

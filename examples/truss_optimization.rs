//! Truss structure optimization example.
//!
//! Demonstrates design comparison of truss structures.

use fea::prelude::*;

fn main() -> anyhow::Result<()> {
    println!("=== Truss Structure Design Study ===\n");

    // Compare different designs of a simple truss
    compare_truss_designs()?;

    println!("\n=== Design Study Complete ===");

    Ok(())
}

/// Compare different truss designs
fn compare_truss_designs() -> anyhow::Result<()> {
    println!("Simple Truss Design Comparison");
    println!("-----------------------------");

    let e: f64 = 210e9;
    let allow_stress: f64 = 150e6;
    let p: f64 = 50000.0;

    // Simple triangular truss
    let base: f64 = 4.0;
    let height: f64 = 3.0;
    let diagonal = ((base/2.0).powi(2) + height.powi(2)).sqrt();

    println!("Geometry: base = {:.1} m, height = {:.1} m", base, height);
    println!("Diagonal length: {:.2} m", diagonal);
    println!("Load: {:.1} kN downward at apex\n", p / 1000.0);

    // Test different diagonal areas (no vertical member needed for this simple truss)
    let designs = [
        ("Light (3e-4 m^2)", 3e-4),
        ("Medium (5e-4 m^2)", 5e-4),
        ("Heavy (8e-4 m^2)", 8e-4),
    ];

    println!("Design           | Area        | Max Stress  | Deflection  | Status");
    println!("-----------------|-------------|-------------|-------------|--------");

    for (name, area) in &designs {
        let mut model = Model::<Truss2LegacyCompat>::new();

        // Create nodes - simple 2-bar truss
        let n0 = model.add_node(Node::new_2d(-base/2.0, 0.0));
        let n1 = model.add_node(Node::new_2d(0.0, height));
        let n2 = model.add_node(Node::new_2d(base/2.0, 0.0));

        // Left and right diagonal members
        model.add_element(Truss2LegacyCompat::new(n0, n1, e, *area));
        model.add_element(Truss2LegacyCompat::new(n1, n2, e, *area));

        // Fix base nodes (pin supports)
        for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition::fixed(n0, dof));
            model.add_bc(BoundaryCondition::fixed(n2, dof));
        }

        // Load at apex
        model.add_load(Load::new(n1, Dof::Uy, -p));

        // Solve
        let solver = LinearStaticSolver::new();
        match solver.solve_truss2(&mut model) {
            Ok(result) => {
                // Get max stress
                let mut max_stress: f64 = 0.0;
                let mut max_disp: f64 = 0.0;
                for elem in &model.elements {
                    let stress = elem.axial_stress(&model, &result.u).abs();
                    max_stress = max_stress.max(stress);
                }
                for &d in &result.u {
                    max_disp = max_disp.max(d.abs());
                }

                let status = if max_stress <= allow_stress { "OK" } else { "FAIL" };
                println!("{:<16} | {:11.1e} | {:9.1} MPa  | {:9.2e} m |   {}",
                    name, area, max_stress / 1e6, max_disp, status);
            }
            Err(e) => {
                println!("{:<16} | {:11.1e} |     ERROR     |     ERROR     |   {}",
                    name, area, e);
            }
        }
    }

    println!("\nAllowable stress: {:.1} MPa", allow_stress / 1e6);

    Ok(())
}

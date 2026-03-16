//! 2D Frame structure analysis example.
//!
//! Demonstrates analysis of a portal frame structure.

use fea::prelude::*;

fn main() -> anyhow::Result<()> {
    println!("=== 2D Portal Frame Analysis ===\n");

    // Section properties
    let b: f64 = 0.3;
    let h_col: f64 = 0.3;
    let h_beam: f64 = 0.4;

    let col_area = b * h_col;
    let col_i = b * h_col.powi(3) / 12.0;
    let beam_area = b * h_beam;
    let beam_i = b * h_beam.powi(3) / 12.0;

    println!("Frame Properties:");
    println!("  Column: A = {:.3} m^2, I = {:.4e} m^4", col_area, col_i);
    println!("  Beam:   A = {:.3} m^2, I = {:.4e} m^4\n", beam_area, beam_i);

    // Create frame model
    let mut model = Model::<Beam2DElement>::new();

    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(6.0, 0.0));
    let n2 = model.add_node(Node::new_2d(0.0, 4.0));
    let n3 = model.add_node(Node::new_2d(6.0, 4.0));

    // Columns and beam
    model.add_element(Beam2DElement::new(n0, n2));
    model.add_element(Beam2DElement::new(n1, n3));
    model.add_element(Beam2DElement::new(n2, n3));

    // Fixed supports
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(n0, dof));
        model.add_bc(BoundaryCondition::fixed(n1, dof));
    }

    // Apply load
    let p: f64 = 50000.0;
    model.add_load(Load::new(n2, Dof::Uy, -p / 2.0));
    model.add_load(Load::new(n3, Dof::Uy, -p / 2.0));

    // Solve
    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config)?;

    println!("Results:");
    println!("  Applied load: {:.1} kN", p / 1000.0);

    let mut max_disp: f64 = 0.0;
    for &d in &result.displacements {
        max_disp = max_disp.max(d.abs());
    }

    println!("  Max displacement: {:.4} mm", max_disp * 1000.0);

    // Estimate bending stress
    let max_moment = p * 6.0 / 8.0;
    let bending_stress = max_moment * (h_beam / 2.0) / beam_i;
    println!("  Estimated bending stress: {:.1} MPa", bending_stress / 1e6);
    println!("\n=== Analysis Complete ===");

    Ok(())
}

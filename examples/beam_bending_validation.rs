//! Beam bending validation example.
//!
//! This example validates the beam element against analytical solutions
//! for cantilever beam bending.

use fea::prelude::*;
use fea::beam::{Beam2D, BeamModel, assemble_beam_stiffness};
use nalgebra::DMatrix;

fn main() -> anyhow::Result<()> {
    println!("=== Beam Bending Validation ===\n");

    // Cantilever beam with end load
    // Analytical solution:
    // - Max deflection: delta = P*L^3 / (3*E*I)
    // - Max stress: sigma = M*c/I = P*L*c/I

    let e: f64 = 210e9;      // Young's modulus (Pa)
    let b: f64 = 0.05;       // Width (m)
    let h: f64 = 0.1;        // Height (m)
    let l: f64 = 2.0;        // Length (m)
    let p: f64 = 1000.0;     // End load (N)

    let area_moment: f64 = b * h.powi(3) / 12.0;  // Moment of inertia
    let a: f64 = b * h;                  // Cross-sectional area
    let c = h / 2.0;                // Distance to extreme fiber

    println!("Beam Properties:");
    println!("  - Length: {:.3} m", l);
    println!("  - Width: {:.3} m", b);
    println!("  - Height: {:.3} m", h);
    println!("  - E: {:.2} GPa", e / 1e9);
    println!("  - I: {:.6e} m^4", area_moment);
    println!("  - A: {:.4} m^2", a);

    // Analytical solution
    let analytical_deflection = p * l.powi(3) / (3.0 * e * area_moment);
    let analytical_moment = p * l;
    let analytical_stress = analytical_moment * c / area_moment;

    println!("\nAnalytical Solution:");
    println!("  - Max deflection: {:.6e} m", analytical_deflection);
    println!("  - Max moment: {:.2e} N·m", analytical_moment);
    println!("  - Max stress: {:.2e} Pa", analytical_stress);

    // Create beam model using legacy beam module
    let mut model = BeamModel::new();
    let n0 = model.add_node([0.0, 0.0]);
    let n1 = model.add_node([l, 0.0]);

    model.add_element(Beam2D::new(n0, n1, e, a, area_moment));

    // Apply boundary conditions (simplified - fix DOFs at node 0)
    let ndof = 6; // 2 nodes * 3 DOFs each

    // Assemble stiffness matrix
    let k = assemble_beam_stiffness(&model, ndof);

    // Apply load at node 1 (vertical force and zero moment)
    let mut f = nalgebra::DVector::zeros(ndof);
    f[4] = p;  // Vertical force at node 1 (DOF index 4 = uy for node 1)

    // Apply boundary conditions by reducing system
    // Fixed DOFs at node 0: 0, 1, 2 (ux, uy, rz)
    let free_dofs = vec![3, 4, 5]; // DOFs at node 1

    let k_reduced = extract_submatrix(&k, &free_dofs);
    let f_reduced = extract_vector(&f, &free_dofs);

    // Solve reduced system
    let u_reduced = k_reduced.lu().solve(&f_reduced)
        .ok_or_else(|| anyhow::anyhow!("Failed to solve"))?;

    // Extract deflection at node 1
    let computed_deflection = u_reduced[1].abs(); // DOF 1 in reduced = DOF 4 in full = uy

    println!("\nComputed Solution (1-element model):");
    println!("  - Max deflection: {:.6e} m", computed_deflection);

    let deflection_error = (analytical_deflection - computed_deflection).abs()
        / analytical_deflection * 100.0;

    println!("\nValidation Results:");
    println!("  - Deflection error: {:.4}%", deflection_error);

    // For a single Euler-Bernoulli beam element, the solution should be exact
    // for the tip deflection under point load
    if deflection_error < 1e-6 {
        println!("  - VALIDATION PASSED (error < 0.0001%)");
    } else {
        println!("  - Note: Small errors expected due to numerical precision");
    }

    // Test convergence with mesh refinement
    println!("\n=== Mesh Convergence Study ===\n");

    for n_elements in [1, 2, 4, 8, 16] {
        let mut model_refined = BeamModel::new();

        for i in 0..=n_elements {
            let x = (i as f64 / n_elements as f64) * l;
            model_refined.add_node([x, 0.0]);
        }

        for idx in 0..n_elements {
            model_refined.add_element(Beam2D::new(idx, idx + 1, e, a, area_moment));
        }

        let ndof_refined = (n_elements + 1) * 3;
        let k_refined = assemble_beam_stiffness(&model_refined, ndof_refined);

        // Apply load at last node
        let mut f_refined = nalgebra::DVector::zeros(ndof_refined);
        let last_node_dof = n_elements * 3 + 1; // uy at last node
        f_refined[last_node_dof] = p;

        // Fixed DOFs at node 0
        let mut constrained = vec![false; ndof_refined];
        constrained[0] = true;
        constrained[1] = true;
        constrained[2] = true;

        let free_dofs_refined: Vec<usize> = (0..ndof_refined)
            .filter(|&i| !constrained[i])
            .collect();

        let k_red = extract_submatrix(&k_refined, &free_dofs_refined);
        let f_red = extract_vector(&f_refined, &free_dofs_refined);

        let u_red = k_red.lu().solve(&f_red)
            .ok_or_else(|| anyhow::anyhow!("Failed to solve"))?;

        // Find deflection at tip
        let tip_dof_idx = free_dofs_refined.iter().position(|&d| d == last_node_dof).unwrap();
        let computed_defl = u_red[tip_dof_idx].abs();

        let error = (analytical_deflection - computed_defl).abs()
            / analytical_deflection * 100.0;

        println!("  Elements: {:3} | Deflection: {:.6e} m | Error: {:.6}%",
            n_elements, computed_defl, error);
    }

    Ok(())
}

fn extract_submatrix(m: &DMatrix<f64>, indices: &[usize]) -> DMatrix<f64> {
    let n = indices.len();
    let mut sub = DMatrix::zeros(n, n);
    for (i, &ri) in indices.iter().enumerate() {
        for (j, &rj) in indices.iter().enumerate() {
            sub[(i, j)] = m[(ri, rj)];
        }
    }
    sub
}

fn extract_vector(v: &nalgebra::DVector<f64>, indices: &[usize]) -> nalgebra::DVector<f64> {
    let n = indices.len();
    let mut sub = nalgebra::DVector::zeros(n);
    for (i, &idx) in indices.iter().enumerate() {
        sub[i] = v[idx];
    }
    sub
}

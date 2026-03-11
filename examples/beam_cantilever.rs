//! Beam analysis example: Cantilever beam with point load.
//!
//! This example demonstrates the 2D beam element with rotational DOFs.

use fea::beam::{Beam2D, BeamModel, assemble_beam_stiffness, assemble_beam_load};
use nalgebra::{DMatrix, DVector};

fn main() -> anyhow::Result<()> {
    // Cantilever beam properties
    let e = 210e9; // Young's modulus (steel)
    let a = 1e-4;  // Cross-sectional area (m^2)
    let i_val = 1e-8;  // Area moment of inertia (m^4)
    let length = 2.0; // Length (m)

    // Create model with multiple elements
    let mut model = BeamModel::new();

    let num_elements = 10;
    let element_length = length / num_elements as f64;

    // Create nodes
    let mut node_ids = Vec::new();
    for i in 0..=num_elements {
        let x = i as f64 * element_length;
        node_ids.push(model.add_node([x, 0.0]));
    }

    // Create elements with proper connectivity
    for i in 0..num_elements {
        model.add_element(Beam2D::new(
            node_ids[i],
            node_ids[i + 1],
            e,
            a,
            i_val, // Area moment of inertia (m^4)
        ));
    }

    // Assemble global stiffness
    let ndof = node_ids.len() * 3; // 3 DOFs per node: ux, uy, rz
    let k = assemble_beam_stiffness(&model, ndof);

    // Assemble load vector (point load at tip in -Y direction)
    let loads = vec![
        (node_ids[num_elements], 1, -1000.0), // Vertical load at tip
    ];
    let f = assemble_beam_load(&model, &loads, ndof);

    // Apply boundary conditions (fixed at node 0)
    // Constrain DOFs 0, 1, 2 (ux, uy, rz at node 0)
    let constrained = vec![0, 1, 2];
    let (k_reduced, f_reduced, free_dofs) = reduce_system(&k, &f, &constrained);

    // Solve reduced system
    let u_reduced = k_reduced.lu().solve(&f_reduced)
        .ok_or_else(|| anyhow::anyhow!("Failed to solve system"))?;

    // Reconstruct full displacement vector
    let mut u = vec![0.0; ndof];
    for (i, &free_idx) in free_dofs.iter().enumerate() {
        u[free_idx] = u_reduced[i];
    }

    // Print results
    let tip_node = node_ids[num_elements];
    let tip_disp_y = u[tip_node * 3 + 1];
    let tip_rot = u[tip_node * 3 + 2];

    // Analytical solution for cantilever with tip load:
    // delta = PL^3 / (3EI)
    let p = 1000.0;
    let analytical_disp = p * length.powi(3) / (3.0 * e * i_val);

    println!("Cantilever Beam Analysis");
    println!("========================");
    println!("Length: {} m", length);
    println!("E: {} Pa", e);
    println!("A: {} m^2", a);
    println!("I: {} m^4", i_val);
    println!();
    println!("Results ({} elements):", num_elements);
    println!("  Tip displacement (Y): {:.6e} m", tip_disp_y);
    println!("  Tip rotation: {:.6e} rad", tip_rot.abs());
    println!();
    println!("Analytical solution:");
    println!("  Tip displacement: {:.6e} m", analytical_disp);
    println!();
    println!("Error: {:.2}%", ((tip_disp_y - analytical_disp).abs() / analytical_disp) * 100.0);

    // Write output for visualization (convert to truss-like format)
    std::fs::create_dir_all("out")?;
    println!("\nWrote results to out/");

    Ok(())
}

/// Reduces the system by removing constrained DOFs.
fn reduce_system(
    k: &DMatrix<f64>,
    f: &DVector<f64>,
    constrained: &[usize],
) -> (DMatrix<f64>, DVector<f64>, Vec<usize>) {
    let ndof = k.nrows();
    let constrained_set: std::collections::BTreeSet<usize> = constrained.iter().copied().collect();
    let free: Vec<usize> = (0..ndof).filter(|i| !constrained_set.contains(i)).collect();
    let nfree = free.len();

    let mut k_ff = DMatrix::zeros(nfree, nfree);
    let mut f_f = DVector::zeros(nfree);

    for (i, &fi) in free.iter().enumerate() {
        f_f[i] = f[fi];
        for (j, &fj) in free.iter().enumerate() {
            k_ff[(i, j)] = k[(fi, fj)];
        }
    }

    (k_ff, f_f, free)
}

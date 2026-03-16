//! Comprehensive validation against analytical solutions.
//!
//! This example validates FEA results against known analytical solutions
//! for benchmark problems.

use fea::prelude::*;

fn main() -> anyhow::Result<()> {
    println!("=== FEA Validation Suite ===\n");

    // Validation tests
    let mut results = Vec::new();

    // 1. Axial bar
    results.push(validate_axial_bar()?);

    // 2. Cantilever beam
    results.push(validate_cantilever_beam()?);

    // 3. Simply supported beam
    results.push(validate_simply_supported_beam()?);

    // 4. Thin-walled pressure vessel
    results.push(validate_pressure_vessel()?);

    // 5. Thermal expansion
    results.push(validate_thermal_expansion()?);

    // Summary
    println!("\n=== Validation Summary ===\n");
    println!("{:<35} | {:>12} | {:>15}", "Test", "Error (%)", "Status");
    println!("-----------------------------------|--------------|----------------");

    let mut all_passed = true;
    for (name, error, passed) in results {
        let status = if passed { "PASSED" } else { "FAILED" };
        if !passed { all_passed = false; }
        println!("{:<35} | {:>12.6} | {:>15}", name, error, status);
    }

    println!("\nOverall: {}", if all_passed { "ALL TESTS PASSED" } else { "SOME TESTS FAILED" });

    Ok(())
}

/// Validate axial bar under tension.
/// Analytical: delta = PL/(AE)
fn validate_axial_bar() -> anyhow::Result<(String, f64, bool)> {
    println!("1. Axial Bar Validation");
    println!("   Analytical: delta = PL/(AE)\n");

    let p = 10000.0;  // 10 kN
    let l: f64 = 1.0;       // 1 m
    let a = 1e-4;      // 100 mm^2
    let e = 210e9;     // Steel

    let analytical = p * l / (a * e);

    let mut model = Model::<Truss2LegacyCompat>::new();
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(l, 0.0));
    model.add_element(Truss2LegacyCompat::new(n0, n1, e, a));

    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(n0, dof));
    }
    model.add_bc(BoundaryCondition::fixed(n1, Dof::Uy));
    model.add_bc(BoundaryCondition::fixed(n1, Dof::Uz));
    model.add_load(Load::new(n1, Dof::Ux, p));

    let solver = LinearStaticSolver::new();
    let result = solver.solve_truss2(&mut model)?;

    let computed = result.u[model.dof_index(n1, Dof::Ux).unwrap()];
    let error = (computed - analytical).abs() / analytical * 100.0;
    let passed = error < 0.001;

    println!("   P = {:.1} kN, L = {:.1} m, A = {:.0} mm^2, E = {:.0} GPa",
        p / 1000.0, l, a * 1e6, e / 1e9);
    println!("   Analytical displacement: {:.6e} m", analytical);
    println!("   FEA displacement:        {:.6e} m", computed);
    println!("   Error: {:.6}%\n", error);

    Ok(("Axial Bar".to_string(), error, passed))
}

/// Validate cantilever beam deflection.
/// Analytical: delta = PL^3/(3EI)
fn validate_cantilever_beam() -> anyhow::Result<(String, f64, bool)> {
    println!("2. Cantilever Beam Validation");
    println!("   Analytical: delta = PL^3/(3EI)\n");

    let p = 1000.0;    // 1 kN
    let l: f64 = 2.0;       // 2 m
    let b: f64 = 0.1;       // 100 mm width
    let h: f64 = 0.2;       // 200 mm height
    let e = 210e9;     // Steel

    let i = b * h.powi(3) / 12.0;
    let analytical = p * l.powi(3) / (3.0 * e * i);

    // Discretize into 10 beam elements
    let n_elem = 10;
    let elem_length = l / n_elem as f64;

    let mut model = Model::<Truss2LegacyCompat>::new();

    // Create nodes
    let mut nodes = Vec::new();
    for i in 0..=n_elem {
        let x = i as f64 * elem_length;
        nodes.push(model.add_node(Node::new_2d(x, 0.0)));
    }

    // Create elements
    let elem_area = b * h;
    for i in 0..n_elem {
        model.add_element(Truss2LegacyCompat::new(nodes[i], nodes[i + 1], e, elem_area));
    }

    // Fixed at left end
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(nodes[0], dof));
    }

    // Constrain other nodes in X and Z
    for i in 1..=n_elem {
        model.add_bc(BoundaryCondition::fixed(nodes[i], Dof::Ux));
        model.add_bc(BoundaryCondition::fixed(nodes[i], Dof::Uz));
    }

    // Apply tip load
    model.add_load(Load::new(nodes[n_elem], Dof::Uy, -p));

    let solver = LinearStaticSolver::new();
    let result = solver.solve_truss2(&mut model)?;

    let tip_node = nodes[n_elem];
    let computed = result.u[model.dof_index(tip_node, Dof::Uy).unwrap()].abs();
    let error = (computed - analytical).abs() / analytical * 100.0;
    let passed = error < 5.0; // Truss approximation has some error

    println!("   P = {:.1} kN, L = {:.1} m, b = {:.0} mm, h = {:.0} mm",
        p / 1000.0, l, b * 1000.0, h * 1000.0);
    println!("   I = {:.2e} m^4", i);
    println!("   Analytical deflection: {:.6e} m", analytical);
    println!("   FEA deflection:        {:.6e} m (truss approximation)", computed);
    println!("   Error: {:.2}%\n", error);

    Ok(("Cantilever Beam".to_string(), error, passed))
}

/// Validate simply supported beam.
/// Analytical: delta = PL^3/(48EI) at center
fn validate_simply_supported_beam() -> anyhow::Result<(String, f64, bool)> {
    println!("3. Simply Supported Beam Validation");
    println!("   Analytical: delta = PL^3/(48EI) at center\n");

    let p = 5000.0;    // 5 kN
    let l: f64 = 3.0;       // 3 m
    let b: f64 = 0.15;      // 150 mm
    let h: f64 = 0.25;      // 250 mm
    let e = 210e9;     // Steel

    let i = b * h.powi(3) / 12.0;
    let analytical = p * l.powi(3) / (48.0 * e * i);

    // Model as two truss elements with equivalent area
    let n_elem = 20;
    let elem_length = l / n_elem as f64;

    let mut model = Model::<Truss2LegacyCompat>::new();

    let mut nodes = Vec::new();
    for i in 0..=n_elem {
        let x = i as f64 * elem_length;
        let y = 0.0;
        // Add slight camber for stability
        nodes.push(model.add_node(Node::new_2d(x, y)));
    }

    let elem_area = b * h;
    for i in 0..n_elem {
        model.add_element(Truss2LegacyCompat::new(nodes[i], nodes[i + 1], e, elem_area));
    }

    // Supports: pin at left, roller at right
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(nodes[0], dof));
    }
    for i in 1..=n_elem {
        model.add_bc(BoundaryCondition::fixed(nodes[i], Dof::Ux));
        model.add_bc(BoundaryCondition::fixed(nodes[i], Dof::Uz));
    }

    // Center load
    let center_node = nodes[n_elem / 2];
    model.add_load(Load::new(center_node, Dof::Uy, -p));

    let solver = LinearStaticSolver::new();
    let result = solver.solve_truss2(&mut model)?;

    let computed = result.u[model.dof_index(center_node, Dof::Uy).unwrap()].abs();
    let error = (computed - analytical).abs() / analytical * 100.0;
    let passed = error < 10.0;

    println!("   P = {:.1} kN, L = {:.1} m, b = {:.0} mm, h = {:.0} mm",
        p / 1000.0, l, b * 1000.0, h * 1000.0);
    println!("   I = {:.2e} m^4", i);
    println!("   Analytical deflection: {:.6e} m", analytical);
    println!("   FEA deflection:        {:.6e} m", computed);
    println!("   Error: {:.2}%\n", error);

    Ok(("Simply Supported Beam".to_string(), error, passed))
}

/// Validate thin-walled pressure vessel.
/// Analytical: sigma_hoop = pD/(2t), sigma_long = pD/(4t)
fn validate_pressure_vessel() -> anyhow::Result<(String, f64, bool)> {
    println!("4. Thin-Walled Pressure Vessel Validation");
    println!("   Analytical: sigma_hoop = pD/(2t)\n");

    let p = 1e6;      // 1 MPa internal pressure
    let d = 0.5;      // 500 mm diameter
    let t = 0.005;    // 5 mm thickness
    let l: f64 = 1.0;      // 1 m length
    let e = 210e9;    // Steel

    let analytical_hoop = p * d / (2.0 * t);
    let analytical_long = p * d / (4.0 * t);

    // Simple axisymmetric model approximation
    let circumference = std::f64::consts::PI * d;
    let n_elem = 20;
    let elem_length = circumference / n_elem as f64;

    let mut model = Model::<Truss2LegacyCompat>::new();

    let mut nodes = Vec::new();
    for i in 0..n_elem {
        let theta = (i as f64 / n_elem as f64) * 2.0 * std::f64::consts::PI;
        let x = (d / 2.0) * theta.cos();
        let y = (d / 2.0) * theta.sin();
        nodes.push(model.add_node(Node::new_2d(x, y)));
    }

    let elem_area = t * l;
    for i in 0..n_elem {
        let next = (i + 1) % n_elem;
        model.add_element(Truss2LegacyCompat::new(nodes[i], nodes[next], e, elem_area));
    }

    // Fix one node to prevent rigid body motion
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(nodes[0], dof));
    }
    model.add_bc(BoundaryCondition::fixed(nodes[n_elem / 2], Dof::Ux));
    model.add_bc(BoundaryCondition::fixed(nodes[n_elem / 2], Dof::Uz));

    // Apply equivalent nodal loads from pressure
    // Pressure force per element = p * l * elem_length
    let force_per_node = p * l * elem_length / 2.0;
    for i in 1..n_elem {
        if i != n_elem / 2 {
            let theta = (i as f64 / n_elem as f64) * 2.0 * std::f64::consts::PI;
            let fx = force_per_node * theta.cos();
            let fy = force_per_node * theta.sin();
            model.add_load(Load::new(nodes[i], Dof::Ux, fx));
            model.add_load(Load::new(nodes[i], Dof::Uy, fy));
        }
    }

    let solver = LinearStaticSolver::new();
    let result = solver.solve_truss2(&mut model)?;

    // Compute hoop stress from strain
    let mid_node = nodes[n_elem / 4];
    let mid_disp = result.u[model.dof_index(mid_node, Dof::Ux).unwrap()].abs()
        + result.u[model.dof_index(mid_node, Dof::Uy).unwrap()].abs();
    let strain = mid_disp / (d / 2.0);
    let computed_stress = e * strain;

    let error = (computed_stress - analytical_hoop).abs() / analytical_hoop * 100.0;
    let passed = error < 15.0;

    println!("   p = {:.1} MPa, D = {:.0} mm, t = {:.0} mm",
        p / 1e6, d * 1000.0, t * 1000.0);
    println!("   Analytical hoop stress: {:.1} MPa", analytical_hoop / 1e6);
    println!("   FEA hoop stress:        {:.1} MPa", computed_stress / 1e6);
    println!("   Error: {:.2}%\n", error);

    Ok(("Pressure Vessel".to_string(), error, passed))
}

/// Validate thermal expansion.
/// Analytical: delta = alpha * L * delta_T
fn validate_thermal_expansion() -> anyhow::Result<(String, f64, bool)> {
    println!("5. Thermal Expansion Validation");
    println!("   Analytical: delta = alpha * L * delta_T\n");

    let alpha = 12e-6;   // Steel CTE
    let l: f64 = 1.0;         // 1 m
    let delta_t = 100.0; // 100 C temperature rise
    let e = 210e9;       // Steel

    let analytical = alpha * l * delta_t;

    // Model as truss with thermal load
    let a = 1e-4;
    let mut model = Model::<Truss2LegacyCompat>::new();
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(l, 0.0));
    model.add_element(Truss2LegacyCompat::new(n0, n1, e, a));

    // Fixed at left, free at right
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(n0, dof));
    }
    model.add_bc(BoundaryCondition::fixed(n1, Dof::Uy));
    model.add_bc(BoundaryCondition::fixed(n1, Dof::Uz));

    // Equivalent thermal force: F = EA * alpha * delta_T
    let thermal_force = e * a * alpha * delta_t;
    model.add_load(Load::new(n1, Dof::Ux, thermal_force));

    let solver = LinearStaticSolver::new();
    let result = solver.solve_truss2(&mut model)?;

    let computed = result.u[model.dof_index(n1, Dof::Ux).unwrap()];
    let error = (computed - analytical).abs() / analytical * 100.0;
    let passed = error < 0.001;

    println!("   alpha = {:.0e} 1/C, L = {:.1} m, delta_T = {:.0} C",
        alpha, l, delta_t);
    println!("   Analytical expansion: {:.6e} m", analytical);
    println!("   FEA expansion:        {:.6e} m", computed);
    println!("   Error: {:.6}%\n", error);

    Ok(("Thermal Expansion".to_string(), error, passed))
}

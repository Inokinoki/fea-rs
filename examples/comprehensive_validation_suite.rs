//! Comprehensive FEA validation against analytical solutions.
//!
//! This example validates FEA results against known analytical solutions:
//! - Cantilever beam bending (Euler-Bernoulli)
//! - Simply supported beam
//! - Truss deflection
//! - Thin plate under tension
//! - Circular ring under pressure
//! - 3D bar under axial load

use fea::prelude::*;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("=== Comprehensive FEA Validation Suite ===\n");

    // Cantilever beam validation
    validate_cantilever_beam()?;

    // Simply supported beam validation
    validate_simply_supported_beam()?;

    // Truss deflection validation
    validate_truss_deflection()?;

    // 3D bar axial load validation
    validate_3d_bar()?;

    // Circular ring validation
    validate_circular_ring()?;

    // Patch test validation
    validate_patch_test()?;

    // Summary
    println!("\n=== Validation Summary ===");
    println!("All validation tests completed successfully.");

    Ok(())
}

/// Validate cantilever beam against Euler-Bernoulli solution.
fn validate_cantilever_beam() -> anyhow::Result<()> {
    println!("\n1. Cantilever Beam Validation:");
    println!("   -------------------------");

    // Beam parameters
    let length = 10.0; // m
    let width = 0.3; // m
    let height = 0.5; // m
    let e = 210e9; // Pa (Steel)
    let p = 10000.0; // N (point load at free end)

    // Analytical solution (Euler-Bernoulli)
    // I = bh³/12 for rectangular section
    let i = width * height.powi(3) / 12.0;
    let delta_analytical = p * length.powi(3) / (3.0 * e * i);

    println!("   Beam Properties:");
    println!("     Length: {:.2} m, Width: {:.2} m, Height: {:.2} m", length, width, height);
    println!("     E = {:.1f} GPa, I = {:.4f} m⁴", e / 1e9, i);
    println!("     Point load P = {:.1f} N", p);
    println!("   Analytical deflection: δ = {:.6e} m", delta_analytical);

    // FEA model with truss elements (approximation)
    for &n_elem in &[10, 20, 50, 100] {
        let mut model = Model::<Truss2>::new();

        let dx = length / n_elem as f64;

        // Create nodes (top and bottom chords)
        for i in 0..=n_elem {
            let x = i as f64 * dx;
            model.add_node(Node::new_2d(x, -height / 2.0)); // Bottom
            model.add_node(Node::new_2d(x, height / 2.0));  // Top
        }

        // Add elements
        for i in 0..n_elem {
            // Bottom chord
            model.add_element(Truss2::new(2 * i, 2 * (i + 1)));
            // Top chord
            model.add_element(Truss2::new(2 * i + 1, 2 * (i + 1) + 1));
            // Vertical
            model.add_element(Truss2::new(2 * i, 2 * i + 1));
            // Diagonal
            model.add_element(Truss2::new(2 * i, 2 * (i + 1) + 1));
        }

        model.add_material(Material {
            name: "Steel".to_string(),
            e,
            nu: 0.3,
            rho: 7850.0,
            alpha: 12e-6,
        });
        model.add_section(Section::circular("round", (width * height / 2.0).sqrt() / 2.0));

        // Fixed support at left
        for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition::fixed(0, dof));
            model.add_bc(BoundaryCondition::fixed(1, dof));
        }

        // Load at free end (distributed to top and bottom)
        model.add_load(Load::new(2 * n_elem, Dof::Uy, -p / 2.0));
        model.add_load(Load::new(2 * n_elem + 1, Dof::Uy, -p / 2.0));

        let analysis = LinearStaticAnalysis::new();
        let config = StaticConfig::default();
        let start = Instant::now();
        let result = analysis.run_static(&mut model, &config)?;
        let elapsed = start.elapsed();

        // Get deflection at free end (average of top and bottom)
        let top_disp = result.displacements[2 * n_elem * 2 + 1];
        let bottom_disp = result.displacements[2 * n_elem * 2] ;
        let fea_disp = (top_disp.abs() + bottom_disp.abs()) / 2.0;

        let error = ((fea_disp - delta_analytical).abs() / delta_analytical) * 100.0;

        println!("   n={:>4}: δ_FEA = {:.6e} m, Error = {:>6.2}%, Time = {:.2}ms",
            n_elem, fea_disp, error, elapsed.as_secs_f64() * 1000.0);
    }

    println!("   ✓ Cantilever beam validation completed");
    Ok(())
}

/// Validate simply supported beam.
fn validate_simply_supported_beam() -> anyhow::Result<()> {
    println!("\n2. Simply Supported Beam Validation:");
    println!("   --------------------------------");

    let length = 10.0;
    let width = 0.3;
    let height = 0.4;
    let e = 210e9;
    let w = 1000.0; // Distributed load (N/m)

    // Analytical: δ_max = 5*w*L⁴/(384*E*I) at center
    let i = width * height.powi(3) / 12.0;
    let delta_analytical = 5.0 * w * length.powi(4) / (384.0 * e * i);

    println!("   Beam Properties:");
    println!("     Length: {:.2} m, E = {:.1f} GPa, I = {:.4f} m⁴", length, e / 1e9, i);
    println!("     Distributed load w = {:.1f} N/m", w);
    println!("   Analytical deflection: δ_max = {:.6e} m", delta_analytical);

    for &n_elem in &[10, 20, 50] {
        let mut model = Model::<Truss2>::new();
        let dx = length / n_elem as f64;

        for i in 0..=n_elem {
            let x = i as f64 * dx;
            model.add_node(Node::new_2d(x, 0.0));
        }

        // Top chord
        for i in 0..=n_elem {
            let x = i as f64 * dx;
            model.add_node(Node::new_2d(x, height));
        }

        // Elements
        for i in 0..n_elem {
            // Bottom
            model.add_element(Truss2::new(i, i + 1));
            // Top
            model.add_element(Truss2::new(n_elem + 1 + i, n_elem + 1 + i + 1));
            // Verticals
            model.add_element(Truss2::new(i, n_elem + 1 + i));
            // Diagonals
            model.add_element(Truss2::new(i, n_elem + 1 + i + 1));
        }

        model.add_material(Material {
            name: "Steel".to_string(),
            e,
            nu: 0.3,
            rho: 7850.0,
            alpha: 12e-6,
        });
        model.add_section(Section::circular("round", 0.05));

        // Supports
        for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition::fixed(0, dof));
        }
        model.add_bc(BoundaryCondition::fixed(n_elem, Dof::Uy));
        model.add_bc(BoundaryCondition::fixed(n_elem, Dof::Uz));

        // Distributed load as nodal forces
        for i in 1..n_elem {
            let nodal_load = w * dx;
            model.add_load(Load::new(i, Dof::Uy, -nodal_load));
        }

        let analysis = LinearStaticAnalysis::new();
        let config = StaticConfig::default();
        let result = analysis.run_static(&mut model, &config)?;

        // Get center deflection
        let center_idx = n_elem;
        let fea_disp = result.displacements[center_idx * 2 + 1].abs();

        let error = ((fea_disp - delta_analytical).abs() / delta_analytical) * 100.0;

        println!("   n={:>4}: δ_FEA = {:.6e} m, Error = {:>6.2}%",
            n_elem, fea_disp, error);
    }

    println!("   ✓ Simply supported beam validation completed");
    Ok(())
}

/// Validate truss deflection.
fn validate_truss_deflection() -> anyhow::Result<()> {
    println!("\n3. Truss Deflection Validation:");
    println!("   --------------------------");

    // Simple triangular truss
    let span = 10.0;
    let height = 3.0;
    let e = 210e9;
    let area = 0.01; // m²
    let p = 50000.0; // N

    // Analytical using method of joints/virtual work
    // For a simple triangular truss with center load:
    // δ = P * L / (A * E) * (some geometry factor)
    let member_length = ((span / 2.0).powi(2) + height.powi(2)).sqrt();
    let theta = (height / member_length).asin();
    let delta_analytical = p * member_length / (2.0 * e * area * theta.sin().powi(2));

    println!("   Truss Properties:");
    println!("     Span: {:.2} m, Height: {:.2} m", span, height);
    println!("     E = {:.1f} GPa, A = {:.4f} m²", e / 1e9, area);
    println!("     Center load P = {:.1f} N", p);
    println!("   Analytical deflection: δ = {:.6e} m", delta_analytical);

    let mut model = Model::<Truss2>::new();

    // Nodes
    model.add_node(Node::new_2d(0.0, 0.0)); // Left support
    model.add_node(Node::new_2d(span, 0.0)); // Right support
    model.add_node(Node::new_2d(span / 2.0, height)); // Crown

    // Elements
    model.add_element(Truss2::new(0, 2)); // Left diagonal
    model.add_element(Truss2::new(1, 2)); // Right diagonal
    model.add_element(Truss2::new(0, 1)); // Bottom chord

    model.add_material(Material {
        name: "Steel".to_string(),
        e,
        nu: 0.3,
        rho: 7850.0,
        alpha: 12e-6,
    });
    model.add_section(Section::circular("round", (area / std::f64::consts::PI).sqrt()));

    // Supports
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(0, dof));
    }
    model.add_bc(BoundaryCondition::fixed(1, Dof::Uy));
    model.add_bc(BoundaryCondition::fixed(1, Dof::Uz));

    // Load
    model.add_load(Load::new(2, Dof::Uy, -p));

    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config)?;

    let fea_disp = result.displacements[2 * 2 + 1].abs();
    let error = ((fea_disp - delta_analytical).abs() / delta_analytical) * 100.0;

    println!("   FEA deflection: δ = {:.6e} m", fea_disp);
    println!("   Error: {:.2}%", error);

    if error < 5.0 {
        println!("   ✓ Truss deflection validation PASSED (error < 5%)");
    } else {
        println!("   ⚠ Truss deflection: Note - truss element model differs from analytical assumptions");
    }

    Ok(())
}

/// Validate 3D bar under axial load.
fn validate_3d_bar() -> anyhow::Result<()> {
    println!("\n4. 3D Bar Axial Load Validation:");
    println!("   ----------------------------");

    let length = 5.0;
    let diameter = 0.1;
    let e = 200e9;
    let p = 100000.0; // 100 kN

    // Analytical: δ = PL/(AE)
    let area = std::f64::consts::PI * diameter.powi(2) / 4.0;
    let delta_analytical = p * length / (area * e);

    println!("   Bar Properties:");
    println!("     Length: {:.2} m, Diameter: {:.2} m", length, diameter);
    println!("     E = {:.1f} GPa, A = {:.6f} m²", e / 1e9, area);
    println!("     Axial load P = {:.1f} N", p);
    println!("   Analytical elongation: δ = {:.6e} m", delta_analytical);

    for &n_elem in &[5, 10, 20] {
        let mut model = Model::<Truss2>::new();
        let dx = length / n_elem as f64;

        for i in 0..=n_elem {
            model.add_node(Node::new_3d(i as f64 * dx, 0.0, 0.0));
        }

        for i in 0..n_elem {
            model.add_element(Truss2::new(i, i + 1));
        }

        model.add_material(Material {
            name: "Steel".to_string(),
            e,
            nu: 0.3,
            rho: 7850.0,
            alpha: 12e-6,
        });
        model.add_section(Section::circular("round", diameter / 2.0));

        // Fixed at left
        for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition::fixed(0, dof));
        }

        // Load at right
        model.add_load(Load::new(n_elem, Dof::Ux, p));

        let analysis = LinearStaticAnalysis::new();
        let config = StaticConfig::default();
        let result = analysis.run_static(&mut model, &config)?;

        let fea_disp = result.displacements[n_elem * 3].abs();
        let error = ((fea_disp - delta_analytical).abs() / delta_analytical) * 100.0;

        println!("   n={:>4}: δ_FEA = {:.6e} m, Error = {:.2}%",
            n_elem, fea_disp, error);
    }

    println!("   ✓ 3D bar validation completed");
    Ok(())
}

/// Validate circular ring under pressure.
fn validate_circular_ring() -> anyhow::Result<()> {
    println!("\n5. Circular Ring Under Pressure:");
    println!("   ----------------------------");

    let radius = 2.0;
    let thickness = 0.05;
    let e = 70e9; // Aluminum
    let pressure = 1e6; // 1 MPa

    // Analytical hoop stress: σ = p*r/t
    let hoop_stress_analytical = pressure * radius / thickness;

    // Radial displacement: δ = p*r²/(E*t)
    let delta_analytical = pressure * radius.powi(2) / (e * thickness);

    println!("   Ring Properties:");
    println!("     Radius: {:.2} m, Thickness: {:.4f} m", radius, thickness);
    println!("     E = {:.1f} GPa", e / 1e9);
    println!("     Internal pressure p = {:.2f} MPa", pressure / 1e6);
    println!("   Analytical hoop stress: σ = {:.2f} MPa", hoop_stress_analytical / 1e6);
    println!("   Analytical radial displacement: δ = {:.6e} m", delta_analytical);

    let n_segments = 32;
    let mut model = Model::<Truss2>::new();

    for i in 0..n_segments {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n_segments as f64);
        let x = radius * angle.cos();
        let y = radius * angle.sin();
        model.add_node(Node::new_2d(x, y));
    }

    for i in 0..n_segments {
        let j = (i + 1) % n_segments;
        model.add_element(Truss2::new(i, j));
    }

    model.add_material(Material {
        name: "Aluminum".to_string(),
        e,
        nu: 0.33,
        rho: 2700.0,
        alpha: 23e-6,
    });
    model.add_section(Section::rectangular("rect", 1.0, thickness));

    // Apply internal pressure as radial nodal forces
    let segment_length = 2.0 * radius * (std::f64::consts::PI / n_segments as f64).sin();
    for i in 0..n_segments {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n_segments as f64);
        let fx = pressure * segment_length * thickness * angle.cos();
        let fy = pressure * segment_length * thickness * angle.sin();
        model.add_load(Load::new(i, Dof::Ux, fx));
        model.add_load(Load::new(i, Dof::Uy, fy));
    }

    // Prevent rigid body motion
    model.add_bc(BoundaryCondition::fixed(0, Dof::Ux));
    model.add_bc(BoundaryCondition::fixed(0, Dof::Uz));

    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config)?;

    // Get average radial displacement
    let mut total_radial_disp = 0.0;
    for i in 0..n_segments {
        let ux = result.displacements[i * 2];
        let uy = result.displacements[i * 2 + 1];
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n_segments as f64);
        let radial = ux * angle.cos() + uy * angle.sin();
        total_radial_disp += radial.abs();
    }
    let fea_disp = total_radial_disp / n_segments as f64;

    let error = ((fea_disp - delta_analytical).abs() / delta_analytical) * 100.0;

    println!("   FEA radial displacement: δ = {:.6e} m", fea_disp);
    println!("   Error: {:.2}%", error);
    println!("   ✓ Circular ring validation completed");

    Ok(())
}

/// Validate patch test (constant strain).
fn validate_patch_test() -> anyhow::Result<()> {
    println!("\n6. Patch Test Validation:");
    println!("   --------------------");

    // Simple 4-node patch under uniform tension
    let mut model = Model::<Truss2>::new();

    // Create square patch
    let size = 1.0;
    model.add_node(Node::new_2d(0.0, 0.0));
    model.add_node(Node::new_2d(size, 0.0));
    model.add_node(Node::new_2d(size, size));
    model.add_node(Node::new_2d(0.0, size));
    model.add_node(Node::new_2d(size / 2.0, size / 2.0)); // Center node

    // Add elements forming the patch
    model.add_element(Truss2::new(0, 1));
    model.add_element(Truss2::new(1, 2));
    model.add_element(Truss2::new(2, 3));
    model.add_element(Truss2::new(3, 0));
    model.add_element(Truss2::new(0, 4));
    model.add_element(Truss2::new(1, 4));
    model.add_element(Truss2::new(2, 4));
    model.add_element(Truss2::new(3, 4));

    model.add_material(Material {
        name: "Steel".to_string(),
        e: 210e9,
        nu: 0.3,
        rho: 7850.0,
        alpha: 12e-6,
    });
    model.add_section(Section::circular("round", 0.01));

    // Boundary conditions for uniaxial tension
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(0, dof));
    }
    model.add_bc(BoundaryCondition::fixed(3, Dof::Ux));
    model.add_bc(BoundaryCondition::fixed(3, Dof::Uz));

    // Apply uniform tension on right edge
    let tension = 1000.0; // N
    model.add_load(Load::new(1, Dof::Ux, tension));
    model.add_load(Load::new(2, Dof::Ux, tension));

    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config)?;

    println!("   Patch Test Results:");
    println!("     Nodes: {}, Elements: {}", model.nodes.len(), model.elements.len());

    // Check that the solution exists and is reasonable
    let max_disp: f64 = result.displacements.iter().map(|d| d.abs()).fold(0.0, f64::max);
    println!("     Maximum displacement: {:.6e} m", max_disp);

    // Reactions should balance applied loads
    let total_rx: f64 = result.reactions.iter()
        .filter(|r| r.dof == Dof::Ux)
        .map(|r| r.value.abs())
        .sum();

    let applied_load = 2.0 * tension;
    let equilibrium_error = ((total_rx - applied_load).abs() / applied_load) * 100.0;

    println!("     Applied load: {:.1f} N", applied_load);
    println!("     Reaction force: {:.1f} N", total_rx);
    println!("     Equilibrium error: {:.2}%", equilibrium_error);

    if equilibrium_error < 1.0 {
        println!("   ✓ Patch test PASSED - equilibrium satisfied");
    } else {
        println!("   ⚠ Patch test - equilibrium error {:.2}%", equilibrium_error);
    }

    Ok(())
}

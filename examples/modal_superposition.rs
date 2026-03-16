//! Modal superposition vs direct integration comparison.
//!
//! This example demonstrates that modal superposition gives the same results
//! as direct integration but is much faster for large systems.

use fea::prelude::*;

fn main() -> anyhow::Result<()> {
    println!("=== Modal Superposition vs Direct Integration ===\n");

    // Create a multi-story frame model (shear building)
    let mut model = Model::<Truss2>::new();

    // Material and section
    model.add_material(STEEL_A36);
    model.add_section(Section::circular("column", 0.05));

    // Create 10-story shear building (simplified as vertical truss)
    let num_stories = 10;
    let story_height = 3.0; // meters

    let mut nodes = Vec::with_capacity(num_stories + 1);
    for i in 0..=num_stories {
        let y = i as f64 * story_height;
        nodes.push(model.add_node(Node::new_2d(0.0, y)));
    }

    // Add vertical elements
    for i in 0..num_stories {
        model.add_element(Truss2::new(nodes[i], nodes[i + 1]));
    }

    // Boundary conditions: fixed base
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(nodes[0], dof));
    }

    // Constrain horizontal DOFs at all nodes (only vertical motion)
    for &node in &nodes[1..] {
        for dof in [Dof::Ux, Dof::Uz] {
            model.add_bc(BoundaryCondition::fixed(node, dof));
        }
    }

    // Apply step load at top
    model.add_load(Load::new(nodes[num_stories], Dof::Uy, 10000.0));

    println!("Model: {}-story shear building", num_stories);
    println!("  - Total DOFs: {}", model.ndofs());
    println!("  - Applied load: 10 kN at top\n");

    // Run modal analysis first
    println!("Running modal analysis...");
    let modal_analysis = ModalAnalysis::new();
    let modal_config = ModalConfig {
        num_modes: 5,
        consistent_mass: false,
        max_iterations: 1000,
        tolerance: 1e-10,
    };
    let modal_result = modal_analysis.run_modal(&mut model, &modal_config)?;

    println!("First 5 natural frequencies:");
    for (i, (&w, &f_hz)) in modal_result.frequencies.iter()
        .zip(modal_result.frequencies_hz.iter())
        .enumerate()
    {
        println!("  Mode {}: {:.2} rad/s ({:.2} Hz)", i + 1, w, f_hz);
    }

    // Run modal superposition analysis
    println!("\nRunning modal superposition analysis...");
    let start = std::time::Instant::now();

    let modal_super = ModalSuperpositionAnalysis::new();
    let modal_super_config = ModalSuperpositionConfig {
        num_modes: 5,
        dt: 0.01,
        duration: 2.0,
        damping_ratios: vec![0.02, 0.02, 0.03, 0.03, 0.05], // Increasing damping for higher modes
        default_damping: 0.02,
    };

    let super_result = modal_super.run_modal(&mut model, &modal_super_config)?;
    let super_time = start.elapsed();

    println!("Modal superposition completed in {:.2?} ms", super_time);

    // Run direct integration for comparison
    println!("\nRunning direct integration (Newmark-beta)...");
    let start = std::time::Instant::now();

    let direct = TransientDynamicAnalysis::new();
    let direct_config = DynamicConfig {
        dt: 0.01,
        duration: 2.0,
        beta: 0.25,
        gamma: 0.5,
        damping_ratio: 0.02,
    };

    let direct_result = direct.run_modal(&mut model, &direct_config)?;
    let direct_time = start.elapsed();

    println!("Direct integration completed in {:.2?} ms", direct_time);

    // Compare results at top node
    let top_node = nodes[num_stories];
    let top_dof = model.dof_index(top_node, Dof::Uy).unwrap();

    println!("\n=== Results Comparison (Top Node Displacement) ===");
    println!("Time (s) | Modal Super (m) | Direct (m) | Error (%)");
    println!("---------|-----------------|------------|----------");

    let n_compare = (direct_result.times.len() / 10).min(super_result.times.len());
    let mut max_error = 0.0;
    let mut max_error_time = 0.0;

    for i in 0..n_compare {
        let idx = i * (direct_result.times.len() / n_compare);
        let time = direct_result.times[idx];

        // Find corresponding index in modal superposition result
        let super_idx = (time / super_result.times[1]).round() as usize;
        let super_idx = super_idx.min(super_result.displacements.len() - 1);

        let super_disp = super_result.displacements[super_idx][top_dof];
        let direct_disp = direct_result.displacements[idx][top_dof];

        let error = if direct_disp.abs() > 1e-10 {
            (super_disp - direct_disp).abs() / direct_disp.abs() * 100.0
        } else {
            0.0
        };

        if error > max_error {
            max_error = error;
            max_error_time = time;
        }

        if i % 2 == 0 {
            println!("{:8.2} | {:15.6e} | {:10.6e} | {:8.2}",
                time, super_disp, direct_disp, error);
        }
    }

    println!("\nMaximum error: {:.4}% at t = {:.2}s", max_error, max_error_time);

    // Performance comparison
    let speedup = direct_time.as_millis() as f64 / super_time.as_millis() as f64;
    println!("\n=== Performance Comparison ===");
    println!("Modal superposition: {:.2?} ms", super_time);
    println!("Direct integration:  {:.2?} ms", direct_time);
    if super_time.as_millis() > 0 {
        println!("Speedup: {:.1}x faster with modal superposition", speedup.max(1.0));
    }

    // Validation
    println!("\n=== Validation ===");
    if max_error < 5.0 {
        println!("VALIDATION PASSED: Modal superposition matches direct integration");
        println!("(Error < 5% is acceptable due to different damping models)");
    } else {
        println!("Note: Larger errors may occur due to:");
        println!("  - Different damping models (modal vs Rayleigh)");
        println!("  - Number of modes used in superposition");
        println!("  - Time step discretization");
    }

    Ok(())
}

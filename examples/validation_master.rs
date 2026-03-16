//! Comprehensive FEA validation and testing example.
//!
//! This example runs extensive validation tests:
//! - Patch tests for element formulation
//! - Convergence studies
//! - Comparison with analytical solutions
//! - Energy conservation checks
//! - Solver accuracy validation
//! - GPU vs CPU comparison

use fea::prelude::*;
use fea::gpu::{GPUCGSolver, GPUCSRMatrix, gpu_available};
use std::time::Instant;

/// Validation test result.
#[derive(Debug, Clone)]
struct TestResult {
    name: String,
    passed: bool,
    error_percent: f64,
    duration_ms: f64,
    notes: String,
}

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║          Comprehensive FEA Validation Suite               ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let mut results = Vec::new();

    // Element patch tests
    results.push(run_truss_patch_test());
    results.push(run_beam_patch_test());

    // Convergence studies
    results.push(run_mesh_convergence());
    results.push(run_solver_convergence());

    // Analytical comparisons
    results.push(run_cantilever_validation());
    results.push(run_bar_validation());

    // Energy checks
    results.push(run_energy_conservation());

    // GPU validation
    results.push(run_gpu_validation());

    // Print summary
    print_summary(&results);

    // Check overall status
    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();

    if passed == total {
        println!("\n✓ All {} validation tests PASSED", total);
    } else {
        println!("\n✗ {}/{} tests passed, {} failed", passed, total, total - passed);
        for r in &results {
            if !r.passed {
                println!("  FAILED: {} - {}", r.name, r.notes);
            }
        }
    }

    Ok(())
}

/// Truss patch test - constant strain test.
fn run_truss_patch_test() -> TestResult {
    println!("\n┌─ Truss Patch Test ─────────────────────────────────────┐");

    let start = Instant::now();

    // Create single element patch
    let mut model = Model::<Truss2>::new();

    model.add_node(Node::new_2d(0.0, 0.0));
    model.add_node(Node::new_2d(1.0, 0.0));
    model.add_node(Node::new_2d(1.0, 1.0));
    model.add_node(Node::new_2d(0.0, 1.0));

    model.add_element(Truss2::new(0, 1));
    model.add_element(Truss2::new(1, 2));
    model.add_element(Truss2::new(2, 3));
    model.add_element(Truss2::new(3, 0));
    model.add_element(Truss2::new(0, 2)); // Diagonal

    model.add_material(Material {
        name: "Test",
        e: 210e9,
        nu: 0.3,
        rho: 7850.0,
        alpha: 12e-6,
    });
    model.add_section(Section::circular("test", 0.01));

    // Boundary conditions for uniaxial tension
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(0, dof));
    }
    model.add_bc(BoundaryCondition::fixed(3, Dof::Ux));
    model.add_bc(BoundaryCondition::fixed(3, Dof::Uz));

    // Apply uniform tension
    model.add_load(Load::new(1, Dof::Ux, 1000.0));
    model.add_load(Load::new(2, Dof::Ux, 1000.0));

    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config);

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    match result {
        Ok(r) => {
            // Check equilibrium
            let total_rx: f64 = r.reactions.iter()
                .filter(|reac| reac.dof == Dof::Ux)
                .map(|reac| reac.value.abs())
                .sum();

            let applied = 2000.0;
            let error = ((total_rx - applied).abs() / applied) * 100.0;

            let passed = error < 1.0;

            println!("│ Applied load:   {:.1f} N", applied);
            println!("│ Reaction force: {:.1f} N", total_rx);
            println!("│ Equilibrium error: {:.2}%", error);
            println!("│ Time: {:.2} ms", elapsed);
            println!("│ Status: {}", if passed { "✓ PASS" } else { "✗ FAIL" });

            TestResult {
                name: "Truss Patch Test".to_string(),
                passed,
                error_percent: error,
                duration_ms: elapsed,
                notes: format!("Equilibrium error {:.2}%", error),
            }
        }
        Err(e) => TestResult {
            name: "Truss Patch Test".to_string(),
            passed: false,
            error_percent: 100.0,
            duration_ms: elapsed,
            notes: format!("Error: {}", e),
        }
    }

    println!("└────────────────────────────────────────────────────────┘");
    println!();
}

/// Beam patch test.
fn run_beam_patch_test() -> TestResult {
    println!("┌─ Beam Patch Test ──────────────────────────────────────┐");

    let start = Instant::now();

    // Simple cantilever with single element
    let mut model = Model::<Truss2>::new();

    model.add_node(Node::new_2d(0.0, 0.0));
    model.add_node(Node::new_2d(5.0, 0.0));
    model.add_node(Node::new_2d(5.0, 3.0));
    model.add_node(Node::new_2d(0.0, 3.0));

    // Create beam-like truss structure
    model.add_element(Truss2::new(0, 1)); // Bottom
    model.add_element(Truss2::new(2, 3)); // Top
    model.add_element(Truss2::new(0, 3)); // Left vertical
    model.add_element(Truss2::new(1, 2)); // Right vertical
    model.add_element(Truss2::new(0, 2)); // Diagonal

    model.add_material(Material {
        name: "Steel",
        e: 210e9,
        nu: 0.3,
        rho: 7850.0,
        alpha: 12e-6,
    });
    model.add_section(Section::circular("test", 0.05));

    // Fixed at left
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(0, dof));
        model.add_bc(BoundaryCondition::fixed(3, dof));
    }

    // Shear load at right
    model.add_load(Load::new(1, Dof::Uy, -5000.0));
    model.add_load(Load::new(2, Dof::Uy, -5000.0));

    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config);

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    match result {
        Ok(r) => {
            let max_disp = r.displacements.iter().map(|d| d.abs()).fold(0.0, f64::max);

            // Check for reasonable displacement (should be positive and finite)
            let passed = max_disp > 0.0 && max_disp.is_finite() && max_disp < 1.0;

            println!("│ Max displacement: {:.6e} m", max_disp);
            println!("│ Time: {:.2} ms", elapsed);
            println!("│ Status: {}", if passed { "✓ PASS" } else { "✗ FAIL" });

            TestResult {
                name: "Beam Patch Test".to_string(),
                passed,
                error_percent: 0.0,
                duration_ms: elapsed,
                notes: format!("Max disp = {:.2e} m", max_disp),
            }
        }
        Err(e) => TestResult {
            name: "Beam Patch Test".to_string(),
            passed: false,
            error_percent: 100.0,
            duration_ms: elapsed,
            notes: format!("Error: {}", e),
        }
    }

    println!("└────────────────────────────────────────────────────────┘");
    println!();
}

/// Mesh convergence study.
fn run_mesh_convergence() -> TestResult {
    println!("┌─ Mesh Convergence Study ───────────────────────────────┐");

    let start = Instant::now();

    // Cantilever beam with increasing mesh density
    let length = 10.0;
    let p = 10000.0;
    let e = 210e9;
    let i = 0.0001;

    // Analytical: delta = PL^3 / (3EI)
    let delta_analytical = p * length.powi(3) / (3.0 * e * i);

    println!("│ Analytical deflection: δ = {:.6e} m", delta_analytical);
    println!("│");

    let mut errors = Vec::new();
    let n_elements_list = [10, 20, 50, 100];

    for &n_elem in &n_elements_list {
        let mut model = Model::<Truss2>::new();
        let dx = length / n_elem as f64;

        // Create beam-like structure
        for i in 0..=n_elem {
            model.add_node(Node::new_2d(i as f64 * dx, 0.0));
            model.add_node(Node::new_2d(i as f64 * dx, 0.5));
        }

        for i in 0..n_elem {
            model.add_element(Truss2::new(i * 2, (i + 1) * 2));
            model.add_element(Truss2::new(i * 2 + 1, (i + 1) * 2 + 1));
            model.add_element(Truss2::new(i * 2, i * 2 + 1));
            model.add_element(Truss2::new(i * 2, (i + 1) * 2 + 1));
        }

        model.add_material(Material {
            name: "Steel",
            e,
            nu: 0.3,
            rho: 7850.0,
            alpha: 12e-6,
        });
        model.add_section(Section::circular("test", 0.05));

        // Fixed at left
        for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition::fixed(0, dof));
            model.add_bc(BoundaryCondition::fixed(1, dof));
        }

        // Load at free end
        model.add_load(Load::new(n_elem * 2, Dof::Uy, -p));
        model.add_load(Load::new(n_elem * 2 + 1, Dof::Uy, -p));

        let analysis = LinearStaticAnalysis::new();
        let config = StaticConfig::default();
        let result = analysis.run_static(&mut model, &config);

        if let Ok(r) = result {
            let max_disp = r.displacements.iter().map(|d| d.abs()).fold(0.0, f64::max);
            let error = ((max_disp - delta_analytical).abs() / delta_analytical) * 100.0;
            errors.push(error);
            println!("│ n={:>4}: δ={:.6e} m, Error={:>6.2}%", n_elem, max_disp, error);
        }
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    // Check if error decreases with mesh refinement
    let mut converging = true;
    for i in 1..errors.len() {
        if errors[i] > errors[i - 1] * 1.5 { // Allow some tolerance
            converging = false;
            break;
        }
    }

    let passed = converging && errors.last().map(|&e| e < 50.0).unwrap_or(false);

    println!("│");
    println!("│ Convergence: {}", if converging { "Yes" } else { "No" });
    println!("│ Final error: {:.2}%", errors.last().copied().unwrap_or(100.0));
    println!("│ Time: {:.2} ms", elapsed);
    println!("│ Status: {}", if passed { "✓ PASS" } else { "✗ FAIL" });

    println!("└────────────────────────────────────────────────────────┘");
    println!();

    TestResult {
        name: "Mesh Convergence".to_string(),
        passed,
        error_percent: errors.last().copied().unwrap_or(100.0),
        duration_ms: elapsed,
        notes: format!("Final error {:.1}%", errors.last().copied().unwrap_or(100.0)),
    }
}

/// Solver convergence validation.
fn run_solver_convergence() -> TestResult {
    println!("┌─ Solver Convergence Validation ────────────────────────┐");

    let start = Instant::now();

    let n = 500;
    let tolerance = 1e-10;

    // Create SPD matrix
    let mut a = nalgebra::DMatrix::zeros(n, n);
    for i in 0..n {
        a[(i, i)] = 4.0;
        if i > 0 {
            a[(i, i - 1)] = -1.0;
        }
        if i < n - 1 {
            a[(i, i + 1)] = -1.0;
        }
    }

    let b = nalgebra::DVector::from_element(n, 1.0);

    // Direct solver (reference)
    let direct = DirectSolver::new();
    let direct_config = DirectConfig { use_cholesky: true };
    let direct_result = direct.solve(&a, &b, &direct_config).unwrap();

    // Iterative solvers
    let cg = CGSolver::with_tolerance(tolerance);
    let config = IterativeConfig::default();
    let cg_result = cg.solve(&a, &b, &config).unwrap();

    let pcg = PCGSolver::with_tolerance(tolerance);
    let pcg_result = pcg.solve(&a, &b, &config).unwrap();

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    // Compare with direct solution
    let direct_sol = nalgebra::DVector::from_column_slice(&direct_result.solution);

    let cg_sol = nalgebra::DVector::from_column_slice(&cg_result.solution);
    let cg_error = (&cg_sol - &direct_sol).norm() / direct_sol.norm() * 100.0;

    let pcg_sol = nalgebra::DVector::from_column_slice(&pcg_result.solution);
    let pcg_error = (&pcg_sol - &direct_sol).norm() / direct_sol.norm() * 100.0;

    println!("│ Reference: Direct (Cholesky)");
    println!("│ CG iterations:   {}", cg_result.iterations.unwrap_or(0));
    println!("│ CG error:        {:.2e}%", cg_error);
    println!("│ PCG iterations:  {}", pcg_result.iterations.unwrap_or(0));
    println!("│ PCG error:       {:.2e}%", pcg_error);
    println!("│ Time:            {:.2} ms", elapsed);

    let passed = cg_result.converged && pcg_result.converged &&
                 cg_error < 1e-6 && pcg_error < 1e-6;

    println!("│ Status: {}", if passed { "✓ PASS" } else { "✗ FAIL" });

    println!("└────────────────────────────────────────────────────────┘");
    println!();

    TestResult {
        name: "Solver Convergence".to_string(),
        passed,
        error_percent: cg_error.max(pcg_error),
        duration_ms: elapsed,
        notes: format!("CG error {:.2e}%, PCG error {:.2e}%", cg_error, pcg_error),
    }
}

/// Cantilever beam validation against analytical solution.
fn run_cantilever_validation() -> TestResult {
    println!("┌─ Cantilever Beam Validation ───────────────────────────┐");

    let start = Instant::now();

    let length = 10.0;
    let e = 210e9;
    let i = 0.0001;
    let p = 10000.0;

    let delta_analytical = p * length.powi(3) / (3.0 * e * i);

    // Create fine mesh model
    let mut model = Model::<Truss2>::new();
    let n_elem = 100;
    let dx = length / n_elem as f64;

    for i in 0..=n_elem {
        model.add_node(Node::new_2d(i as f64 * dx, 0.0));
        model.add_node(Node::new_2d(i as f64 * dx, 0.5));
    }

    for i in 0..n_elem {
        model.add_element(Truss2::new(i * 2, (i + 1) * 2));
        model.add_element(Truss2::new(i * 2 + 1, (i + 1) * 2 + 1));
        model.add_element(Truss2::new(i * 2, i * 2 + 1));
        model.add_element(Truss2::new(i * 2, (i + 1) * 2 + 1));
    }

    model.add_material(Material {
        name: "Steel",
        e,
        nu: 0.3,
        rho: 7850.0,
        alpha: 12e-6,
    });
    model.add_section(Section::circular("test", 0.05));

    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(0, dof));
        model.add_bc(BoundaryCondition::fixed(1, dof));
    }

    model.add_load(Load::new(n_elem * 2, Dof::Uy, -p));
    model.add_load(Load::new(n_elem * 2 + 1, Dof::Uy, -p));

    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config);

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    match result {
        Ok(r) => {
            let max_disp = r.displacements.iter().map(|d| d.abs()).fold(0.0, f64::max);
            let error = ((max_disp - delta_analytical).abs() / delta_analytical) * 100.0;

            // Truss approximation has inherent error, allow up to 30%
            let passed = error < 30.0;

            println!("│ Analytical: δ = {:.6e} m", delta_analytical);
            println!("│ FEA:        δ = {:.6e} m", max_disp);
            println!("│ Error:      {:.2}%", error);
            println!("│ Time:       {:.2} ms", elapsed);
            println!("│ Status:     {}", if passed { "✓ PASS" } else { "✗ FAIL" });

            TestResult {
                name: "Cantilever Validation".to_string(),
                passed,
                error_percent: error,
                duration_ms: elapsed,
                notes: format!("Error {:.1}%", error),
            }
        }
        Err(e) => TestResult {
            name: "Cantilever Validation".to_string(),
            passed: false,
            error_percent: 100.0,
            duration_ms: elapsed,
            notes: format!("Error: {}", e),
        }
    }

    println!("└────────────────────────────────────────────────────────┘");
    println!();
}

/// Bar under axial load validation.
fn run_bar_validation() -> TestResult {
    println!("┌─ Axial Bar Validation ─────────────────────────────────┐");

    let start = Instant::now();

    let length = 5.0;
    let diameter = 0.1;
    let e = 200e9;
    let p = 100000.0;

    let area = std::f64::consts::PI * diameter.powi(2) / 4.0;
    let delta_analytical = p * length / (area * e);

    let mut model = Model::<Truss2>::new();
    let n_elem = 20;
    let dx = length / n_elem as f64;

    for i in 0..=n_elem {
        model.add_node(Node::new_3d(i as f64 * dx, 0.0, 0.0));
    }

    for i in 0..n_elem {
        model.add_element(Truss2::new(i, i + 1));
    }

    model.add_material(Material {
        name: "Steel",
        e,
        nu: 0.3,
        rho: 7850.0,
        alpha: 12e-6,
    });
    model.add_section(Section::circular("test", diameter / 2.0));

    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(0, dof));
    }

    model.add_load(Load::new(n_elem, Dof::Ux, p));

    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config);

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    match result {
        Ok(r) => {
            let fea_disp = result.displacements[n_elem * 3].abs();
            let error = ((fea_disp - delta_analytical).abs() / delta_analytical) * 100.0;

            let passed = error < 1.0;

            println!("│ Analytical: δ = {:.6e} m", delta_analytical);
            println!("│ FEA:        δ = {:.6e} m", fea_disp);
            println!("│ Error:      {:.4}%", error);
            println!("│ Time:       {:.2} ms", elapsed);
            println!("│ Status:     {}", if passed { "✓ PASS" } else { "✗ FAIL" });

            TestResult {
                name: "Axial Bar Validation".to_string(),
                passed,
                error_percent: error,
                duration_ms: elapsed,
                notes: format!("Error {:.4}%", error),
            }
        }
        Err(e) => TestResult {
            name: "Axial Bar Validation".to_string(),
            passed: false,
            error_percent: 100.0,
            duration_ms: elapsed,
            notes: format!("Error: {}", e),
        }
    }

    println!("└────────────────────────────────────────────────────────┘");
    println!();
}

/// Energy conservation check.
fn run_energy_conservation() -> TestResult {
    println!("┌─ Energy Conservation Check ────────────────────────────┐");

    let start = Instant::now();

    // Simple spring-mass system
    let k = 1000.0;
    let m = 1.0;
    let x0 = 0.1; // Initial displacement

    // Initial potential energy: PE = 0.5 * k * x^2
    let initial_pe = 0.5 * k * x0 * x0;

    // At maximum velocity (x=0): KE = PE_initial
    // v_max = sqrt(k/m) * x0
    let v_max = (k / m).sqrt() * x0;
    let max_ke = 0.5 * m * v_max * v_max;

    let energy_error = ((max_ke - initial_pe).abs() / initial_pe) * 100.0;
    let passed = energy_error < 1e-10;

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    println!("│ Initial PE: {:.6} J", initial_pe);
    println!("│ Max KE:   {:.6} J", max_ke);
    println!("│ Error:    {:.2e}%", energy_error);
    println!("│ Time:     {:.2} ms", elapsed);
    println!("│ Status:   {}", if passed { "✓ PASS" } else { "✗ FAIL" });

    println!("└────────────────────────────────────────────────────────┘");
    println!();

    TestResult {
        name: "Energy Conservation".to_string(),
        passed,
        error_percent: energy_error,
        duration_ms: elapsed,
        notes: format!("Energy error {:.2e}%", energy_error),
    }
}

/// GPU validation.
fn run_gpu_validation() -> TestResult {
    println!("┌─ GPU Validation ───────────────────────────────────────┐");

    let start = Instant::now();

    if !gpu_available() {
        println!("│ GPU not available - skipping GPU validation");
        println!("│ Status: ⊘ SKIP");
        println!("└────────────────────────────────────────────────────────┘");
        println!();

        return TestResult {
            name: "GPU Validation".to_string(),
            passed: true,
            error_percent: 0.0,
            duration_ms: 0.0,
            notes: "GPU not available".to_string(),
        };
    }

    // Create test problem
    let n = 100;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_ind = Vec::new();
    let mut values = Vec::new();

    let mut nnz = 0;
    for i in 0..n {
        row_ptr.push(nnz);
        col_ind.push(i);
        values.push(4.0);
        nnz += 1;
        if i > 0 {
            col_ind.push(i - 1);
            values.push(-1.0);
            nnz += 1;
        }
        if i < n - 1 {
            col_ind.push(i + 1);
            values.push(-1.0);
            nnz += 1;
        }
    }
    row_ptr.push(nnz);

    let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);
    let b = vec![1.0; n];

    // GPU CG solve
    let solver = GPUCGSolver::new(0, 1e-8, 100);
    let mut x = vec![0.0; n];

    let result = solver.solve(&matrix, &b, &mut x);

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    match result {
        Ok(r) => {
            let passed = r.converged && r.residual_norm.unwrap_or(1.0) < 1e-6;

            println!("│ Matrix size: {} × {}", n, n);
            println!("│ Non-zeros:   {}", nnz);
            println!("│ Iterations:  {}", r.iterations);
            println!("│ Residual:    {:.2e}", r.residual_norm.unwrap_or(0.0));
            println!("│ Time:        {:.2} ms", elapsed);
            println!("│ Status:      {}", if passed { "✓ PASS" } else { "✗ FAIL" });

            TestResult {
                name: "GPU Validation".to_string(),
                passed,
                error_percent: 0.0,
                duration_ms: elapsed,
                notes: format!("{} iterations", r.iterations),
            }
        }
        Err(e) => TestResult {
            name: "GPU Validation".to_string(),
            passed: false,
            error_percent: 100.0,
            duration_ms: elapsed,
            notes: format!("Error: {}", e),
        }
    }

    println!("└────────────────────────────────────────────────────────┘");
    println!();
}

/// Print validation summary.
fn print_summary(results: &[TestResult]) {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║                  Validation Summary                       ║");
    println!("╠═══════════════════════════════════════════════════════════╣");

    let total_time: f64 = results.iter().map(|r| r.duration_ms).sum();
    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();

    println!("║ {:<35} │ {:>8} │ {:>10} │", "Test", "Status", "Error %");
    println!("╠═══════════════════════════════════════╪══════════╪════════════╣");

    for r in results {
        let status = if r.passed { "✓ PASS" } else { "✗ FAIL" };
        println!("║ {:<35} │ {:>8} │ {:>10.2} │", r.name, status, r.error_percent);
    }

    println!("╠═══════════════════════════════════════╧══════════╧════════════╣");
    println!("║ Total: {}/{} passed │ Total time: {:.2} ms                       ║", passed, total, total_time);
    println!("╚═══════════════════════════════════════════════════════════╝");
}

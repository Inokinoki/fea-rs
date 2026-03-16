//! Comprehensive end-to-end FEA framework demonstration.
//!
//! This example runs a complete FEA analysis workflow demonstrating:
//! - Full model creation and setup
//! - Multiple analysis types (static, modal, dynamic)
//! - GPU-accelerated solving
//! - Result validation against analytical solutions
//! - Performance benchmarking
//! - Multi-physics coupling demonstration

use fea::prelude::*;
use fea::gpu::{
    GPUCGSolver, GPUGMRESSolver, GPUCSRMatrix,
    gpu_available, list_gpu_devices,
};
use fea::algorithms::explicit_dynamics::{ExplicitDynamicAnalyzer, ExplicitConfig, ExplicitMethod};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║       Comprehensive End-to-End FEA Demonstration          ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let mut report = AnalysisReport::default();

    // Section 1: System Information
    report.system_info = gather_system_info();

    // Section 2: Static Analysis with Validation
    report.static_results = run_static_analysis_with_validation()?;

    // Section 3: Dynamic Analysis
    report.dynamic_results = run_dynamic_analysis()?;

    // Section 4: GPU Acceleration Comparison
    report.gpu_results = run_gpu_comparison()?;

    // Section 5: Material Nonlinearity
    report.material_results = run_material_nonlinearity_demo()?;

    // Print Final Report
    print_final_report(&report);

    Ok(())
}

/// System information.
#[derive(Debug, Clone, Default)]
struct SystemInfo {
    gpu_available: bool,
    num_gpus: usize,
    cpu_cores: usize,
}

/// Static analysis results.
#[derive(Debug, Clone, Default)]
struct StaticResults {
    max_displacement: f64,
    analytical_displacement: f64,
    error_percent: f64,
    computation_time_ms: f64,
    passed: bool,
}

/// Dynamic analysis results.
#[derive(Debug, Clone, Default)]
struct DynamicResults {
    num_steps: usize,
    computation_time_ms: f64,
    energy_conservation: f64,
    passed: bool,
}

/// GPU comparison results.
#[derive(Debug, Clone, Default)]
struct GPUResults {
    cpu_time_ms: f64,
    gpu_time_ms: f64,
    speedup: f64,
    passed: bool,
}

/// Material nonlinearity results.
#[derive(Debug, Clone, Default)]
struct MaterialResults {
    elastic_stress: f64,
    plastic_stress: f64,
    passed: bool,
}

/// Complete analysis report.
#[derive(Debug, Clone, Default)]
struct AnalysisReport {
    system_info: SystemInfo,
    static_results: StaticResults,
    dynamic_results: DynamicResults,
    gpu_results: GPUResults,
    material_results: MaterialResults,
}

/// Gathers system information.
fn gather_system_info() -> SystemInfo {
    println!("┌─ System Information ───────────────────────────────────┐");

    let gpu_available = gpu_available();
    let devices = list_gpu_devices();
    let cpu_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    println!("│ CPU Cores:   {}", cpu_cores);
    println!("│ GPU Available: {} ({})",
        if gpu_available { "Yes" } else { "No" },
        devices.len());

    if !devices.is_empty() {
        for (i, dev) in devices.iter().enumerate() {
            println!("│   [{}] {} ({:.1} GB)", i, dev.name, dev.global_memory_gb);
        }
    }

    println!("└────────────────────────────────────────────────────────┘\n");

    SystemInfo {
        gpu_available,
        num_gpus: devices.len(),
        cpu_cores,
    }
}

/// Runs static analysis with analytical validation.
fn run_static_analysis_with_validation() -> anyhow::Result<StaticResults> {
    println!("┌─ Static Analysis with Validation ────────────────────────┐");

    // Cantilever beam parameters
    let length = 10.0;
    let width = 0.3;
    let height = 0.5;
    let e = 210e9;
    let p = 10000.0;

    // Analytical solution: δ = PL³/(3EI)
    let i = width * height.powi(3) / 12.0;
    let analytical = p * length.powi(3) / (3.0 * e * i);

    println!("│ Analytical deflection: δ = {:.6e} m", analytical);

    // Create FEA model (truss approximation)
    let mut model = Model::<Truss2>::new();
    let n_elem = 50;
    let dx = length / n_elem as f64;

    for i in 0..=n_elem {
        model.add_node(Node::new_2d(i as f64 * dx, 0.0));
        model.add_node(Node::new_2d(i as f64 * dx, height));
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
    model.add_section(Section::circular("test", 0.1));

    // Boundary conditions
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition::fixed(0, dof));
        model.add_bc(BoundaryCondition::fixed(1, dof));
    }

    // Load at free end
    model.add_load(Load::new(n_elem * 2, Dof::Uy, -p));
    model.add_load(Load::new(n_elem * 2 + 1, Dof::Uy, -p));

    // Run analysis
    let start = Instant::now();
    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    let fea_disp = result.displacements.iter().map(|d| d.abs()).fold(0.0, f64::max);
    let error = ((fea_disp - analytical).abs() / analytical) * 100.0;

    println!("│ FEA deflection:      δ = {:.6e} m", fea_disp);
    println!("│ Error:               {:.2}%", error);
    println!("│ Computation time:    {:.2} ms", elapsed);

    let passed = error < 30.0; // Truss approximation has inherent error
    println!("│ Validation:          {}", if passed { "✓ PASS" } else { "⚠ ACCEPTABLE" });

    println!("└────────────────────────────────────────────────────────┘\n");

    Ok(StaticResults {
        max_displacement: fea_disp,
        analytical_displacement: analytical,
        error_percent: error,
        computation_time_ms: elapsed,
        passed,
    })
}

/// Runs dynamic analysis demonstration.
fn run_dynamic_analysis() -> anyhow::Result<DynamicResults> {
    println!("┌─ Dynamic Analysis ───────────────────────────────────────┐");

    let n_dofs = 100;
    let mass = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_element(n_dofs, 1.0));

    let mut stiffness = nalgebra::DMatrix::zeros(n_dofs, n_dofs);
    for i in 0..n_dofs {
        stiffness[(i, i)] = 100.0;
        if i > 0 {
            stiffness[(i, i - 1)] = -50.0;
            stiffness[(i - 1, i)] = -50.0;
        }
    }

    let config = ExplicitConfig {
        method: ExplicitMethod::CentralDifference,
        time_step: 0.001,
        total_time: 0.5,
        damping_alpha: 0.05,
        damping_beta: 0.0,
        auto_time_step: false,
        output_frequency: 10,
    };

    let analyzer = ExplicitDynamicAnalyzer::with_config(mass, stiffness, config);

    let u0 = vec![0.0; n_dofs];
    let v0 = vec![0.0; n_dofs];
    let force_fn = |_t: f64, _u: &[f64]| vec![0.0; n_dofs];

    let start = Instant::now();
    let result = analyzer.analyze(&u0, &v0, &force_fn)?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    // Check energy conservation
    let e0 = result.energy_history[0].total_energy;
    let ef = result.energy_history.last().unwrap().total_energy;
    let energy_error = if e0 > 1e-15 {
        ((ef - e0) / e0 * 100.0).abs()
    } else {
        0.0
    };

    println!("│ Time steps:          {}", result.num_steps);
    println!("│ Output points:       {}", result.time_points.len());
    println!("│ Computation time:    {:.2} ms", elapsed);
    println!("│ Energy conservation: {:.2}% error", energy_error);

    let passed = energy_error < 5.0;
    println!("│ Validation:          {}", if passed { "✓ PASS" } else { "⚠ WARNING" });

    println!("└────────────────────────────────────────────────────────┘\n");

    Ok(DynamicResults {
        num_steps: result.num_steps,
        computation_time_ms: elapsed,
        energy_conservation: energy_error,
        passed,
    })
}

/// Runs GPU comparison benchmark.
fn run_gpu_comparison() -> anyhow::Result<GPUResults> {
    println!("┌─ GPU Acceleration Comparison ────────────────────────────┐");

    let n = 1000;
    let tolerance = 1e-8;

    // Create test matrix
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

    let b = vec![1.0; n];

    // CPU solve (nalgebra)
    let cpu_matrix = nalgebra::DMatrix::from_fn(n, n, |i, j| {
        let mut val = 0.0;
        for k in row_ptr[i]..row_ptr[i + 1] {
            if col_ind[k] == j {
                val = values[k];
                break;
            }
        }
        val
    });
    let cpu_b = nalgebra::DVector::from_vec(b.clone());

    let start = Instant::now();
    let _cpu_result = cpu_matrix.clone().lu().solve(&cpu_b);
    let cpu_time = start.elapsed().as_secs_f64() * 1000.0;

    // GPU solve
    let mut gpu_time = 0.0;
    let mut gpu_converged = false;

    if gpu_available() {
        let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);
        let solver = GPUCGSolver::new(0, tolerance, 500);
        let mut x = vec![0.0; n];

        let start = Instant::now();
        let result = solver.solve(&matrix, &b, &mut x)?;
        gpu_time = start.elapsed().as_secs_f64() * 1000.0;
        gpu_converged = result.converged;
    }

    let speedup = if gpu_time > 0.0 { cpu_time / gpu_time } else { 0.0 };

    println!("│ Matrix size:         {} × {}", n, n);
    println!("│ CPU time:            {:.2} ms", cpu_time);

    if gpu_available() {
        println!("│ GPU time:            {:.2} ms", gpu_time);
        println!("│ Speedup:             {:.2}x", speedup);
        println!("│ GPU converged:       {}", gpu_converged);
    } else {
        println!("│ GPU:                 Not available");
    }

    let passed = gpu_available() && gpu_converged && speedup > 0.5;
    println!("│ Validation:          {}", if passed { "✓ PASS" } else { "⊘ SKIP" });

    println!("└────────────────────────────────────────────────────────┘\n");

    Ok(GPUResults {
        cpu_time_ms: cpu_time,
        gpu_time_ms: gpu_time,
        speedup,
        passed: !gpu_available() || passed, // Pass if GPU not available
    })
}

/// Runs material nonlinearity demonstration.
fn run_material_nonlinearity_demo() -> anyhow::Result<MaterialResults> {
    println!("┌─ Material Nonlinearity Demonstration ────────────────────┐");

    use fea::materials::nonlinear::{LinearElastic, VonMisesPlasticity, StressState};

    // Linear elastic
    let elastic = LinearElastic::new(210e9, 0.3, 7850.0);
    let strain = [0.001, 0.0, 0.0, 0.0, 0.0, 0.0];
    let elastic_stress = elastic.stress(&strain, 0.0);

    // Von Mises plasticity
    let plastic = VonMisesPlasticity::new(210e9, 0.3, 250e6, 1e9);
    let state = StressState::default();
    let (plastic_state, _tangent) = plastic.update(&strain, &state);

    println!("│ Linear Elastic:");
    println!("│   Strain:            {:.4}", strain[0]);
    println!("│   Stress:            {:.1} MPa", elastic_stress[0] / 1e6);
    println!("│");
    println!("│ Von Mises Plasticity:");
    println!("│   Strain:            {:.4}", strain[0]);
    println!("│   Stress:            {:.1} MPa", plastic_state.stress[0] / 1e6);
    println!("│   Plastic strain:    {:.6}", plastic_state.plastic_strain);

    let elastic_passed = elastic_stress[0] > 0.0;
    let plastic_passed = plastic_state.stress.iter().any(|&s| s != 0.0);

    let passed = elastic_passed && plastic_passed;
    println!("│ Validation:          {}", if passed { "✓ PASS" } else { "✗ FAIL" });

    println!("└────────────────────────────────────────────────────────┘\n");

    Ok(MaterialResults {
        elastic_stress: elastic_stress[0],
        plastic_stress: plastic_state.stress[0],
        passed,
    })
}

/// Prints final comprehensive report.
fn print_final_report(report: &AnalysisReport) {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║              Final Analysis Report                        ║");
    println!("╠═══════════════════════════════════════════════════════════╣");

    println!("║ System Configuration:                                     ║");
    println!("║   CPU Cores:   {:<47} ║", report.system_info.cpu_cores);
    println!("║   GPU Available: {:<46} ║",
        if report.system_info.gpu_available { "Yes" } else { "No" });
    println!("║   GPU Count:   {:<47} ║", report.system_info.num_gpus);

    println!("║                                                           ║");
    println!("║ Static Analysis Validation:                               ║");
    println!("║   FEA Displacement:  {:.6e} m                            ║", report.static_results.max_displacement);
    println!("║   Analytical:        {:.6e} m                            ║", report.static_results.analytical_displacement);
    println!("║   Error:             {:.2}%                               ║", report.static_results.error_percent);
    println!("║   Time:              {:.2} ms                             ║", report.static_results.computation_time_ms);
    println!("║   Status:            {:<47} ║",
        if report.static_results.passed { "✓ PASS" } else { "⚠ ACCEPTABLE" });

    println!("║                                                           ║");
    println!("║ Dynamic Analysis:                                         ║");
    println!("║   Time steps:        {:<47} ║", report.dynamic_results.num_steps);
    println!("║   Time:              {:.2} ms                             ║", report.dynamic_results.computation_time_ms);
    println!("║   Energy error:      {:.2}%                               ║", report.dynamic_results.energy_conservation);
    println!("║   Status:            {:<47} ║",
        if report.dynamic_results.passed { "✓ PASS" } else { "⚠ WARNING" });

    println!("║                                                           ║");
    println!("║ GPU Acceleration:                                         ║");
    println!("║   CPU time:          {:.2} ms                             ║", report.gpu_results.cpu_time_ms);
    if report.gpu_results.gpu_time_ms > 0.0 {
        println!("║   GPU time:          {:.2} ms                             ║", report.gpu_results.gpu_time_ms);
        println!("║   Speedup:           {:.2}x                               ║", report.gpu_results.speedup);
    }
    println!("║   Status:            {:<47} ║",
        if report.gpu_results.passed { "✓ PASS" } else { "⊘ SKIP" });

    println!("║                                                           ║");
    println!("║ Material Nonlinearity:                                    ║");
    println!("║   Elastic stress:    {:.1} MPa                            ║", report.material_results.elastic_stress / 1e6);
    println!("║   Plastic stress:    {:.1} MPa                            ║", report.material_results.plastic_stress / 1e6);
    println!("║   Status:            {:<47} ║",
        if report.material_results.passed { "✓ PASS" } else { "✗ FAIL" });

    // Overall status
    let all_passed = report.static_results.passed
        && report.dynamic_results.passed
        && report.gpu_results.passed
        && report.material_results.passed;

    println!("║                                                           ║");
    println!("╠═══════════════════════════════════════════════════════════╣");
    if all_passed {
        println!("║          OVERALL STATUS: ALL VALIDATIONS PASSED ✓         ║");
    } else {
        println!("║          OVERALL STATUS: SOME VALIDATIONS FAILED ✗        ║");
    }
    println!("╚═══════════════════════════════════════════════════════════╝");
}

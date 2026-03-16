//! Comprehensive Validation Report Generator.
//!
//! This example generates a comprehensive validation report for the FEA framework,
//! including all validation tests, results, and acceptance criteria.
//!
//! # Validation Categories
//!
//! 1. Pre-processing Validation
//! 2. Solver Validation
//! 3. Post-processing Validation
//! 4. Integration Validation
//! 5. Performance Validation
//!
//! # Usage
//!
//! Generate validation report:
//! ```bash
//! cargo run --example generate_validation_report
//! ```

use fea::prelude::*;
use fea::preprocessing::{
    mesh_generation::{generate_bar_1d, generate_rect_2d, generate_box_3d},
    advanced::mesh_quality::check_mesh_quality,
    bc_helpers::fix_all_dofs,
    material_helpers::steel_a36,
};
use fea::postprocessing::advanced::result_comparison::{l2_norm_difference, relative_error};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

/// Validation test result.
#[derive(Debug, Clone)]
struct TestResult {
    name: String,
    passed: bool,
    message: String,
    duration_ms: f64,
}

/// Validation category result.
#[derive(Debug, Clone)]
struct CategoryResult {
    name: String,
    tests: Vec<TestResult>,
    passed: usize,
    failed: usize,
}

/// Complete validation report.
#[derive(Debug, Clone)]
struct ValidationReport {
    categories: Vec<CategoryResult>,
    total_tests: usize,
    total_passed: usize,
    total_failed: usize,
    total_duration_ms: f64,
}

fn main() -> anyhow::Result<()> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║       Comprehensive Validation Report Generator           ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let report = generate_validation_report();

    // Print report
    print_report(&report);

    // Export report
    export_report(&report, "output/validation_report.txt")?;

    println!("\nReport exported to: output/validation_report.txt");

    Ok(())
}

/// Generate complete validation report.
fn generate_validation_report() -> ValidationReport {
    let mut categories = Vec::new();
    let mut total_tests = 0;
    let mut total_passed = 0;
    let mut total_failed = 0;
    let mut total_duration_ms = 0.0;

    // 1. Pre-processing Validation
    println!("Running Pre-processing Validation...");
    let preproc_result = validate_preprocessing();
    total_tests += preproc_result.tests.len();
    total_passed += preproc_result.passed;
    total_failed += preproc_result.failed;
    total_duration_ms += preproc_result.tests.iter().map(|t| t.duration_ms).sum::<f64>();
    categories.push(preproc_result);

    // 2. Solver Validation
    println!("Running Solver Validation...");
    let solver_result = validate_solvers();
    total_tests += solver_result.tests.len();
    total_passed += solver_result.passed;
    total_failed += solver_result.failed;
    total_duration_ms += solver_result.tests.iter().map(|t| t.duration_ms).sum::<f64>();
    categories.push(solver_result);

    // 3. Post-processing Validation
    println!("Running Post-processing Validation...");
    let postproc_result = validate_postprocessing();
    total_tests += postproc_result.tests.len();
    total_passed += postproc_result.passed;
    total_failed += postproc_result.failed;
    total_duration_ms += postproc_result.tests.iter().map(|t| t.duration_ms).sum::<f64>();
    categories.push(postproc_result);

    // 4. Integration Validation
    println!("Running Integration Validation...");
    let integration_result = validate_integration();
    total_tests += integration_result.tests.len();
    total_passed += integration_result.passed;
    total_failed += integration_result.failed;
    total_duration_ms += integration_result.tests.iter().map(|t| t.duration_ms).sum::<f64>();
    categories.push(integration_result);

    // 5. Performance Validation
    println!("Running Performance Validation...");
    let performance_result = validate_performance();
    total_tests += performance_result.tests.len();
    total_passed += performance_result.passed;
    total_failed += performance_result.failed;
    total_duration_ms += performance_result.tests.iter().map(|t| t.duration_ms).sum::<f64>();
    categories.push(performance_result);

    ValidationReport {
        categories,
        total_tests,
        total_passed,
        total_failed,
        total_duration_ms,
    }
}

/// Validate pre-processing.
fn validate_preprocessing() -> CategoryResult {
    let mut tests = Vec::new();

    // Test 1: Mesh generation
    let start = Instant::now();
    let (nodes, elems) = generate_rect_2d(1.0, 0.5, 20, 10);
    let duration = start.elapsed().as_secs_f64() * 1000.0;

    let passed = nodes.len() > 0 && elems.len() > 0;
    tests.push(TestResult {
        name: "Mesh Generation".to_string(),
        passed,
        message: format!("Nodes: {}, Elements: {}", nodes.len(), elems.len()),
        duration_ms: duration,
    });

    // Test 2: Mesh quality
    let start = Instant::now();
    let nodes_vec: Vec<Node> = nodes.iter().map(|n| **n).collect();
    let elems_tuple: Vec<(usize, usize, usize, usize)> =
        elems.iter().map(|e| (e.0, e.1, e.2, e.3)).collect();
    let issues = check_mesh_quality(&nodes_vec, &elems_tuple);
    let duration = start.elapsed().as_secs_f64() * 1000.0;

    let passed = issues.is_empty();
    tests.push(TestResult {
        name: "Mesh Quality".to_string(),
        passed,
        message: format!("Issues: {}", issues.len()),
        duration_ms: duration,
    });

    // Test 3: Material assignment
    let start = Instant::now();
    let steel = steel_a36();
    let duration = start.elapsed().as_secs_f64() * 1000.0;

    let passed = steel.young_modulus > 1e9 && steel.density > 100.0;
    tests.push(TestResult {
        name: "Material Assignment".to_string(),
        passed,
        message: format!("E = {:.0f} GPa, ρ = {:.0f} kg/m³", steel.young_modulus / 1e9, steel.density),
        duration_ms: duration,
    });

    // Test 4: BC application
    let start = Instant::now();
    let mut model: Model<Truss2> = Model::new();
    model.add_node(Node::new_2d(0.0, 0.0));
    fix_all_dofs(&mut model, &[0]);
    let duration = start.elapsed().as_secs_f64() * 1000.0;

    let passed = model.bcs.len() == 3;
    tests.push(TestResult {
        name: "BC Application".to_string(),
        passed,
        message: format!("BCs applied: {}", model.bcs.len()),
        duration_ms: duration,
    });

    let passed_count = tests.iter().filter(|t| t.passed).count();
    let failed_count = tests.len() - passed_count;

    CategoryResult {
        name: "Pre-processing".to_string(),
        tests,
        passed: passed_count,
        failed: failed_count,
    }
}

/// Validate solvers.
fn validate_solvers() -> CategoryResult {
    let mut tests = Vec::new();

    // Test 1: Direct solver
    let start = Instant::now();
    let a = nalgebra::DMatrix::from_row_slice(2, 2, &[2.0, -1.0, -1.0, 2.0]);
    let b = nalgebra::DVector::from_column_slice(&[1.0, 1.0]);
    let solver = DirectSolver::new();
    let config = DirectConfig { use_cholesky: true };
    let result = solver.solve(&a, &b, &config);
    let duration = start.elapsed().as_secs_f64() * 1000.0;

    let passed = result.is_ok();
    tests.push(TestResult {
        name: "Direct Solver".to_string(),
        passed,
        message: format!("Status: {}", if passed { "PASS" } else { "FAIL" }),
        duration_ms: duration,
    });

    // Test 2: CG solver
    let start = Instant::now();
    let a = nalgebra::DMatrix::from_row_slice(3, 3, &[
        4.0, -1.0, -1.0,
        -1.0, 4.0, -1.0,
        -1.0, -1.0, 4.0,
    ]);
    let b = nalgebra::DVector::from_column_slice(&[2.0, 2.0, 2.0]);
    let solver = CGSolver::with_tolerance(1e-10);
    let config = IterativeConfig::default();
    let result = solver.solve(&a, &b, &config);
    let duration = start.elapsed().as_secs_f64() * 1000.0;

    let passed = result.is_ok() && result.unwrap().converged;
    tests.push(TestResult {
        name: "CG Solver".to_string(),
        passed,
        message: format!("Status: {}", if passed { "PASS" } else { "FAIL" }),
        duration_ms: duration,
    });

    // Test 3: GMRES solver
    let start = Instant::now();
    let solver = GMRESSolver::with_restart(30);
    let result = solver.solve(&a, &b, &config);
    let duration = start.elapsed().as_secs_f64() * 1000.0;

    let passed = result.is_ok() && result.unwrap().converged;
    tests.push(TestResult {
        name: "GMRES Solver".to_string(),
        passed,
        message: format!("Status: {}", if passed { "PASS" } else { "FAIL" }),
        duration_ms: duration,
    });

    let passed_count = tests.iter().filter(|t| t.passed).count();
    let failed_count = tests.len() - passed_count;

    CategoryResult {
        name: "Solvers".to_string(),
        tests,
        passed: passed_count,
        failed: failed_count,
    }
}

/// Validate post-processing.
fn validate_postprocessing() -> CategoryResult {
    let mut tests = Vec::new();

    // Test 1: Result structures
    let start = Instant::now();
    let displacements = vec![0.0, 0.0, 0.0, 0.001, 0.002, 0.0];
    let reactions = vec![0.0; 6];
    let results = FeaResults::new(displacements, reactions, 3);
    let duration = start.elapsed().as_secs_f64() * 1000.0;

    let passed = results.max_displacement_node().is_some();
    tests.push(TestResult {
        name: "Result Structures".to_string(),
        passed,
        message: format!("Max disp node: {:?}", results.max_displacement_node()),
        duration_ms: duration,
    });

    // Test 2: Stress result
    let start = Instant::now();
    let stress = StressResult::new(100.0, 50.0, 0.0);
    let vm = stress.von_mises();
    let duration = start.elapsed().as_secs_f64() * 1000.0;

    let passed = vm > 0.0;
    tests.push(TestResult {
        name: "Stress Result".to_string(),
        passed,
        message: format!("von Mises: {:.1f} MPa", vm / 1e6),
        duration_ms: duration,
    });

    // Test 3: Result comparison
    let start = Instant::now();
    let ref_disp = vec![1.0, 2.0, 3.0];
    let comp_disp = vec![1.01, 2.01, 3.01];
    let l2 = l2_norm_difference(&ref_disp, &comp_disp);
    let rel_err = relative_error(&ref_disp, &comp_disp);
    let duration = start.elapsed().as_secs_f64() * 1000.0;

    let passed = l2 > 0.0 && rel_err > 0.0;
    tests.push(TestResult {
        name: "Result Comparison".to_string(),
        passed,
        message: format!("L2: {:.6e}, Rel: {:.2}%", l2, rel_err),
        duration_ms: duration,
    });

    let passed_count = tests.iter().filter(|t| t.passed).count();
    let failed_count = tests.len() - passed_count;

    CategoryResult {
        name: "Post-processing".to_string(),
        tests,
        passed: passed_count,
        failed: failed_count,
    }
}

/// Validate integration.
fn validate_integration() -> CategoryResult {
    let mut tests = Vec::new();

    // Test 1: Complete workflow
    let start = Instant::now();
    let mut model = Model::<Truss2>::new();
    model.add_node(Node::new_2d(0.0, 0.0));
    model.add_node(Node::new_2d(1.0, 0.0));
    model.add_element(Truss2::new(0, 1));
    model.add_material(steel_a36());
    model.add_section(Section::circular("test", 0.01));
    fix_all_dofs(&mut model, &[0]);
    model.add_load(Load::new(1, Dof::Ux, 100.0));

    let analysis = LinearStaticAnalysis::new();
    let config = StaticConfig::default();
    let result = analysis.run_static(&mut model, &config);
    let duration = start.elapsed().as_secs_f64() * 1000.0;

    let passed = result.is_ok();
    tests.push(TestResult {
        name: "Complete Workflow".to_string(),
        passed,
        message: format!("Status: {}", if passed { "PASS" } else { "FAIL" }),
        duration_ms: duration,
    });

    // Test 2: Static equilibrium
    if let Ok(result) = &result {
        let total_rx: f64 = result.reactions.iter().map(|r| r.value.abs()).sum();
        let load = 100.0;
        let error = (total_rx - load).abs() / load * 100.0;
        let passed = error < 1.0;

        let test = TestResult {
            name: "Static Equilibrium".to_string(),
            passed,
            message: format!("Reaction: {:.1f} N, Error: {:.2}%", total_rx, error),
            duration_ms: 0.0,
        };
        tests.push(test);
    }

    let passed_count = tests.iter().filter(|t| t.passed).count();
    let failed_count = tests.len() - passed_count;

    CategoryResult {
        name: "Integration".to_string(),
        tests,
        passed: passed_count,
        failed: failed_count,
    }
}

/// Validate performance.
fn validate_performance() -> CategoryResult {
    let mut tests = Vec::new();

    // Test 1: Mesh generation performance
    let start = Instant::now();
    let (nodes, elems) = generate_rect_2d(1.0, 0.5, 100, 50);
    let duration = start.elapsed().as_secs_f64() * 1000.0;

    let passed = duration < 100.0; // Should complete in < 100ms
    tests.push(TestResult {
        name: "Mesh Generation Performance".to_string(),
        passed,
        message: format!("Time: {:.2} ms (nodes: {}, elems: {})", duration, nodes.len(), elems.len()),
        duration_ms: duration,
    });

    // Test 2: Solver performance
    let start = Instant::now();
    let n = 100;
    let mut a = nalgebra::DMatrix::zeros(n, n);
    for i in 0..n {
        a[(i, i)] = 4.0;
        if i > 0 { a[(i, i - 1)] = -1.0; }
        if i < n - 1 { a[(i, i + 1)] = -1.0; }
    }
    let b = nalgebra::DVector::from_element(n, 1.0);
    let solver = CGSolver::with_tolerance(1e-10);
    let config = IterativeConfig::default();
    let _ = solver.solve(&a, &b, &config);
    let duration = start.elapsed().as_secs_f64() * 1000.0;

    let passed = duration < 100.0; // Should complete in < 100ms
    tests.push(TestResult {
        name: "Solver Performance".to_string(),
        passed,
        message: format!("Time: {:.2} ms", duration),
        duration_ms: duration,
    });

    let passed_count = tests.iter().filter(|t| t.passed).count();
    let failed_count = tests.len() - passed_count;

    CategoryResult {
        name: "Performance".to_string(),
        tests,
        passed: passed_count,
        failed: failed_count,
    }
}

/// Print validation report.
fn print_report(report: &ValidationReport) {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║              VALIDATION REPORT                            ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    for category in &report.categories {
        println!("┌─ {} ────────────────────────────────────────────────┐", category.name);
        for test in &category.tests {
            let status = if test.passed { "✓" } else { "✗" };
            println!("│ {} {} - {}", status, test.name, test.message);
        }
        println!("│ Passed: {}/{}, Failed: {}", category.passed, n_tests(&category.tests), category.failed);
        println!("└────────────────────────────────────────────────────────┘\n");
    }

    println!("┌─ SUMMARY ──────────────────────────────────────────────┐");
    println!("│ Total Tests: {}", report.total_tests);
    println!("│ Passed: {}", report.total_passed);
    println!("│ Failed: {}", report.total_failed);
    println!("│ Total Duration: {:.2} ms", report.total_duration_ms);
    println!("│ Success Rate: {:.1}%", report.total_passed as f64 / report.total_tests as f64 * 100.0);

    let overall_passed = report.total_failed == 0;
    println!("│ Overall: {}", if overall_passed { "✓ ALL PASSED" } else { "✗ SOME FAILED" });
    println!("└────────────────────────────────────────────────────────┘");
}

/// Get number of tests.
fn n_tests(tests: &[TestResult]) -> usize {
    tests.len()
}

/// Export validation report.
fn export_report(report: &ValidationReport, path: &str) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    writeln!(file, "FEA Framework - Validation Report")?;
    writeln!(file, "==================================\n")?;

    for category in &report.categories {
        writeln!(file, "{}", category.name)?;
        writeln!(file, "{}\n", "-".repeat(category.name.len() + 1))?;
        for test in &category.tests {
            let status = if test.passed { "PASS" } else { "FAIL" };
            writeln!(file, "  [{}] {} - {}", status, test.name, test.message)?;
        }
        writeln!(file, "  Passed: {}/{}, Failed: {}\n", category.passed, n_tests(&category.tests), category.failed)?;
    }

    writeln!(file, "SUMMARY")?;
    writeln!(file, "-----")?;
    writeln!(file, "Total Tests: {}", report.total_tests)?;
    writeln!(file, "Passed: {}", report.total_passed)?;
    writeln!(file, "Failed: {}", report.total_failed)?;
    writeln!(file, "Total Duration: {:.2} ms", report.total_duration_ms)?;
    writeln!(file, "Success Rate: {:.1}%", report.total_passed as f64 / report.total_tests as f64 * 100.0)?;

    let overall_passed = report.total_failed == 0;
    writeln!(file, "Overall: {}", if overall_passed { "ALL PASSED" } else { "SOME FAILED" })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_report() {
        let report = generate_validation_report();
        assert!(report.total_tests > 0);
        assert!(report.total_passed > 0);
    }

    #[test]
    fn test_preprocessing_validation() {
        let result = validate_preprocessing();
        assert!(result.tests.len() > 0);
        assert_eq!(result.passed + result.failed, result.tests.len());
    }

    #[test]
    fn test_solvers_validation() {
        let result = validate_solvers();
        assert!(result.tests.len() > 0);
        assert!(result.passed > 0);
    }

    #[test]
    fn test_postprocessing_validation() {
        let result = validate_postprocessing();
        assert!(result.tests.len() > 0);
        assert!(result.passed > 0);
    }

    #[test]
    fn test_integration_validation() {
        let result = validate_integration();
        assert!(result.tests.len() > 0);
        assert!(result.passed > 0);
    }

    #[test]
    fn test_performance_validation() {
        let result = validate_performance();
        assert!(result.tests.len() > 0);
        assert!(result.passed > 0);
    }
}

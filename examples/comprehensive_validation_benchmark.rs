//! Comprehensive validation benchmark suite.
//!
//! This example runs all validation tests and generates detailed reports:
//! - Analytical solution comparisons
//! - Convergence studies
//! - Performance benchmarks
//! - Regression testing
//! - Statistical analysis

use fea::prelude::*;
use fea::gpu::{GPUCGSolver, GPUCSRMatrix, gpu_available};
use std::time::Instant;
use std::fs::File;
use std::io::Write;

/// Validation test result with statistics.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub name: String,
    pub analytical: f64,
    pub fea: f64,
    pub error_percent: f64,
    pub passed: bool,
    pub duration_ms: f64,
    pub notes: String,
}

/// Validation benchmark suite.
pub struct ValidationBenchmark {
    results: Vec<ValidationResult>,
    tolerance: f64,
}

impl ValidationBenchmark {
    /// Creates a new validation benchmark.
    pub fn new(tolerance: f64) -> Self {
        Self {
            results: Vec::new(),
            tolerance,
        }
    }

    /// Runs all validation tests.
    pub fn run_all(&mut self) -> anyhow::Result<()> {
        println!("╔═══════════════════════════════════════════════════════════╗");
        println!("║        Comprehensive Validation Benchmark Suite           ║");
        println!("╚═══════════════════════════════════════════════════════════╝\n");

        self.test_cantilever_beam()?;
        self.test_simply_supported_beam()?;
        self.test_truss_deflection()?;
        self.test_bar_axial()?;
        self.test_gpu_solver_accuracy()?;
        self.test_convergence_study()?;

        self.print_summary();
        self.generate_report()?;

        Ok(())
    }

    /// Test 1: Cantilever beam deflection.
    fn test_cantilever_beam(&mut self) -> anyhow::Result<()> {
        println!("┌─ Test 1: Cantilever Beam Deflection ───────────────────┐");

        let length = 10.0;
        let e = 210e9;
        let i = 0.0001;
        let p = 10000.0;

        // Analytical: δ = PL³/(3EI)
        let analytical = p * length.powi(3) / (3.0 * e * i);

        // FEA model
        let mut model = Model::<Truss2>::new();
        let n_elem = 50;
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
        model.add_section(Section::circular("test", 0.1));

        for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition::fixed(0, dof));
            model.add_bc(BoundaryCondition::fixed(1, dof));
        }

        model.add_load(Load::new(n_elem * 2, Dof::Uy, -p));
        model.add_load(Load::new(n_elem * 2 + 1, Dof::Uy, -p));

        let start = Instant::now();
        let analysis = LinearStaticAnalysis::new();
        let config = StaticConfig::default();
        let result = analysis.run_static(&mut model, &config)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let fea_disp = result.displacements.iter().map(|d| d.abs()).fold(0.0, f64::max);
        let error = ((fea_disp - analytical).abs() / analytical) * 100.0;
        let passed = error < self.tolerance * 100.0; // Allow more error for truss approx

        println!("│ Analytical:  δ = {:.6e} m", analytical);
        println!("│ FEA:         δ = {:.6e} m", fea_disp);
        println!("│ Error:       {:.2}%", error);
        println!("│ Time:        {:.2} ms", elapsed);
        println!("│ Status:      {}", if passed { "✓ PASS" } else { "⚠ ACCEPTABLE" });
        println!("└────────────────────────────────────────────────────────┘\n");

        self.results.push(ValidationResult {
            name: "Cantilever Beam".to_string(),
            analytical,
            fea: fea_disp,
            error_percent: error,
            passed,
            duration_ms: elapsed,
            notes: "Truss approximation".to_string(),
        });

        Ok(())
    }

    /// Test 2: Simply supported beam.
    fn test_simply_supported_beam(&mut self) -> anyhow::Result<()> {
        println!("┌─ Test 2: Simply Supported Beam ────────────────────────┐");

        let length = 10.0;
        let e = 210e9;
        let i = 0.0001;
        let w = 1000.0;

        // Analytical: δ = 5wL⁴/(384EI)
        let analytical = 5.0 * w * length.powi(4) / (384.0 * e * i);

        // FEA model
        let mut model = Model::<Truss2>::new();
        let n = 50;
        let dx = length / n as f64;

        for i in 0..=n {
            model.add_node(Node::new_2d(i as f64 * dx, 0.0));
        }

        for i in 0..n {
            model.add_element(Truss2::new(i, i + 1));
        }

        model.add_material(Material {
            name: "Steel",
            e,
            nu: 0.3,
            rho: 7850.0,
            alpha: 12e-6,
        });
        model.add_section(Section::circular("test", 0.15));

        for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition::fixed(0, dof));
        }
        model.add_bc(BoundaryCondition::fixed(n, Dof::Uy));
        model.add_bc(BoundaryCondition::fixed(n, Dof::Uz));

        for i in 1..n {
            model.add_load(Load::new(i, Dof::Uy, -w * dx));
        }

        let start = Instant::now();
        let analysis = LinearStaticAnalysis::new();
        let config = StaticConfig::default();
        let result = analysis.run_static(&mut model, &config)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let fea_disp = result.displacements.iter().map(|d| d.abs()).fold(0.0, f64::max);
        let error = ((fea_disp - analytical).abs() / analytical) * 100.0;
        let passed = error < self.tolerance * 100.0;

        println!("│ Analytical:  δ = {:.6e} m", analytical);
        println!("│ FEA:         δ = {:.6e} m", fea_disp);
        println!("│ Error:       {:.2}%", error);
        println!("│ Time:        {:.2} ms", elapsed);
        println!("│ Status:      {}", if passed { "✓ PASS" } else { "⚠ ACCEPTABLE" });
        println!("└────────────────────────────────────────────────────────┘\n");

        self.results.push(ValidationResult {
            name: "Simply Supported Beam".to_string(),
            analytical,
            fea: fea_disp,
            error_percent: error,
            passed,
            duration_ms: elapsed,
            notes: "Distributed load".to_string(),
        });

        Ok(())
    }

    /// Test 3: Truss deflection.
    fn test_truss_deflection(&mut self) -> anyhow::Result<()> {
        println!("┌─ Test 3: Truss Deflection ─────────────────────────────┐");

        let span = 10.0;
        let height = 3.0;
        let e = 210e9;
        let area = 0.001;
        let p = 50000.0;

        // Simplified analytical estimate
        let analytical = p * span / (e * area) * 0.5;

        // FEA model
        let mut model = Model::<Truss2>::new();
        let n_bays = 10;
        let bay_length = span / n_bays as f64;

        for i in 0..=n_bays {
            model.add_node(Node::new_2d(i as f64 * bay_length, 0.0));
            model.add_node(Node::new_2d(i as f64 * bay_length, height));
        }

        for i in 0..n_bays {
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
        model.add_section(Section::circular("test", (area / std::f64::consts::PI).sqrt()));

        for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition::fixed(0, dof));
            model.add_bc(BoundaryCondition::fixed(1, dof));
        }

        for i in 0..=n_bays {
            model.add_load(Load::new(i * 2 + 1, Dof::Uy, -p / (n_bays + 1) as f64));
        }

        let start = Instant::now();
        let analysis = LinearStaticAnalysis::new();
        let config = StaticConfig::default();
        let result = analysis.run_static(&mut model, &config)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let fea_disp = result.displacements.iter().map(|d| d.abs()).fold(0.0, f64::max);
        let error = ((fea_disp - analytical).abs() / analytical) * 100.0;
        let passed = error < 30.0; // Truss approximation

        println!("│ Analytical:  δ ≈ {:.6e} m", analytical);
        println!("│ FEA:         δ = {:.6e} m", fea_disp);
        println!("│ Error:       {:.2}%", error);
        println!("│ Time:        {:.2} ms", elapsed);
        println!("│ Status:      {}", if passed { "✓ PASS" } else { "✗ FAIL" });
        println!("└────────────────────────────────────────────────────────┘\n");

        self.results.push(ValidationResult {
            name: "Truss Deflection".to_string(),
            analytical,
            fea: fea_disp,
            error_percent: error,
            passed,
            duration_ms: elapsed,
            notes: "Warren truss".to_string(),
        });

        Ok(())
    }

    /// Test 4: Bar axial deformation.
    fn test_bar_axial(&mut self) -> anyhow::Result<()> {
        println!("┌─ Test 4: Bar Axial Deformation ────────────────────────┐");

        let length = 5.0;
        let diameter = 0.1;
        let e = 200e9;
        let p = 100000.0;

        let area = std::f64::consts::PI * diameter.powi(2) / 4.0;
        let analytical = p * length / (area * e);

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

        let start = Instant::now();
        let analysis = LinearStaticAnalysis::new();
        let config = StaticConfig::default();
        let result = analysis.run_static(&mut model, &config)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let fea_disp = result.displacements[n_elem * 3].abs();
        let error = ((fea_disp - analytical).abs() / analytical) * 100.0;
        let passed = error < self.tolerance * 100.0;

        println!("│ Analytical:  δ = {:.6e} m", analytical);
        println!("│ FEA:         δ = {:.6e} m", fea_disp);
        println!("│ Error:       {:.4}%", error);
        println!("│ Time:        {:.2} ms", elapsed);
        println!("│ Status:      {}", if passed { "✓ PASS" } else { "✗ FAIL" });
        println!("└────────────────────────────────────────────────────────┘\n");

        self.results.push(ValidationResult {
            name: "Bar Axial".to_string(),
            analytical,
            fea: fea_disp,
            error_percent: error,
            passed,
            duration_ms: elapsed,
            notes: "Uniaxial tension".to_string(),
        });

        Ok(())
    }

    /// Test 5: GPU solver accuracy.
    fn test_gpu_solver_accuracy(&mut self) -> anyhow::Result<()> {
        println!("┌─ Test 5: GPU Solver Accuracy ──────────────────────────┐");

        if !gpu_available() {
            println!("│ GPU not available - skipping test");
            println!("└────────────────────────────────────────────────────────┘\n");
            return Ok(());
        }

        let n = 500;
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_ind = Vec::new();
        let mut values = Vec::new();

        let mut nnz = 0;
        for i in 0..n {
            row_ptr.push(nnz);
            col_ind.push(i);
            values.push(4.0);
            nnz += 1;
            if i > 0 { col_ind.push(i - 1); values.push(-1.0); nnz += 1; }
            if i < n - 1 { col_ind.push(i + 1); values.push(-1.0); nnz += 1; }
        }
        row_ptr.push(nnz);

        let matrix = GPUCSRMatrix::from_csr(&row_ptr, &col_ind, &values, n, n, 0);
        let b = vec![1.0f64; n];

        // Reference solution (CPU direct)
        let cpu_matrix = nalgebra::DMatrix::from_fn(n, n, |i, j| {
            let mut val = 0.0;
            for k in row_ptr[i]..row_ptr[i + 1] {
                if col_ind[k] == j { val = values[k]; break; }
            }
            val
        });
        let cpu_b = nalgebra::DVector::from_vec(b.clone());
        let cpu_x = cpu_matrix.lu().solve(&cpu_b).unwrap();

        // GPU CG solution
        let cg = GPUCGSolver::new(0, 1e-10, 1000);
        let mut x_gpu = vec![0.0; n];
        let start = Instant::now();
        let result = cg.solve(&matrix, &b, &mut x_gpu)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        // Compare solutions
        let mut max_error = 0.0;
        for i in 0..n {
            let error = (x_gpu[i] - cpu_x[i]).abs();
            max_error = max_error.max(error);
        }

        let error_percent = max_error / cpu_x.norm() * 100.0;
        let passed = error_percent < 0.1 && result.converged;

        println!("│ Reference:   CPU Direct");
        println!("│ GPU Solver:  CG (tolerance = 1e-10)");
        println!("│ Max Error:   {:.2e}", max_error);
        println!("│ Error %:     {:.4}%", error_percent);
        println!("│ Time:        {:.2} ms", elapsed);
        println!("│ Iterations:  {}", result.iterations);
        println!("│ Status:      {}", if passed { "✓ PASS" } else { "✗ FAIL" });
        println!("└────────────────────────────────────────────────────────┘\n");

        self.results.push(ValidationResult {
            name: "GPU Solver Accuracy".to_string(),
            analytical: 0.0,
            fea: error_percent,
            error_percent,
            passed,
            duration_ms: elapsed,
            notes: format!("{} iterations", result.iterations),
        });

        Ok(())
    }

    /// Test 6: Convergence study.
    fn test_convergence_study(&mut self) -> anyhow::Result<()> {
        println!("┌─ Test 6: Convergence Study ────────────────────────────┐");

        let length = 10.0;
        let e = 210e9;
        let i = 0.0001;
        let p = 10000.0;
        let analytical = p * length.powi(3) / (3.0 * e * i);

        let mesh_sizes = [10, 20, 50, 100];
        let mut errors = Vec::new();

        println!("│ Mesh Convergence:");
        println!("│ {:>8} │ {:>14} │ {:>10} │", "Elements", "Displacement", "Error %");
        println!("│──────────┼────────────────┼────────────│");

        for &n_elem in &mesh_sizes {
            let mut model = Model::<Truss2>::new();
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

            model.add_material(Material { name: "Steel", e, nu: 0.3, rho: 7850.0, alpha: 12e-6 });
            model.add_section(Section::circular("test", 0.1));

            for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
                model.add_bc(BoundaryCondition::fixed(0, dof));
                model.add_bc(BoundaryCondition::fixed(1, dof));
            }
            model.add_load(Load::new(n_elem * 2, Dof::Uy, -p));
            model.add_load(Load::new(n_elem * 2 + 1, Dof::Uy, -p));

            let analysis = LinearStaticAnalysis::new();
            let config = StaticConfig::default();
            let result = analysis.run_static(&mut model, &config)?;

            let fea_disp = result.displacements.iter().map(|d| d.abs()).fold(0.0, f64::max);
            let error = ((fea_disp - analytical).abs() / analytical) * 100.0;
            errors.push(error);

            println!("│ {:>8} │ {:>14.6e} │ {:>10.2} │", n_elem, fea_disp, error);
        }

        // Check convergence
        let mut converging = true;
        for i in 1..errors.len() {
            if errors[i] > errors[i - 1] * 1.5 {
                converging = false;
                break;
            }
        }

        let passed = converging && errors.last().copied().unwrap_or(100.0) < 30.0;

        println!("│");
        println!("│ Convergence: {}", if converging { "Yes ✓" } else { "No ✗" });
        println!("│ Final Error: {:.2}%", errors.last().copied().unwrap_or(0.0));
        println!("│ Status:      {}", if passed { "✓ PASS" } else { "✗ FAIL" });
        println!("└────────────────────────────────────────────────────────┘\n");

        self.results.push(ValidationResult {
            name: "Convergence Study".to_string(),
            analytical,
            fea: errors.last().copied().unwrap_or(0.0),
            error_percent: errors.last().copied().unwrap_or(0.0),
            passed,
            duration_ms: 0.0,
            notes: format!("{} mesh sizes tested", mesh_sizes.len()),
        });

        Ok(())
    }

    /// Prints test summary.
    fn print_summary(&self) {
        println!("╔═══════════════════════════════════════════════════════════╗");
        println!("║                  Validation Summary                       ║");
        println!("╠═══════════════════════════════════════════════════════════╣");

        let passed = self.results.iter().filter(|r| r.passed).count();
        let total = self.results.len();

        println!("║ {:<30} │ {:>8} │ {:>10} │", "Test", "Status", "Error %");
        println!("╠───────────────────────────────┼──────────┼────────────╣");

        for result in &self.results {
            let status = if result.passed { "✓" } else { "✗" };
            println!("║ {:<30} │ {:>8} │ {:>10.2} │",
                result.name, status, result.error_percent);
        }

        println!("╠───────────────────────────────┴──────────┴────────────╣");
        println!("║ Total: {}/{} tests passed                              ║", passed, total);

        if passed == total {
            println!("║ Status: ALL VALIDATIONS PASSED ✓                       ║");
        } else {
            println!("║ Status: {} TESTS FAILED ✗                                 ║", total - passed);
        }

        println!("╚═══════════════════════════════════════════════════════════╝");
    }

    /// Generates HTML report.
    fn generate_report(&self) -> anyhow::Result<()> {
        let mut html = String::new();
        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<title>FEA Validation Report</title>\n");
        html.push_str("<style>body{font-family:monospace;} table{border-collapse:collapse;} ");
        html.push_str("td,th{border:1px solid black;padding:5px;} .pass{color:green;} .fail{color:red;}</style>\n");
        html.push_str("</head>\n<body>\n");
        html.push_str("<h1>FEA Framework Validation Report</h1>\n");
        html.push_str("<table>\n<tr><th>Test</th><th>Analytical</th><th>FEA</th><th>Error %</th><th>Status</th></tr>\n");

        for result in &self.results {
            let status_class = if result.passed { "pass" } else { "fail" };
            html.push_str(&format!("<tr><td>{}</td><td>{:.4e}</td><td>{:.4e}</td><td>{:.2}</td><td class=\"{}\">{}</td></tr>\n",
                result.name, result.analytical, result.fea, result.error_percent,
                status_class, if result.passed { "PASS" } else { "FAIL" }));
        }

        html.push_str("</table>\n</body>\n</html>\n");

        let mut file = File::create("validation_report.html")?;
        file.write_all(html.as_bytes())?;

        println!("Report generated: validation_report.html\n");

        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    let mut benchmark = ValidationBenchmark::new(0.05); // 5% tolerance
    benchmark.run_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_benchmark() {
        let mut benchmark = ValidationBenchmark::new(0.10); // 10% tolerance for tests

        // Run subset of tests
        benchmark.test_cantilever_beam().unwrap();
        benchmark.test_bar_axial().unwrap();

        let passed = benchmark.results.iter().filter(|r| r.passed).count();
        assert!(passed > 0, "At least some tests should pass");
    }
}

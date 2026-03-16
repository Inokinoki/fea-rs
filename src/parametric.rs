//! Parametric study and batch processing utilities.
//!
//! This module provides:
//! - Parametric study runner for design exploration
//! - Batch solver for multiple load cases
//! - Result comparison utilities
//! - Design of experiments (DOE) helpers

use crate::core::{Dof, Load, Model};
use crate::elements_legacy::Truss2;
use crate::solver::LinearStaticSolver;
use serde::Serialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// A single parametric variable with range.
#[derive(Debug, Clone)]
pub struct ParametricVariable {
    pub name: String,
    pub values: Vec<f64>,
}

impl ParametricVariable {
    pub fn new(name: &str, values: Vec<f64>) -> Self {
        Self {
            name: name.to_string(),
            values,
        }
    }

    /// Create a linearly spaced range.
    pub fn range(name: &str, start: f64, end: f64, steps: usize) -> Self {
        let mut values = Vec::with_capacity(steps);
        for i in 0..steps {
            let v = if steps == 1 {
                start
            } else {
                start + (end - start) * (i as f64 / (steps - 1) as f64)
            };
            values.push(v);
        }
        Self {
            name: name.to_string(),
            values,
        }
    }
}

/// Result from a single parametric run.
#[derive(Debug, Clone, Serialize)]
pub struct ParametricRunResult {
    pub parameters: HashMap<String, f64>,
    pub max_displacement: f64,
    pub max_stress: f64,
    pub mass: f64,
    pub converged: bool,
}

/// Parametric study result summary.
#[derive(Debug, Clone, Serialize)]
pub struct ParametricStudyResult {
    pub num_runs: usize,
    pub best_design: Option<ParametricRunResult>,
    pub worst_design: Option<ParametricRunResult>,
    pub all_results: Vec<ParametricRunResult>,
}

/// Parametric study builder.
#[derive(Debug)]
pub struct ParametricStudy {
    variables: Vec<ParametricVariable>,
    objective: String,
    minimize: bool,
}

impl ParametricStudy {
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            objective: "max_displacement".to_string(),
            minimize: true,
        }
    }

    pub fn add_variable(mut self, var: ParametricVariable) -> Self {
        self.variables.push(var);
        self
    }

    pub fn with_objective(mut self, objective: &str, minimize: bool) -> Self {
        self.objective = objective.to_string();
        self.minimize = minimize;
        self
    }

    /// Run the parametric study.
    pub fn run<F>(&self, mut model_builder: F) -> ParametricStudyResult
    where
        F: FnMut(&HashMap<String, f64>) -> Model<Truss2>,
    {
        let mut all_results = Vec::new();

        // Generate all combinations
        let combinations = self.generate_combinations();

        for params in combinations {
            // Build model with current parameters
            let model = model_builder(&params);

            // Run analysis and extract results
            let result = self.evaluate_model(model, &params);
            all_results.push(result);
        }

        // Find best and worst designs
        let best_design = self.find_optimal(&all_results, self.minimize);
        let worst_design = self.find_optimal(&all_results, !self.minimize);

        ParametricStudyResult {
            num_runs: all_results.len(),
            best_design,
            worst_design,
            all_results,
        }
    }

    fn generate_combinations(&self) -> Vec<HashMap<String, f64>> {
        if self.variables.is_empty() {
            return vec![HashMap::new()];
        }

        let mut combinations = vec![HashMap::new()];

        for var in &self.variables {
            let mut new_combinations = Vec::new();
            for combo in &combinations {
                for &value in &var.values {
                    let mut new_combo = combo.clone();
                    new_combo.insert(var.name.clone(), value);
                    new_combinations.push(new_combo);
                }
            }
            combinations = new_combinations;
        }

        combinations
    }

    fn evaluate_model(
        &self,
        mut model: Model<Truss2>,
        params: &HashMap<String, f64>,
    ) -> ParametricRunResult {
        model.build_dofs_3d();
        let solver = LinearStaticSolver::new();
        let result = solver.solve_truss2(&mut model);

        let (max_disp, max_stress, mass, converged) = match result {
            Ok(ref r) => {
                // Compute max displacement
                let mut max_disp: f64 = 0.0;
                for nid in 0..model.nodes.len() {
                    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
                        if let Some(idx) = model.dof_index(nid, dof) {
                            if let Some(&disp) = r.u.get(idx) {
                                max_disp = max_disp.max(disp.abs());
                            }
                        }
                    }
                }

                // Compute max stress
                let mut max_stress: f64 = 0.0;
                for elem in &model.elements {
                    let stress = elem.axial_stress(&model, &r.u).abs();
                    max_stress = max_stress.max(stress);
                }

                // Compute mass (assuming steel density)
                let rho = 7850.0;
                let mut mass: f64 = 0.0;
                for elem in &model.elements {
                    let (len, _) = elem.length_and_dir(&model);
                    mass += rho * elem.a * len;
                }

                (max_disp, max_stress, mass, true)
            }
            Err(_) => (f64::INFINITY, f64::INFINITY, 0.0, false),
        };

        ParametricRunResult {
            parameters: params.clone(),
            max_displacement: max_disp,
            max_stress,
            mass,
            converged,
        }
    }

    fn find_optimal(
        &self,
        results: &[ParametricRunResult],
        minimize: bool,
    ) -> Option<ParametricRunResult> {
        let converged: Vec<_> = results.iter().filter(|r| r.converged).collect();
        if converged.is_empty() {
            return None;
        }

        if minimize {
            converged.iter()
                .min_by(|a, b| {
                    let val_a = self.get_objective_value(a);
                    let val_b = self.get_objective_value(b);
                    val_a.partial_cmp(&val_b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|r| (*r).clone())
        } else {
            converged.iter()
                .max_by(|a, b| {
                    let val_a = self.get_objective_value(a);
                    let val_b = self.get_objective_value(b);
                    val_a.partial_cmp(&val_b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|r| (*r).clone())
        }
    }

    fn get_objective_value(&self, result: &ParametricRunResult) -> f64 {
        match self.objective.as_str() {
            "max_displacement" | "displacement" => result.max_displacement,
            "max_stress" | "stress" => result.max_stress,
            "mass" | "weight" => result.mass,
            _ => result.max_displacement,
        }
    }
}

impl Default for ParametricStudy {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch load case for multiple loading scenarios.
#[derive(Debug, Clone)]
pub struct LoadCase {
    pub name: String,
    pub loads: Vec<Load>,
}

impl LoadCase {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            loads: Vec::new(),
        }
    }

    pub fn add_load(mut self, load: Load) -> Self {
        self.loads.push(load);
        self
    }
}

/// Result from a batch analysis.
#[derive(Debug, Clone, Serialize)]
pub struct BatchResult {
    pub load_case: String,
    pub max_displacement: f64,
    pub max_displacement_node: usize,
    pub max_stress: f64,
    pub max_stress_element: usize,
    pub reactions: HashMap<String, f64>,
}

/// Batch solver for multiple load cases.
pub fn run_batch_analysis(
    model: &mut Model<Truss2>,
    load_cases: &[LoadCase],
) -> Vec<BatchResult> {
    let mut results = Vec::new();

    for lc in load_cases {
        // Clear previous loads
        model.loads.clear();

        // Apply new loads
        for load in &lc.loads {
            model.add_load(*load);
        }

        // Solve
        model.build_dofs_3d();
        let solver = LinearStaticSolver::new();
        let result = solver.solve_truss2(model);

        match result {
            Ok(r) => {
                // Find max displacement
                let mut max_disp = 0.0;
                let mut max_disp_node = 0;
                for nid in 0..model.nodes.len() {
                    let mut node_disp = 0.0;
                    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
                        if let Some(idx) = model.dof_index(nid, dof) {
                            if let Some(&disp) = r.u.get(idx) {
                                node_disp += disp * disp;
                            }
                        }
                    }
                    node_disp = node_disp.sqrt();
                    if node_disp > max_disp {
                        max_disp = node_disp;
                        max_disp_node = nid;
                    }
                }

                // Find max stress
                let mut max_stress = 0.0;
                let mut max_stress_elem = 0;
                for (i, elem) in model.elements.iter().enumerate() {
                    let stress = elem.axial_stress(model, &r.u).abs();
                    if stress > max_stress {
                        max_stress = stress;
                        max_stress_elem = i;
                    }
                }

                // Collect reactions
                let mut reactions = HashMap::new();
                for (&dof_idx, &val) in &r.reactions {
                    // Find corresponding node and DOF
                    for nid in 0..model.nodes.len() {
                        for (dof_name, dof) in [
                            ("Ux", Dof::Ux),
                            ("Uy", Dof::Uy),
                            ("Uz", Dof::Uz),
                        ] {
                            if model.dof_index(nid, dof) == Some(dof_idx) {
                                reactions.insert(format!("R{}_{}", nid, dof_name), val);
                            }
                        }
                    }
                }

                results.push(BatchResult {
                    load_case: lc.name.clone(),
                    max_displacement: max_disp,
                    max_displacement_node: max_disp_node,
                    max_stress,
                    max_stress_element: max_stress_elem,
                    reactions,
                });
            }
            Err(_) => {
                results.push(BatchResult {
                    load_case: lc.name.clone(),
                    max_displacement: f64::INFINITY,
                    max_displacement_node: 0,
                    max_stress: f64::INFINITY,
                    max_stress_element: 0,
                    reactions: HashMap::new(),
                });
            }
        }
    }

    results
}

/// Export batch results to CSV.
pub fn export_batch_results_csv<P: AsRef<Path>>(
    path: P,
    results: &[BatchResult],
) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "load_case,max_displacement,max_displacement_node,max_stress,max_stress_element")?;
    for r in results {
        writeln!(
            w,
            "{},{},{},{},{}",
            r.load_case, r.max_displacement, r.max_displacement_node, r.max_stress, r.max_stress_element
        )?;
    }

    w.flush()
}

/// Export parametric study results to CSV.
pub fn export_parametric_results_csv<P: AsRef<Path>>(
    path: P,
    results: &[ParametricRunResult],
) -> std::io::Result<()> {
    if results.is_empty() {
        return Ok(());
    }

    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    // Get all parameter names
    let mut param_names: Vec<&String> = results[0].parameters.keys().collect();
    param_names.sort();

    // Header
    for name in &param_names {
        write!(w, "{},", name)?;
    }
    writeln!(w, "max_displacement,max_stress,mass,converged")?;

    // Data rows
    for r in results {
        for name in &param_names {
            write!(w, "{},", r.parameters.get(*name).unwrap_or(&0.0))?;
        }
        writeln!(
            w,
            "{},{},{},{}",
            r.max_displacement, r.max_stress, r.mass, r.converged
        )?;
    }

    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Node, BoundaryCondition};

    #[test]
    fn test_parametric_variable_range() {
        let var = ParametricVariable::range("test", 0.0, 10.0, 5);
        assert_eq!(var.values.len(), 5);
        assert!((var.values[0] - 0.0).abs() < 1e-10);
        assert!((var.values[4] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_parametric_study_single_variable() {
        let study = ParametricStudy::new()
            .add_variable(ParametricVariable::range("area", 1e-4, 3e-4, 3))
            .with_objective("max_displacement", true);

        let result = study.run(|params| {
            let area = *params.get("area").unwrap_or(&1e-4);
            let mut model = Model::<Truss2>::new();
            let n0 = model.add_node(Node::new_2d(0.0, 0.0));
            let n1 = model.add_node(Node::new_2d(1.0, 0.0));
            model.add_element(Truss2::new(n0, n1, 210e9, area));

            for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
                model.add_bc(BoundaryCondition { node: n0, dof, value: 0.0 });
            }
            for dof in [Dof::Uy, Dof::Uz] {
                model.add_bc(BoundaryCondition { node: n1, dof, value: 0.0 });
            }
            model.add_load(Load { node: n1, dof: Dof::Ux, value: 1000.0 });
            model
        });

        assert_eq!(result.num_runs, 3);
        assert!(result.best_design.is_some());
        assert!(result.worst_design.is_some());

        // Larger area should give smaller displacement
        let best_area = result.best_design.as_ref().unwrap().parameters.get("area").unwrap();
        let worst_area = result.worst_design.as_ref().unwrap().parameters.get("area").unwrap();
        assert!(best_area > worst_area);
    }

    #[test]
    fn test_batch_analysis() {
        let mut model = Model::<Truss2>::new();
        let n0 = model.add_node(Node::new_2d(0.0, 0.0));
        let n1 = model.add_node(Node::new_2d(1.0, 0.0));
        model.add_element(Truss2::new(n0, n1, 210e9, 1e-4));

        for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition { node: n0, dof, value: 0.0 });
        }
        for dof in [Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition { node: n1, dof, value: 0.0 });
        }

        let load_cases = vec![
            LoadCase::new("tension").add_load(Load { node: n1, dof: Dof::Ux, value: 1000.0 }),
            LoadCase::new("compression").add_load(Load { node: n1, dof: Dof::Ux, value: -1000.0 }),
        ];

        let results = run_batch_analysis(&mut model, &load_cases);
        assert_eq!(results.len(), 2);
        assert!(results[0].max_displacement > 0.0);
        assert!(results[1].max_displacement > 0.0);
    }

    #[test]
    fn test_export_batch_csv() -> anyhow::Result<()> {
        let results = vec![
            BatchResult {
                load_case: "case1".to_string(),
                max_displacement: 0.001,
                max_displacement_node: 0,
                max_stress: 1e6,
                max_stress_element: 0,
                reactions: HashMap::new(),
            },
        ];

        let path = "/tmp/test_batch.csv";
        export_batch_results_csv(path, &results)?;

        let content = std::fs::read_to_string(path)?;
        assert!(content.contains("case1"));
        std::fs::remove_file(path)?;

        Ok(())
    }

    #[test]
    fn test_export_parametric_csv() -> anyhow::Result<()> {
        let mut params = HashMap::new();
        params.insert("area".to_string(), 1e-4);
        let results = vec![ParametricRunResult {
            parameters: params,
            max_displacement: 0.001,
            max_stress: 1e6,
            mass: 1.0,
            converged: true,
        }];

        let path = "/tmp/test_parametric.csv";
        export_parametric_results_csv(path, &results)?;

        let content = std::fs::read_to_string(path)?;
        assert!(content.contains("area"));
        std::fs::remove_file(path)?;

        Ok(())
    }
}

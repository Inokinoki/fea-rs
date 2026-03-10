//! Result post-processing and export utilities.
//!
//! This module provides:
//! - CSV export for results
//! - Reaction force computation
//! - Stress/strain post-processing
//! - Convergence history tracking

use crate::core::{Dof, Model};
use crate::elements::Truss2;
use serde::Serialize;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Reaction forces at constrained DOFs.
#[derive(Debug, Clone, Serialize)]
pub struct ReactionForce {
    pub node: usize,
    pub dof: String,
    pub value: f64,
}

/// Element results for post-processing.
#[derive(Debug, Clone, Serialize)]
pub struct ElementResult {
    pub element_id: usize,
    pub node_ids: [usize; 2],
    pub axial_force: f64,
    pub axial_stress: f64,
    pub axial_strain: f64,
    pub elongation: f64,
}

/// Nodal displacement result.
#[derive(Debug, Clone, Serialize)]
pub struct NodalDisplacement {
    pub node: usize,
    pub ux: f64,
    pub uy: f64,
    pub uz: f64,
    pub magnitude: f64,
}

/// Convergence history for iterative solvers.
#[derive(Debug, Clone, Default)]
pub struct ConvergenceHistory {
    pub iterations: Vec<usize>,
    pub residual_norms: Vec<f64>,
    pub converged: bool,
}

impl ConvergenceHistory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, iteration: usize, residual: f64) {
        self.iterations.push(iteration);
        self.residual_norms.push(residual);
    }

    /// Returns convergence rate (average reduction per iteration).
    pub fn convergence_rate(&self) -> Option<f64> {
        if self.residual_norms.len() < 2 {
            return None;
        }
        let r0 = self.residual_norms[0];
        let r_final = *self.residual_norms.last().unwrap();
        if r0 > 0.0 && r_final > 0.0 {
            let n = self.residual_norms.len() as f64;
            Some((r0 / r_final).powf(1.0 / n))
        } else {
            None
        }
    }

    /// Writes convergence data to CSV.
    pub fn write_csv<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);

        writeln!(w, "iteration,residual_norm,log10_residual")?;
        for (i, &r) in self.residual_norms.iter().enumerate() {
            let log_r = if r > 0.0 { r.log10() } else { f64::NEG_INFINITY };
            writeln!(w, "{},{},{}", i, r, log_r)?;
        }

        w.flush()
    }

    /// Exports convergence plot as SVG.
    pub fn write_svg<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        if self.residual_norms.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "No convergence data to plot"
            ));
        }

        let width: f64 = 600.0;
        let height: f64 = 400.0;
        let margin: f64 = 50.0;
        let plot_width: f64 = width - 2.0 * margin;
        let plot_height: f64 = height - 2.0 * margin;

        let file = File::create(path)?;
        let mut w = BufWriter::new(file);

        // SVG header
        writeln!(w, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")?;
        writeln!(w, "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\">", width, height)?;
        writeln!(w, "  <rect width=\"100%\" height=\"100%\" fill=\"#0b0f14\"/>")?;

        // Compute scales
        let n = self.residual_norms.len();
        let x_max = (n as f64) - 1.0;
        let log_residuals: Vec<f64> = self.residual_norms.iter()
            .map(|&r| if r > 0.0 { r.log10() } else { -15.0 })
            .collect();
        let y_min = log_residuals.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = log_residuals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_range = (y_max - y_min).max(1.0);

        // Grid lines
        writeln!(w, "  <g stroke=\"#334155\" stroke-width=\"1\" opacity=\"0.5\">")?;
        for i in 0..5 {
            let y = margin + (i as f64 / 4.0) * plot_height;
            writeln!(w, "    <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\"/>",
                margin, y, width - margin, y)?;
        }
        writeln!(w, "  </g>")?;

        // Convergence line
        let mut path_d = String::new();
        for (i, &log_r) in log_residuals.iter().enumerate() {
            let x = margin + (i as f64 / x_max.max(1.0)) * plot_width;
            let y = margin + plot_height - ((log_r - y_min) / y_range) * plot_height;
            if i == 0 {
                path_d.push_str(&format!("M {},{}", x, y));
            } else {
                path_d.push_str(&format!(" L {},{}", x, y));
            }
        }
        writeln!(w, "  <path d=\"{}\" fill=\"none\" stroke=\"#4ade80\" stroke-width=\"2\"/>", path_d)?;

        // Axis labels
        writeln!(w, "  <text x=\"{}\" y=\"{}\" fill=\"#e6edf3\" font-size=\"14\" text-anchor=\"middle\">Iteration</text>",
            width / 2.0, height - 10.0)?;
        writeln!(w, "  <text x=\"20\" y=\"{}\" fill=\"#e6edf3\" font-size=\"14\" text-anchor=\"middle\" transform=\"rotate(-90, 20, {})\">log10(Residual)</text>",
            height / 2.0, height / 2.0)?;

        // Title
        let status = if self.converged { "Converged" } else { "Not converged" };
        writeln!(w, "  <text x=\"{}\" y=\"30\" fill=\"#4ade80\" font-size=\"16\" text-anchor=\"middle\">Convergence History ({})</text>",
            width / 2.0, status)?;

        writeln!(w, "</svg>")?;
        w.flush()
    }
}

/// Computes reaction forces from the solution.
pub fn compute_reactions(
    model: &Model<Truss2>,
    k: &nalgebra::DMatrix<f64>,
    u: &[f64],
    f: &[f64],
) -> Vec<ReactionForce> {
    let u_vec = nalgebra::DVector::from_column_slice(u);
    let f_vec = nalgebra::DVector::from_column_slice(f);
    let r = k * u_vec - f_vec;

    let mut reactions = Vec::new();

    for bc in &model.bcs {
        if let Some(dof_idx) = model.dof_index(bc.node, bc.dof) {
            if dof_idx < r.len() {
                reactions.push(ReactionForce {
                    node: bc.node,
                    dof: format!("{:?}", bc.dof),
                    value: r[dof_idx],
                });
            }
        }
    }

    reactions
}

/// Extracts element-level results.
pub fn extract_element_results(model: &Model<Truss2>, u: &[f64]) -> Vec<ElementResult> {
    let mut results = Vec::with_capacity(model.elements.len());

    for (elem_id, elem) in model.elements.iter().enumerate() {
        let (length, _dir) = elem.length_and_dir(model);
        let stress = elem.axial_stress(model, u);
        let strain = stress / elem.e;
        let force = stress * elem.a;
        let elongation = strain * length;

        results.push(ElementResult {
            element_id: elem_id,
            node_ids: [elem.n1, elem.n2],
            axial_force: force,
            axial_stress: stress,
            axial_strain: strain,
            elongation,
        });
    }

    results
}

/// Extracts nodal displacements.
pub fn extract_nodal_displacements(model: &Model<Truss2>, u: &[f64]) -> Vec<NodalDisplacement> {
    let mut results = Vec::with_capacity(model.nodes.len());

    for nid in 0..model.nodes.len() {
        let ux = model
            .dof_index(nid, Dof::Ux)
            .and_then(|i| u.get(i))
            .copied()
            .unwrap_or(0.0);
        let uy = model
            .dof_index(nid, Dof::Uy)
            .and_then(|i| u.get(i))
            .copied()
            .unwrap_or(0.0);
        let uz = model
            .dof_index(nid, Dof::Uz)
            .and_then(|i| u.get(i))
            .copied()
            .unwrap_or(0.0);

        let magnitude = (ux * ux + uy * uy + uz * uz).sqrt();

        results.push(NodalDisplacement {
            node: nid,
            ux,
            uy,
            uz,
            magnitude,
        });
    }

    results
}

/// CSV writer for results.
#[derive(Debug, Default)]
pub struct CsvWriter;

impl CsvWriter {
    pub fn new() -> Self {
        Self
    }

    /// Writes nodal displacements to CSV.
    pub fn write_displacements<P: AsRef<Path>>(
        &self,
        path: P,
        displacements: &[NodalDisplacement],
    ) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);

        writeln!(w, "node,ux,uy,uz,magnitude")?;
        for d in displacements {
            writeln!(w, "{},{},{},{},{}", d.node, d.ux, d.uy, d.uz, d.magnitude)?;
        }

        w.flush()
    }

    /// Writes element results to CSV.
    pub fn write_element_results<P: AsRef<Path>>(
        &self,
        path: P,
        results: &[ElementResult],
    ) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);

        writeln!(w, "element_id,node1,node2,axial_force,axial_stress,axial_strain,elongation")?;
        for r in results {
            writeln!(
                w,
                "{},{},{},{},{},{},{}",
                r.element_id,
                r.node_ids[0],
                r.node_ids[1],
                r.axial_force,
                r.axial_stress,
                r.axial_strain,
                r.elongation
            )?;
        }

        w.flush()
    }

    /// Writes reaction forces to CSV.
    pub fn write_reactions<P: AsRef<Path>>(
        &self,
        path: P,
        reactions: &[ReactionForce],
    ) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);

        writeln!(w, "node,dof,reaction_force")?;
        for r in reactions {
            writeln!(w, "{},{},{}", r.node, r.dof, r.value)?;
        }

        w.flush()
    }
}

/// Statistics for result analysis.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ResultStatistics {
    pub max_displacement: f64,
    pub max_displacement_node: usize,
    pub max_stress: f64,
    pub max_stress_element: usize,
    pub min_stress: f64,
    pub min_stress_element: usize,
    pub max_axial_force: f64,
    pub max_axial_force_element: usize,
}

impl ResultStatistics {
    /// Computes statistics from results.
    pub fn from_results(
        displacements: &[NodalDisplacement],
        element_results: &[ElementResult],
    ) -> Self {
        let mut stats = Self::default();

        // Max displacement
        for (i, d) in displacements.iter().enumerate() {
            if d.magnitude > stats.max_displacement {
                stats.max_displacement = d.magnitude;
                stats.max_displacement_node = i;
            }
        }

        // Stress and force statistics
        for (i, r) in element_results.iter().enumerate() {
            if r.axial_stress.abs() > stats.max_stress.abs() {
                stats.max_stress = r.axial_stress;
                stats.max_stress_element = i;
            }
            if r.axial_stress < stats.min_stress {
                stats.min_stress = r.axial_stress;
                stats.min_stress_element = i;
            }
            if r.axial_force.abs() > stats.max_axial_force.abs() {
                stats.max_axial_force = r.axial_force;
                stats.max_axial_force_element = i;
            }
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BoundaryCondition, Load, Node};
    use crate::solver::LinearStaticSolver;

    #[test]
    fn test_extract_nodal_displacements() -> anyhow::Result<()> {
        let mut model = Model::<Truss2>::new();
        let n0 = model.add_node(Node::new_2d(0.0, 0.0));
        let n1 = model.add_node(Node::new_2d(1.0, 0.0));
        model.add_element(Truss2::new(n0, n1, 210e9, 1e-4));

        for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition {
                node: n0,
                dof,
                value: 0.0,
            });
        }
        for dof in [Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition {
                node: n1,
                dof,
                value: 0.0,
            });
        }
        model.add_load(Load {
            node: n1,
            dof: Dof::Ux,
            value: 1000.0,
        });

        let solver = LinearStaticSolver::new();
        let result = solver.solve_truss2(&mut model)?;

        let displacements = extract_nodal_displacements(&model, &result.u);

        assert_eq!(displacements.len(), 2);
        assert!(displacements[0].magnitude < 1e-10); // Fixed node
        assert!(displacements[1].magnitude > 0.0); // Loaded node

        Ok(())
    }

    #[test]
    fn test_extract_element_results() -> anyhow::Result<()> {
        let mut model = Model::<Truss2>::new();
        let n0 = model.add_node(Node::new_2d(0.0, 0.0));
        let n1 = model.add_node(Node::new_2d(1.0, 0.0));
        model.add_element(Truss2::new(n0, n1, 210e9, 1e-4));

        for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition {
                node: n0,
                dof,
                value: 0.0,
            });
        }
        for dof in [Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition {
                node: n1,
                dof,
                value: 0.0,
            });
        }
        model.add_load(Load {
            node: n1,
            dof: Dof::Ux,
            value: 1000.0,
        });

        let solver = LinearStaticSolver::new();
        let result = solver.solve_truss2(&mut model)?;

        let elem_results = extract_element_results(&model, &result.u);

        assert_eq!(elem_results.len(), 1);
        assert!(elem_results[0].axial_force > 0.0); // Tension
        assert!(elem_results[0].axial_stress > 0.0);

        Ok(())
    }

    #[test]
    fn test_result_statistics() {
        let displacements = vec![
            NodalDisplacement {
                node: 0,
                ux: 0.0,
                uy: 0.0,
                uz: 0.0,
                magnitude: 0.0,
            },
            NodalDisplacement {
                node: 1,
                ux: 0.001,
                uy: 0.002,
                uz: 0.0,
                magnitude: 0.002236,
            },
        ];

        let element_results = vec![
            ElementResult {
                element_id: 0,
                node_ids: [0, 1],
                axial_force: 1000.0,
                axial_stress: 1e7,
                axial_strain: 5e-5,
                elongation: 0.0001,
            },
        ];

        let stats = ResultStatistics::from_results(&displacements, &element_results);

        assert!((stats.max_displacement - 0.002236).abs() < 1e-6);
        assert_eq!(stats.max_displacement_node, 1);
        assert!((stats.max_stress - 1e7).abs() < 1e3);
        assert_eq!(stats.max_stress_element, 0);
    }

    #[test]
    fn test_convergence_history() {
        let mut history = ConvergenceHistory::new();
        history.record(1, 1.0);
        history.record(2, 0.1);
        history.record(3, 0.01);
        history.record(4, 0.001);

        assert_eq!(history.iterations.len(), 4);
        assert_eq!(history.residual_norms.len(), 4);

        // Convergence rate should be around 10 (each iteration reduces by 10x)
        // Allow for numerical precision issues
        let rate = history.convergence_rate();
        assert!(rate.is_some());
        let rate = rate.unwrap();
        assert!(rate > 5.0 && rate < 15.0, "Convergence rate should be around 10, got {}", rate);
    }

    #[test]
    fn test_csv_writer_displacements() -> anyhow::Result<()> {
        let displacements = vec![
            NodalDisplacement {
                node: 0,
                ux: 0.0,
                uy: 0.0,
                uz: 0.0,
                magnitude: 0.0,
            },
            NodalDisplacement {
                node: 1,
                ux: 0.001,
                uy: 0.0,
                uz: 0.0,
                magnitude: 0.001,
            },
        ];

        let path = "/tmp/test_disp.csv";
        let writer = CsvWriter::new();
        writer.write_displacements(path, &displacements)?;

        let content = std::fs::read_to_string(path)?;
        assert!(content.contains("node,ux,uy,uz,magnitude"));
        assert!(content.contains("0,0,0,0,0"));
        assert!(content.contains("1,0.001,0,0,0.001"));

        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn test_csv_writer_convergence() -> anyhow::Result<()> {
        let mut history = ConvergenceHistory::new();
        history.record(0, 1.0);
        history.record(1, 0.1);
        history.record(2, 0.01);

        let path = "/tmp/test_conv.csv";
        history.write_csv(path)?;

        let content = std::fs::read_to_string(path)?;
        assert!(content.contains("iteration,residual_norm,log10_residual"));
        assert!(content.contains("0,1,0"));
        assert!(content.contains("1,0.1,-1"));

        std::fs::remove_file(path)?;
        Ok(())
    }
}

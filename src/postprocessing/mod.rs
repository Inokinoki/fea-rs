//! Post-processing module for FEA results.
#![allow(unused_variables)]
//!
//! This module provides:
//! - Result visualization utilities
//! - Stress/strain recovery
//! - Output file generation (VTK, CSV, JSON)
//! - Report generation
//! - Result statistics and analysis

use crate::core::{Node, Load, BoundaryCondition};


use std::fs::File;
use std::io::Write;
use std::path::Path;

/// FEA analysis results structure.
#[derive(Debug, Clone)]
pub struct FeaResults {
    /// Nodal displacements.
    pub displacements: Vec<f64>,
    /// Nodal reaction forces.
    pub reactions: Vec<f64>,
    /// Element stresses (if available).
    pub element_stresses: Option<Vec<f64>>,
    /// Element strains (if available).
    pub element_strains: Option<Vec<f64>>,
    /// Number of DOFs per node.
    pub dof_per_node: usize,
}

impl FeaResults {
    /// Creates new FEA results.
    pub fn new(displacements: Vec<f64>, reactions: Vec<f64>, dof_per_node: usize) -> Self {
        Self {
            displacements,
            reactions,
            element_stresses: None,
            element_strains: None,
            dof_per_node,
        }
    }

    /// Returns displacement at a specific node.
    pub fn node_displacement(&self, node_id: usize) -> Option<Vec<f64>> {
        let start = node_id * self.dof_per_node;
        let end = start + self.dof_per_node;
        if end <= self.displacements.len() {
            Some(self.displacements[start..end].to_vec())
        } else {
            None
        }
    }

    /// Returns displacement magnitude at a node.
    pub fn displacement_magnitude(&self, node_id: usize) -> Option<f64> {
        self.node_displacement(node_id).map(|disp| {
            disp.iter().map(|d| d * d).sum::<f64>().sqrt()
        })
    }

    /// Returns maximum displacement magnitude.
    pub fn max_displacement_magnitude(&self) -> f64 {
        let mut max_disp: f64 = 0.0;
        for node_id in 0..(self.displacements.len() / self.dof_per_node) {
            if let Some(mag) = self.displacement_magnitude(node_id) {
                max_disp = max_disp.max(mag);
            }
        }
        max_disp
    }

    /// Returns node with maximum displacement.
    pub fn max_displacement_node(&self) -> Option<usize> {
        let mut max_node: Option<usize> = None;
        let mut max_mag: f64 = 0.0;

        for node_id in 0..(self.displacements.len() / self.dof_per_node) {
            if let Some(mag) = self.displacement_magnitude(node_id) {
                if mag > max_mag {
                    max_mag = mag;
                    max_node = Some(node_id);
                }
            }
        }

        max_node
    }

    /// Returns reaction force at a specific node.
    pub fn node_reaction(&self, node_id: usize) -> Option<Vec<f64>> {
        let start = node_id * self.dof_per_node;
        let end = start + self.dof_per_node;
        if end <= self.reactions.len() {
            Some(self.reactions[start..end].to_vec())
        } else {
            None
        }
    }

    /// Returns total reaction force magnitude.
    pub fn total_reaction_magnitude(&self) -> f64 {
        self.reactions.iter().map(|r| r * r).sum::<f64>().sqrt()
    }
}

/// Stress result at a point.
#[derive(Debug, Clone)]
pub struct StressResult {
    pub sigma_x: f64,
    pub sigma_y: f64,
    pub sigma_z: f64,
    pub tau_xy: f64,
    pub tau_yz: f64,
    pub tau_xz: f64,
}

impl StressResult {
    /// Creates a new stress result.
    pub fn new(sigma_x: f64, sigma_y: f64, sigma_z: f64) -> Self {
        Self {
            sigma_x,
            sigma_y,
            sigma_z,
            tau_xy: 0.0,
            tau_yz: 0.0,
            tau_xz: 0.0,
        }
    }

    /// Creates a full 3D stress result.
    pub fn new_3d(
        sigma_x: f64,
        sigma_y: f64,
        sigma_z: f64,
        tau_xy: f64,
        tau_yz: f64,
        tau_xz: f64,
    ) -> Self {
        Self {
            sigma_x,
            sigma_y,
            sigma_z,
            tau_xy,
            tau_yz,
            tau_xz,
        }
    }

    /// Computes von Mises equivalent stress.
    pub fn von_mises(&self) -> f64 {
        let s1 = self.sigma_x - self.sigma_y;
        let s2 = self.sigma_y - self.sigma_z;
        let s3 = self.sigma_z - self.sigma_x;
        let t1 = self.tau_xy;
        let t2 = self.tau_yz;
        let t3 = self.tau_xz;

        ((s1 * s1 + s2 * s2 + s3 * s3 + 6.0 * (t1 * t1 + t2 * t2 + t3 * t3)) / 2.0).sqrt()
    }

    /// Computes principal stresses.
    pub fn principal_stresses(&self) -> (f64, f64, f64) {
        // For 2D plane stress
        let avg = (self.sigma_x + self.sigma_y) / 2.0;
        let radius = (((self.sigma_x - self.sigma_y) / 2.0).powi(2) + self.tau_xy.powi(2)).sqrt();

        let sigma_1 = avg + radius;
        let sigma_2 = avg - radius;
        let sigma_3 = self.sigma_z;

        // Sort
        let mut stresses = [sigma_1, sigma_2, sigma_3];
        stresses.sort_by(|a, b| b.partial_cmp(a).unwrap());

        (stresses[0], stresses[1], stresses[2])
    }

    /// Computes maximum shear stress.
    pub fn max_shear_stress(&self) -> f64 {
        let (s1, _, s3) = self.principal_stresses();
        (s1 - s3) / 2.0
    }
}

/// Strain result at a point.
#[derive(Debug, Clone)]
pub struct StrainResult {
    pub epsilon_x: f64,
    pub epsilon_y: f64,
    pub epsilon_z: f64,
    pub gamma_xy: f64,
    pub gamma_yz: f64,
    pub gamma_xz: f64,
}

impl StrainResult {
    /// Creates a new strain result.
    pub fn new(epsilon_x: f64, epsilon_y: f64, epsilon_z: f64) -> Self {
        Self {
            epsilon_x,
            epsilon_y,
            epsilon_z,
            gamma_xy: 0.0,
            gamma_yz: 0.0,
            gamma_xz: 0.0,
        }
    }

    /// Computes von Mises equivalent strain.
    pub fn von_mises(&self) -> f64 {
        let sqrt2_3: f64 = (2.0_f64 / 3.0_f64).sqrt();
        let e1 = self.epsilon_x - self.epsilon_y;
        let e2 = self.epsilon_y - self.epsilon_z;
        let e3 = self.epsilon_z - self.epsilon_x;
        let g1 = self.gamma_xy;
        let g2 = self.gamma_yz;
        let g3 = self.gamma_xz;

        sqrt2_3 * ((e1 * e1 + e2 * e2 + e3 * e3 + 0.5 * (g1 * g1 + g2 * g2 + g3 * g3)).sqrt())
    }
}

/// VTK file export for results.
pub mod vtk_export {
    use super::*;

    /// Exports displacement results to VTK format.
    pub fn export_displacements<P: AsRef<Path>>(
        path: P,
        nodes: &[Node],
        results: &FeaResults,
        scale_factor: f64,
    ) -> anyhow::Result<()> {
        let mut file = File::create(path)?;

        writeln!(file, "# vtk DataFile Version 3.0")?;
        writeln!(file, "FEA Displacement Results")?;
        writeln!(file, "ASCII")?;
        writeln!(file, "DATASET UNSTRUCTURED_GRID")?;
        writeln!(file)?;

        // Points (deformed configuration)
        writeln!(file, "POINTS {} float", nodes.len())?;
        for (i, node) in nodes.iter().enumerate() {
            let orig = get_node_coords(node);
            let disp = results.node_displacement(i).unwrap_or(vec![0.0; 3]);

            writeln!(
                file,
                "{} {} {}",
                orig[0] + disp[0] * scale_factor,
                orig[1] + disp[1] * scale_factor,
                orig[2] + disp.get(2).copied().unwrap_or(0.0) * scale_factor
            )?;
        }
        writeln!(file)?;

        // Displacement vectors
        writeln!(file, "POINT_DATA {}", nodes.len())?;
        writeln!(file, "VECTORS displacements float")?;
        for i in 0..nodes.len() {
            let disp = results.node_displacement(i).unwrap_or(vec![0.0; 3]);
            writeln!(file, "{} {} {}", disp[0], disp[1], disp.get(2).copied().unwrap_or(0.0))?;
        }

        // Displacement magnitude
        writeln!(file, "SCALARS displacement_magnitude float")?;
        writeln!(file, "LOOKUP_TABLE default")?;
        for i in 0..nodes.len() {
            let mag = results.displacement_magnitude(i).unwrap_or(0.0);
            writeln!(file, "{}", mag)?;
        }

        Ok(())
    }

    /// Exports stress results to VTK format.
    pub fn export_stresses<P: AsRef<Path>>(
        path: P,
        nodes: &[Node],
        stresses: &[StressResult],
    ) -> anyhow::Result<()> {
        let mut file = File::create(path)?;

        writeln!(file, "# vtk DataFile Version 3.0")?;
        writeln!(file, "FEA Stress Results")?;
        writeln!(file, "ASCII")?;
        writeln!(file, "DATASET UNSTRUCTURED_GRID")?;
        writeln!(file)?;

        // Points
        writeln!(file, "POINTS {} float", nodes.len())?;
        for node in nodes {
            let coords = get_node_coords(node);
            writeln!(file, "{} {} {}", coords[0], coords[1], coords[2])?;
        }
        writeln!(file)?;

        // Von Mises stress
        writeln!(file, "POINT_DATA {}", nodes.len())?;
        writeln!(file, "SCALARS von_mises_stress float")?;
        writeln!(file, "LOOKUP_TABLE default")?;
        for stress in stresses {
            writeln!(file, "{}", stress.von_mises())?;
        }

        // Stress tensor components
        writeln!(file, "TENSORS stress_tensor float")?;
        for stress in stresses {
            writeln!(
                file,
                "{} {} {} {} {} {} {} {} {}",
                stress.sigma_x, stress.tau_xy, stress.tau_xz,
                stress.tau_xy, stress.sigma_y, stress.tau_yz,
                stress.tau_xz, stress.tau_yz, stress.sigma_z
            )?;
        }

        Ok(())
    }

    /// Exports reaction forces to VTK format.
    pub fn export_reactions<P: AsRef<Path>>(
        path: P,
        nodes: &[Node],
        results: &FeaResults,
    ) -> anyhow::Result<()> {
        let mut file = File::create(path)?;

        writeln!(file, "# vtk DataFile Version 3.0")?;
        writeln!(file, "FEA Reaction Forces")?;
        writeln!(file, "ASCII")?;
        writeln!(file, "DATASET UNSTRUCTURED_GRID")?;
        writeln!(file)?;

        // Points
        writeln!(file, "POINTS {} float", nodes.len())?;
        for node in nodes {
            let coords = get_node_coords(node);
            writeln!(file, "{} {} {}", coords[0], coords[1], coords[2])?;
        }
        writeln!(file)?;

        // Reaction forces
        writeln!(file, "POINT_DATA {}", nodes.len())?;
        writeln!(file, "VECTORS reactions float")?;
        for i in 0..nodes.len() {
            let reac = results.node_reaction(i).unwrap_or(vec![0.0; 3]);
            writeln!(file, "{} {} {}", reac[0], reac[1], reac.get(2).copied().unwrap_or(0.0))?;
        }

        Ok(())
    }

    fn get_node_coords(node: &Node) -> [f64; 3] {
        [node.x, node.y, node.z]
    }
}

/// CSV file export.
pub mod csv_export {
    use super::*;

    /// Exports nodal displacements to CSV.
    pub fn export_displacements_csv<P: AsRef<Path>>(
        path: P,
        nodes: &[Node],
        results: &FeaResults,
    ) -> anyhow::Result<()> {
        let mut file = File::create(path)?;

        // Header
        writeln!(file, "NodeID,X,Y,Z,Ux,Uy,Uz,Magnitude")?;

        for (i, node) in nodes.iter().enumerate() {
            let coords = get_node_coords(node);
            let disp = results.node_displacement(i).unwrap_or(vec![0.0; 3]);
            let mag = results.displacement_magnitude(i).unwrap_or(0.0);

            writeln!(
                file,
                "{},{},{},{},{},{},{},{}",
                i, coords[0], coords[1], coords[2],
                disp[0], disp[1], disp.get(2).copied().unwrap_or(0.0),
                mag
            )?;
        }

        Ok(())
    }

    /// Exports reaction forces to CSV.
    pub fn export_reactions_csv<P: AsRef<Path>>(
        path: P,
        nodes: &[Node],
        results: &FeaResults,
    ) -> anyhow::Result<()> {
        let mut file = File::create(path)?;

        // Header
        writeln!(file, "NodeID,NodeX,NodeY,NodeZ,Rx,Ry,Rz,Magnitude")?;

        for (i, node) in nodes.iter().enumerate() {
            let coords = get_node_coords(node);
            let reac = results.node_reaction(i).unwrap_or(vec![0.0; 3]);
            let mag = (reac[0].powi(2) + reac[1].powi(2) + reac[2].powi(2)).sqrt();

            writeln!(
                file,
                "{},{},{},{},{},{},{},{}",
                i, coords[0], coords[1], coords[2],
                reac[0], reac[1], reac.get(2).copied().unwrap_or(0.0),
                mag
            )?;
        }

        Ok(())
    }

    /// Exports stress results to CSV.
    pub fn export_stresses_csv<P: AsRef<Path>>(
        path: P,
        node_ids: &[usize],
        stresses: &[StressResult],
    ) -> anyhow::Result<()> {
        let mut file = File::create(path)?;

        // Header
        writeln!(
            file,
            "NodeID,SigmaX,SigmaY,SigmaZ,TauXY,TauYZ,TauXZ,VonMises,MaxShear"
        )?;

        for (i, &node_id) in node_ids.iter().enumerate() {
            if i < stresses.len() {
                let s = &stresses[i];
                writeln!(
                    file,
                    "{},{},{},{},{},{},{},{},{}",
                    node_id, s.sigma_x, s.sigma_y, s.sigma_z,
                    s.tau_xy, s.tau_yz, s.tau_xz,
                    s.von_mises(), s.max_shear_stress()
                )?;
            }
        }

        Ok(())
    }

    fn get_node_coords(node: &Node) -> [f64; 3] {
        [node.x, node.y, node.z]
    }
}

/// Report generation.
pub mod report_generation {
    use super::*;
    use std::fmt::Write;
    use std::io::Write as IoWrite;

    /// Generates a text report of FEA results.
    pub fn generate_text_report(
        nodes: &[Node],
        elements: usize,
        bcs: &[BoundaryCondition],
        loads: &[Load],
        results: &FeaResults,
    ) -> String {
        let mut report = String::new();

        writeln!(&mut report, "╔═══════════════════════════════════════════════════════════╗").unwrap();
        writeln!(&mut report, "║              FEA Analysis Report                          ║").unwrap();
        writeln!(&mut report, "╚═══════════════════════════════════════════════════════════╝").unwrap();
        writeln!(&mut report).unwrap();

        // Model summary
        writeln!(&mut report, "MODEL SUMMARY").unwrap();
        writeln!(&mut report, "─────────────────────────────────────────").unwrap();
        writeln!(&mut report, "  Nodes:       {}", nodes.len()).unwrap();
        writeln!(&mut report, "  Elements:    {}", elements).unwrap();
        writeln!(&mut report, "  BCs:         {}", bcs.len()).unwrap();
        writeln!(&mut report, "  Loads:       {}", loads.len()).unwrap();
        writeln!(&mut report).unwrap();

        // Displacement summary
        writeln!(&mut report, "DISPLACEMENT SUMMARY").unwrap();
        writeln!(&mut report, "─────────────────────────────────────────").unwrap();

        let max_disp = results.max_displacement_magnitude();
        if let Some(max_node) = results.max_displacement_node() {
            writeln!(&mut report, "  Max displacement:    {:.6e} m", max_disp).unwrap();
            writeln!(&mut report, "  At node:             {}", max_node).unwrap();

            if let Some(disp) = results.node_displacement(max_node) {
                writeln!(&mut report, "  Components:").unwrap();
                for (i, &d) in disp.iter().enumerate() {
                    let dof_name = match i {
                        0 => "Ux",
                        1 => "Uy",
                        2 => "Uz",
                        _ => "Unknown",
                    };
                    writeln!(&mut report, "    {}: {:.6e} m", dof_name, d).unwrap();
                }
            }
        }
        writeln!(&mut report).unwrap();

        // Reaction summary
        writeln!(&mut report, "REACTION SUMMARY").unwrap();
        writeln!(&mut report, "─────────────────────────────────────────").unwrap();
        writeln!(&mut report, "  Total reaction:      {:.6e} N", results.total_reaction_magnitude()).unwrap();
        writeln!(&mut report).unwrap();

        // Equilibrium check
        writeln!(&mut report, "EQUILIBRIUM CHECK").unwrap();
        writeln!(&mut report, "─────────────────────────────────────────").unwrap();

        let total_load: f64 = loads.iter().map(|l| l.value.abs()).sum();
        let total_reaction = results.total_reaction_magnitude();
        let error = if total_load > 0.0 {
            (total_reaction - total_load).abs() / total_load * 100.0
        } else {
            0.0
        };

        writeln!(&mut report, "  Total applied load:  {:.6e} N", total_load).unwrap();
        writeln!(&mut report, "  Total reaction:      {:.6e} N", total_reaction).unwrap();
        writeln!(&mut report, "  Equilibrium error:   {:.2}%", error).unwrap();

        let status = if error < 1.0 { "✓ PASS" } else { "⚠ WARNING" };
        writeln!(&mut report, "  Status:              {}", status).unwrap();

        report
    }

    /// Generates an HTML report with interactive visualization.
    pub fn generate_html_report<P: AsRef<Path>>(
        path: P,
        nodes: &[Node],
        results: &FeaResults,
        title: &str,
    ) -> anyhow::Result<()> {
        let mut file = File::create(path)?;

        writeln!(file, "<!DOCTYPE html>")?;
        writeln!(file, "<html><head>")?;
        writeln!(file, "<title>{}</title>", title)?;
        writeln!(file, "<style>")?;
        writeln!(file, "body {{ font-family: Arial, sans-serif; margin: 20px; }}")?;
        writeln!(file, "h1 {{ color: #333; }}")?;
        writeln!(file, "table {{ border-collapse: collapse; width: 100%; }}")?;
        writeln!(file, "th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}")?;
        writeln!(file, "th {{ background-color: #4CAF50; color: white; }}")?;
        writeln!(file, "tr:nth-child(even) {{ background-color: #f2f2f2; }}")?;
        writeln!(file, ".summary {{ background-color: #e7f3fe; padding: 15px; margin: 10px 0; }}")?;
        writeln!(file, "</style>")?;
        writeln!(file, "</head><body>")?;

        writeln!(file, "<h1>FEA Analysis Report: {}</h1>", title)?;

        // Summary
        writeln!(file, "<div class=\"summary\">")?;
        writeln!(file, "<h2>Summary</h2>")?;
        writeln!(file, "<p><strong>Nodes:</strong> {}</p>", nodes.len())?;
        writeln!(file, "<p><strong>Max Displacement:</strong> {:.6e} m</p>", results.max_displacement_magnitude())?;
        if let Some(max_node) = results.max_displacement_node() {
            writeln!(file, "<p><strong>Max at Node:</strong> {}</p>", max_node)?;
        }
        writeln!(file, "</div>")?;

        // Displacement table
        writeln!(file, "<h2>Nodal Displacements</h2>")?;
        writeln!(file, "<table>")?;
        writeln!(file, "<tr><th>Node</th><th>Ux (m)</th><th>Uy (m)</th><th>Uz (m)</th><th>Magnitude (m)</th></tr>")?;

        for (i, node) in nodes.iter().enumerate() {
            let disp = results.node_displacement(i).unwrap_or(vec![0.0; 3]);
            let mag = results.displacement_magnitude(i).unwrap_or(0.0);

            writeln!(
                file,
                "<tr><td>{}</td><td>{:.6e}</td><td>{:.6e}</td><td>{:.6e}</td><td>{:.6e}</td></tr>",
                i, disp[0], disp[1], disp.get(2).copied().unwrap_or(0.0), mag
            )?;
        }

        writeln!(file, "</table>")?;
        writeln!(file, "</body></html>")?;

        Ok(())
    }
}

/// Result visualization utilities.
pub mod visualization {
    /// Creates a color map for scalar visualization.
    pub fn create_color_map(values: &[f64], colormap: &str) -> Vec<[u8; 3]> {
        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;

        values
            .iter()
            .map(|&v| {
                let t = if range > 0.0 { (v - min_val) / range } else { 0.5 };
                colormap_value(t, colormap)
            })
            .collect()
    }

    fn colormap_value(t: f64, colormap: &str) -> [u8; 3] {
        match colormap {
            "jet" => jet_colormap(t),
            "rainbow" => rainbow_colormap(t),
            "hot" => hot_colormap(t),
            "coolwarm" => coolwarm_colormap(t),
            _ => jet_colormap(t),
        }
    }

    fn jet_colormap(t: f64) -> [u8; 3] {
        let r = (255.0 * (4.0 * t - 1.5).min(1.0).max(0.0)) as u8;
        let g = (255.0 * (2.0 - (4.0 * (t - 0.5)).abs()).min(1.0).max(0.0)) as u8;
        let b = (255.0 * (1.5 - 4.0 * (1.0 - t)).min(1.0).max(0.0)) as u8;
        [r, g, b]
    }

    fn rainbow_colormap(t: f64) -> [u8; 3] {
        let r = (255.0 * (0.5 + 0.5 * (2.0 * std::f64::consts::PI * t).sin())) as u8;
        let g = (255.0 * (0.5 + 0.5 * (2.0 * std::f64::consts::PI * (t - 0.33)).sin())) as u8;
        let b = (255.0 * (0.5 + 0.5 * (2.0 * std::f64::consts::PI * (t - 0.67)).sin())) as u8;
        [r, g, b]
    }

    fn hot_colormap(t: f64) -> [u8; 3] {
        let r = (255.0 * t.min(1.0).max(0.0)) as u8;
        let g = (255.0 * (t - 0.33).min(1.0).max(0.0)) as u8;
        let b = (255.0 * (t - 0.67).min(1.0).max(0.0)) as u8;
        [r, g, b]
    }

    fn coolwarm_colormap(t: f64) -> [u8; 3] {
        let r = (255.0 * (1.0 - t).min(1.0).max(0.0)) as u8;
        let g = (255.0 * 0.7) as u8;
        let b = (255.0 * t.min(1.0).max(0.0)) as u8;
        [r, g, b]
    }
}

/// Advanced post-processing utilities.
pub mod advanced;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fea_results_creation() {
        let displacements = vec![0.001, 0.002, 0.003, 0.001, 0.002, 0.003];
        let reactions = vec![100.0, 200.0, 0.0, 100.0, 200.0, 0.0];
        let results = FeaResults::new(displacements, reactions, 3);

        assert_eq!(results.dof_per_node, 3);
        assert!(results.node_displacement(0).is_some());
    }

    #[test]
    fn test_stress_von_mises() {
        let stress = StressResult::new_3d(100.0, 50.0, 25.0, 10.0, 5.0, 8.0);
        let vm = stress.von_mises();
        assert!(vm > 0.0);
    }

    #[test]
    fn test_stress_principal() {
        let stress = StressResult::new(100.0, 50.0, 0.0);
        let (s1, s2, s3) = stress.principal_stresses();
        assert!(s1 >= s2);
        assert!(s2 >= s3);
    }

    #[test]
    fn test_max_displacement() {
        let displacements = vec![0.0, 0.0, 0.0, 0.001, 0.002, 0.0, 0.0, 0.0, 0.0];
        let reactions = vec![0.0; 9];
        let results = FeaResults::new(displacements, reactions, 3);

        assert!(results.max_displacement_node().is_some());
        assert!(results.max_displacement_magnitude() > 0.0);
    }
}

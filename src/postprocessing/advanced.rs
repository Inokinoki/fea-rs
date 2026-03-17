//! Advanced Post-Processing for FEA Results.
#![allow(unused_variables)]
//!
//! This module provides:
//! - Contour plot generation
//! - Deformed shape animation export
//! - Stress/strain visualization
//! - Result comparison utilities
//! - Report templates

use crate::core::Node;
use std::fs::File;
use std::io::Write;

/// Contour plot generator for scalar fields.
pub struct ContourPlot {
    pub values: Vec<f64>,
    pub min_value: f64,
    pub max_value: f64,
    pub num_levels: usize,
}

impl ContourPlot {
    /// Creates a new contour plot from values.
    pub fn new(values: Vec<f64>, num_levels: usize) -> Self {
        let min_value = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_value = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        Self {
            values,
            min_value,
            max_value,
            num_levels,
        }
    }

    /// Returns contour level values.
    pub fn levels(&self) -> Vec<f64> {
        let range = self.max_value - self.min_value;
        let step = range / (self.num_levels - 1) as f64;

        (0..self.num_levels)
            .map(|i| self.min_value + i as f64 * step)
            .collect()
    }

    /// Returns color for a value (jet colormap).
    pub fn color_for_value(&self, value: f64) -> [u8; 3] {
        let t = if self.max_value > self.min_value {
            (value - self.min_value) / (self.max_value - self.min_value)
        } else {
            0.5
        };

        Self::jet_colormap(t.clamp(0.0, 1.0))
    }

    fn jet_colormap(t: f64) -> [u8; 3] {
        let r = (255.0 * (4.0 * t - 1.5).min(1.0).max(0.0)) as u8;
        let g = (255.0 * (2.0 - (4.0 * (t - 0.5)).abs()).min(1.0).max(0.0)) as u8;
        let b = (255.0 * (1.5 - 4.0 * (1.0 - t)).min(1.0).max(0.0)) as u8;
        [r, g, b]
    }

    /// Exports contour data to SVG format.
    pub fn export_svg<P: AsRef<std::path::Path>>(
        &self,
        path: P,
        nodes: &[Node],
        elements: &[(usize, usize, usize, usize)],
    ) -> std::io::Result<()> {
        let mut file = File::create(path)?;

        writeln!(file, r#"<?xml version="1.0" encoding="UTF-8"?>"#)?;
        writeln!(file, r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600">"#)?;
        writeln!(file, r#"<rect width="800" height="600" fill="white"/>"#)?;

        // Draw elements with color
        for (n0, n1, n2, n3) in elements {
            let c = self.color_for_value(
                (self.values[*n0] + self.values[*n1] + self.values[*n2] + self.values[*n3]) / 4.0
            );

            writeln!(
                file,
                r#"<polygon points="{},{} {},{} {},{} {},{}" fill="rgb({},{},{})" stroke="black" stroke-width="0.5"/>"#,
                Self::map_x(nodes[*n0].x), Self::map_y(nodes[*n0].y),
                Self::map_x(nodes[*n1].x), Self::map_y(nodes[*n1].y),
                Self::map_x(nodes[*n2].x), Self::map_y(nodes[*n2].y),
                Self::map_x(nodes[*n3].x), Self::map_y(nodes[*n3].y),
                c[0], c[1], c[2]
            )?;
        }

        // Draw colorbar
        Self::draw_colorbar(&mut file, self)?;

        writeln!(file, "</svg>")?;
        Ok(())
    }

    fn map_x(x: f64) -> f64 {
        50.0 + x * 100.0 + 300.0
    }

    fn map_y(y: f64) -> f64 {
        550.0 - y * 100.0
    }

    fn draw_colorbar<W: Write>(file: &mut W, plot: &ContourPlot) -> std::io::Result<()> {
        let levels = plot.levels();
        let bar_width = 30.0;
        let bar_height = 400.0;
        let bar_x = 700.0;
        let bar_y = 100.0;
        let level_height = bar_height / (levels.len() - 1) as f64;

        for (i, &level) in levels.iter().enumerate() {
            let c = plot.color_for_value(level);
            let y = bar_y + i as f64 * level_height;

            writeln!(
                file,
                r#"<rect x="{}" y="{}" width="{}" height="{}" fill="rgb({},{},{})" stroke="gray" stroke-width="0.5"/>"#,
                bar_x, y, bar_width, level_height + 1.0, c[0], c[1], c[2]
            )?;

            if i % 2 == 0 {
                writeln!(
                    file,
                    r#"<text x="{}" y="{}" font-size="10" text-anchor="start">{:.2}</text>"#,
                    bar_x + bar_width + 5.0,
                    y + level_height / 2.0 + 3.0,
                    level
                )?;
            }
        }

        Ok(())
    }
}

/// Animation frame for deformed shape.
#[derive(Debug, Clone)]
pub struct AnimationFrame {
    pub frame_number: usize,
    pub scale_factor: f64,
    pub time: f64,
    pub deformed_nodes: Vec<(f64, f64, f64)>,
}

/// Animation generator for deformed shapes.
pub struct DeformedShapeAnimation {
    pub frames: Vec<AnimationFrame>,
}

impl DeformedShapeAnimation {
    /// Creates a new animation from mode shapes.
    pub fn from_mode_shape(nodes: &[Node], mode_shape: &[f64], num_frames: usize) -> Self {
        let mut frames = Vec::with_capacity(num_frames);

        for frame in 0..num_frames {
            let scale = ((frame as f64 / num_frames as f64) * 2.0 * std::f64::consts::PI).sin();
            let mut deformed = Vec::with_capacity(nodes.len());

            for (i, node) in nodes.iter().enumerate() {
                let dx = if i * 3 < mode_shape.len() { mode_shape[i * 3] * scale * 10.0 } else { 0.0 };
                let dy = if i * 3 + 1 < mode_shape.len() { mode_shape[i * 3 + 1] * scale * 10.0 } else { 0.0 };
                let dz = if i * 3 + 2 < mode_shape.len() { mode_shape[i * 3 + 2] * scale * 10.0 } else { 0.0 };

                deformed.push((node.x + dx, node.y + dy, node.z + dz));
            }

            frames.push(AnimationFrame {
                frame_number: frame,
                scale_factor: scale,
                time: frame as f64 / num_frames as f64,
                deformed_nodes: deformed,
            });
        }

        Self { frames }
    }

    /// Exports animation as SVG frames.
    pub fn export_svg_frames<P: AsRef<std::path::Path>>(
        &self,
        base_path: P,
        elements: &[(usize, usize)],
    ) -> std::io::Result<()> {
        for frame in &self.frames {
            let path = format!(
                "{}_{:03}.svg",
                base_path.as_ref().to_string_lossy(),
                frame.frame_number
            );

            let mut file = File::create(&path)?;

            writeln!(file, r#"<?xml version="1.0" encoding="UTF-8"?>"#)?;
            writeln!(file, r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600">"#)?;
            writeln!(file, r#"<rect width="800" height="600" fill="white"/>"#)?;
            writeln!(file, r#"<text x="400" y="30" font-size="16" text-anchor="middle">Frame {} (t={:.2})</text>"#, frame.frame_number, frame.time)?;

            // Draw deformed elements
            for (n0, n1) in elements {
                let (x0, y0, _) = frame.deformed_nodes[*n0];
                let (x1, y1, _) = frame.deformed_nodes[*n1];

                writeln!(
                    file,
                    r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="blue" stroke-width="2"/>"#,
                    400.0 + x0 * 100.0, 300.0 - y0 * 100.0,
                    400.0 + x1 * 100.0, 300.0 - y1 * 100.0
                )?;
            }

            // Draw original nodes
            for (i, (x, y, _)) in frame.deformed_nodes.iter().enumerate() {
                writeln!(
                    file,
                    r#"<circle cx="{}" cy="{}" r="3" fill="red"/>"#,
                    400.0 + x * 100.0, 300.0 - y * 100.0
                )?;

                // Node labels for first few
                if i < 5 {
                    writeln!(
                        file,
                        r#"<text x="{}" y="{}" font-size="10">{}</text>"#,
                        400.0 + x * 100.0 + 5.0, 300.0 - y * 100.0, i
                    )?;
                }
            }

            writeln!(file, "</svg>")?;
        }

        Ok(())
    }
}

/// Result comparison utilities.
pub mod result_comparison {
    /// Compares two result sets and returns differences.
    pub fn compare_displacements(
        displacements_a: &[f64],
        displacements_b: &[f64],
    ) -> Vec<(usize, f64, f64, f64)> {
        let mut differences = Vec::new();

        for i in 0..displacements_a.len().min(displacements_b.len()) {
            let diff = (displacements_a[i] - displacements_b[i]).abs();
            if diff > 1e-6 {
                differences.push((i, displacements_a[i], displacements_b[i], diff));
            }
        }

        differences
    }

    /// Computes L2 norm of difference.
    pub fn l2_norm_difference(displacements_a: &[f64], displacements_b: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..displacements_a.len().min(displacements_b.len()) {
            let diff = displacements_a[i] - displacements_b[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }

    /// Computes relative error.
    pub fn relative_error(reference: &[f64], computed: &[f64]) -> f64 {
        let mut num = 0.0;
        let mut denom = 0.0;

        for i in 0..reference.len().min(computed.len()) {
            let diff = reference[i] - computed[i];
            num += diff * diff;
            denom += reference[i] * reference[i];
        }

        if denom > 1e-15 {
            (num / denom).sqrt() * 100.0
        } else {
            0.0
        }
    }

    /// Generates comparison report.
    pub fn generate_comparison_report(
        name_a: &str,
        displacements_a: &[f64],
        name_b: &str,
        displacements_b: &[f64],
    ) -> String {
        let mut report = String::new();

        report.push_str(&format!("=== Result Comparison Report ===\n\n"));
        report.push_str(&format!("Dataset A: {} ({} DOFs)\n", name_a, displacements_a.len()));
        report.push_str(&format!("Dataset B: {} ({} DOFs)\n\n", name_b, displacements_b.len()));

        let l2 = l2_norm_difference(displacements_a, displacements_b);
        let rel_err = relative_error(displacements_a, displacements_b);

        report.push_str(&format!("L2 Norm Difference: {:.6e}\n", l2));
        report.push_str(&format!("Relative Error: {:.4}%\n\n", rel_err));

        let differences = compare_displacements(displacements_a, displacements_b);
        if !differences.is_empty() {
            report.push_str("Significant Differences (>1e-6):\n");
            for (i, val_a, val_b, diff) in differences.iter().take(10) {
                report.push_str(&format!(
                    "  DOF {}: {:.6e} vs {:.6e} (diff: {:.6e})\n",
                    i, val_a, val_b, diff
                ));
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use result_comparison::*;

    #[test]
    fn test_contour_plot_creation() {
        let values = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let plot = ContourPlot::new(values.clone(), 5);

        let levels = plot.levels();
        assert_eq!(levels.len(), 5);
        assert!((levels[0] - 0.0).abs() < 1e-10);
        assert!((levels[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_result_comparison() {
        let ref_disp = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let comp_disp = vec![1.001, 2.001, 3.001, 4.001, 5.001];

        let l2 = l2_norm_difference(&ref_disp, &comp_disp);
        assert!(l2 > 0.0);
        assert!(l2 < 0.01);

        let rel_err = relative_error(&ref_disp, &comp_disp);
        assert!(rel_err > 0.0);
        assert!(rel_err < 1.0);
    }

    #[test]
    fn test_animation_generation() {
        let nodes = vec![
            Node::new_3d(0.0, 0.0, 0.0),
            Node::new_3d(1.0, 0.0, 0.0),
            Node::new_3d(1.0, 1.0, 0.0),
        ];
        let mode_shape = vec![0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0];

        let animation = DeformedShapeAnimation::from_mode_shape(&nodes, &mode_shape, 10);
        assert_eq!(animation.frames.len(), 10);
    }
}

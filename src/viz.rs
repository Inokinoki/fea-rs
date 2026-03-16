use crate::core::{Dof, Model};
use crate::elements_legacy::Truss2;
use serde::Serialize;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Configuration for visualization export.
#[derive(Debug, Clone)]
pub struct VizConfig {
    /// Scale factor for deformed shape visualization.
    /// Use 0.0 for undeformed geometry only.
    pub deformation_scale: f64,
    /// Whether to include original geometry for comparison.
    pub show_undeformed: bool,
    /// Whether to include reaction forces in output.
    pub show_reactions: bool,
}

impl Default for VizConfig {
    fn default() -> Self {
        Self {
            deformation_scale: 1.0,
            show_undeformed: true,
            show_reactions: false,
        }
    }
}

impl VizConfig {
    /// Creates a new config with automatic deformation scaling.
    ///
    /// The scale factor is computed to make the maximum displacement
    /// visible relative to the model size.
    pub fn with_auto_scale(model: &Model<Truss2>, u: &[f64], target_fraction: f64) -> Self {
        let mut max_disp: f64 = 0.0;
        for nid in 0..model.nodes.len() {
            for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
                if let Some(idx) = model.dof_index(nid, dof) {
                    if let Some(&disp) = u.get(idx) {
                        max_disp = max_disp.max(disp.abs());
                    }
                }
            }
        }

        // Compute model characteristic length
        let mut max_length: f64 = 0.0;
        for e in &model.elements {
            let p1 = model.nodes[e.n1].as_array();
            let p2 = model.nodes[e.n2].as_array();
            let dx = p2[0] - p1[0];
            let dy = p2[1] - p1[1];
            let dz = p2[2] - p1[2];
            let len = (dx * dx + dy * dy + dz * dz).sqrt();
            max_length = max_length.max(len);
        }

        let scale = if max_disp > 1e-15 && max_length > 1e-15 {
            (max_length * target_fraction) / max_disp
        } else {
            1.0
        };

        Self {
            deformation_scale: scale,
            show_undeformed: true,
            show_reactions: false,
        }
    }
}

/// A minimal unstructured grid for VTK legacy export.
#[derive(Debug, Clone)]
pub struct VtkMesh {
    pub points: Vec<[f64; 3]>,
    /// Cell connectivity (point indices per cell).
    pub cells: Vec<Vec<usize>>,
    /// VTK cell types (e.g. 3 = VTK_LINE).
    pub cell_types: Vec<u8>,
}

impl VtkMesh {
    pub fn from_truss2(model: &Model<Truss2>) -> Self {
        let points = model.nodes.iter().map(|n| n.as_array()).collect();
        let mut cells = Vec::with_capacity(model.elements.len());
        let mut cell_types = Vec::with_capacity(model.elements.len());
        for e in &model.elements {
            cells.push(vec![e.n1, e.n2]);
            cell_types.push(3); // VTK_LINE
        }
        Self {
            points,
            cells,
            cell_types,
        }
    }

    /// Creates a mesh with deformed geometry.
    pub fn from_truss2_deformed(model: &Model<Truss2>, u: &[f64], scale: f64) -> Self {
        let mut points = Vec::with_capacity(model.nodes.len());
        for nid in 0..model.nodes.len() {
            let node = model.nodes[nid];
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

            points.push([
                node.x + scale * ux,
                node.y + scale * uy,
                node.z + scale * uz,
            ]);
        }

        let mut cells = Vec::with_capacity(model.elements.len());
        let mut cell_types = Vec::with_capacity(model.elements.len());
        for e in &model.elements {
            cells.push(vec![e.n1, e.n2]);
            cell_types.push(3); // VTK_LINE
        }
        Self {
            points,
            cells,
            cell_types,
        }
    }
}

/// Writes VTK legacy (`.vtk`) unstructured grids (ASCII).
#[derive(Debug, Default)]
pub struct VtkLegacyWriter;

impl VtkLegacyWriter {
    pub fn new() -> Self {
        Self
    }

    /// Writes a truss model to a VTK legacy file (undeformed geometry).
    ///
    /// - `u`: global displacement vector in the model DOF ordering
    /// - Exports:
    ///   - `POINT_DATA`: `displacement` as VECTORS
    ///   - `CELL_DATA`: `axial_stress` as SCALARS
    pub fn write_truss2<P: AsRef<Path>>(
        &self,
        path: P,
        model: &Model<Truss2>,
        u: &[f64],
    ) -> std::io::Result<()> {
        let mesh = VtkMesh::from_truss2(model);
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);

        writeln!(w, "# vtk DataFile Version 2.0")?;
        writeln!(w, "fea-rs")?;
        writeln!(w, "ASCII")?;
        writeln!(w, "DATASET UNSTRUCTURED_GRID")?;

        // Points
        writeln!(w, "POINTS {} float", mesh.points.len())?;
        for p in &mesh.points {
            writeln!(w, "{} {} {}", p[0], p[1], p[2])?;
        }

        // Cells
        let cell_ints: usize = mesh.cells.iter().map(|c| 1 + c.len()).sum();
        writeln!(w, "CELLS {} {}", mesh.cells.len(), cell_ints)?;
        for c in &mesh.cells {
            write!(w, "{}", c.len())?;
            for &pi in c {
                write!(w, " {}", pi)?;
            }
            writeln!(w)?;
        }

        // Cell types
        writeln!(w, "CELL_TYPES {}", mesh.cell_types.len())?;
        for &ct in &mesh.cell_types {
            writeln!(w, "{}", ct)?;
        }

        // Point data: displacement
        writeln!(w, "POINT_DATA {}", mesh.points.len())?;
        writeln!(w, "VECTORS displacement float")?;
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
            writeln!(w, "{} {} {}", ux, uy, uz)?;
        }

        // Cell data: axial stress
        writeln!(w, "CELL_DATA {}", model.elements.len())?;
        writeln!(w, "SCALARS axial_stress float 1")?;
        writeln!(w, "LOOKUP_TABLE default")?;
        for e in &model.elements {
            let s = e.axial_stress(model, u);
            writeln!(w, "{}", s)?;
        }

        // Point data: displacement magnitude
        writeln!(w, "SCALARS displacement_magnitude float 1")?;
        writeln!(w, "LOOKUP_TABLE default")?;
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
            let mag = (ux * ux + uy * uy + uz * uz).sqrt();
            writeln!(w, "{}", mag)?;
        }

        w.flush()?;
        Ok(())
    }

    /// Writes deformed geometry to a VTK legacy file.
    ///
    /// - `scale`: deformation scale factor (use 1.0 for actual, or auto-computed value)
    pub fn write_truss2_deformed<P: AsRef<Path>>(
        &self,
        path: P,
        model: &Model<Truss2>,
        u: &[f64],
        scale: f64,
    ) -> std::io::Result<()> {
        let mesh = VtkMesh::from_truss2_deformed(model, u, scale);
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);

        writeln!(w, "# vtk DataFile Version 2.0")?;
        writeln!(w, "fea-rs (deformed)")?;
        writeln!(w, "ASCII")?;
        writeln!(w, "DATASET UNSTRUCTURED_GRID")?;

        // Points (deformed)
        writeln!(w, "POINTS {} float", mesh.points.len())?;
        for p in &mesh.points {
            writeln!(w, "{} {} {}", p[0], p[1], p[2])?;
        }

        // Cells
        let cell_ints: usize = mesh.cells.iter().map(|c| 1 + c.len()).sum();
        writeln!(w, "CELLS {} {}", mesh.cells.len(), cell_ints)?;
        for c in &mesh.cells {
            write!(w, "{}", c.len())?;
            for &pi in c {
                write!(w, " {}", pi)?;
            }
            writeln!(w)?;
        }

        // Cell types
        writeln!(w, "CELL_TYPES {}", mesh.cell_types.len())?;
        for &ct in &mesh.cell_types {
            writeln!(w, "{}", ct)?;
        }

        // Point data: displacement (original values)
        writeln!(w, "POINT_DATA {}", mesh.points.len())?;
        writeln!(w, "VECTORS displacement float")?;
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
            writeln!(w, "{} {} {}", ux, uy, uz)?;
        }

        // Cell data: axial stress
        writeln!(w, "CELL_DATA {}", model.elements.len())?;
        writeln!(w, "SCALARS axial_stress float 1")?;
        writeln!(w, "LOOKUP_TABLE default")?;
        for e in &model.elements {
            let s = e.axial_stress(model, u);
            writeln!(w, "{}", s)?;
        }

        w.flush()?;
        Ok(())
    }

    /// Writes both undeformed and deformed geometry in a single VTK file.
    ///
    /// This allows toggling between states in ParaView.
    pub fn write_truss2_with_states<P: AsRef<Path>>(
        &self,
        path: P,
        model: &Model<Truss2>,
        u: &[f64],
        config: &VizConfig,
    ) -> std::io::Result<()> {
        let mesh_undeformed = VtkMesh::from_truss2(model);
        let mesh_deformed = VtkMesh::from_truss2_deformed(model, u, config.deformation_scale);
        let total_points = mesh_undeformed.points.len() + mesh_deformed.points.len();
        let total_cells = mesh_undeformed.cells.len() + mesh_deformed.cells.len();

        let file = File::create(path)?;
        let mut w = BufWriter::new(file);

        writeln!(w, "# vtk DataFile Version 2.0")?;
        writeln!(w, "fea-rs (undeformed + deformed)")?;
        writeln!(w, "ASCII")?;
        writeln!(w, "DATASET UNSTRUCTURED_GRID")?;

        // Combined points (undeformed first, then deformed)
        writeln!(w, "POINTS {} float", total_points)?;
        for p in &mesh_undeformed.points {
            writeln!(w, "{} {} {}", p[0], p[1], p[2])?;
        }
        for p in &mesh_deformed.points {
            writeln!(w, "{} {} {}", p[0], p[1], p[2])?;
        }

        // Combined cells (with offset for deformed)
        let cell_ints = (mesh_undeformed.cells.iter().map(|c| 1 + c.len()).sum::<usize>())
            + (mesh_deformed.cells.iter().map(|c| 1 + c.len()).sum::<usize>());
        writeln!(w, "CELLS {} {}", total_cells, cell_ints)?;

        // Undeformed cells
        for c in &mesh_undeformed.cells {
            write!(w, "{}", c.len())?;
            for &pi in c {
                write!(w, " {}", pi)?;
            }
            writeln!(w)?;
        }

        // Deformed cells (offset by number of undeformed points)
        let offset = mesh_undeformed.points.len();
        for c in &mesh_deformed.cells {
            write!(w, "{}", c.len())?;
            for &pi in c {
                write!(w, " {}", pi + offset)?;
            }
            writeln!(w)?;
        }

        // Cell types
        writeln!(w, "CELL_TYPES {}", total_cells)?;
        for _ in 0..mesh_undeformed.cells.len() {
            writeln!(w, "3")?; // VTK_LINE
        }
        for _ in 0..mesh_deformed.cells.len() {
            writeln!(w, "3")?; // VTK_LINE
        }

        w.flush()?;
        Ok(())
    }
}

/// A simple JSON format intended for web visualization.
#[derive(Debug, Clone, Serialize)]
pub struct JsonTruss2 {
    pub points: Vec<[f64; 3]>,
    pub cells: Vec<[usize; 2]>,
    pub point_data: JsonPointData,
    pub cell_data: JsonCellData,
}

#[derive(Debug, Clone, Serialize)]
pub struct JsonPointData {
    pub displacement: Vec<[f64; 3]>,
    /// Displacement magnitude at each node.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub displacement_magnitude: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct JsonCellData {
    pub axial_stress: Vec<f64>,
}

/// Enhanced JSON format with both undeformed and deformed geometry.
#[derive(Debug, Clone, Serialize)]
pub struct JsonTruss2Enhanced {
    pub undeformed: JsonGeometry,
    pub deformed: JsonGeometry,
    pub cell_data: JsonCellData,
    pub metadata: JsonMetadata,
}

#[derive(Debug, Clone, Serialize)]
pub struct JsonGeometry {
    pub points: Vec<[f64; 3]>,
    pub cells: Vec<[usize; 2]>,
    pub point_data: JsonPointData,
}

#[derive(Debug, Clone, Serialize)]
pub struct JsonMetadata {
    pub deformation_scale: f64,
    pub num_nodes: usize,
    pub num_elements: usize,
    pub max_displacement: f64,
    pub max_stress: f64,
    pub min_stress: f64,
}

/// Writes JSON files for simple web visualization.
#[derive(Debug, Default)]
pub struct JsonWriter;

impl JsonWriter {
    pub fn new() -> Self {
        Self
    }

    pub fn write_truss2<P: AsRef<Path>>(
        &self,
        path: P,
        model: &Model<Truss2>,
        u: &[f64],
    ) -> std::io::Result<()> {
        let points: Vec<[f64; 3]> = model.nodes.iter().map(|n| n.as_array()).collect();
        let cells: Vec<[usize; 2]> = model.elements.iter().map(|e| [e.n1, e.n2]).collect();

        let mut displacement = Vec::with_capacity(model.nodes.len());
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
            displacement.push([ux, uy, uz]);
        }

        let axial_stress: Vec<f64> = model
            .elements
            .iter()
            .map(|e| e.axial_stress(model, u))
            .collect();

        let data = JsonTruss2 {
            points,
            cells,
            point_data: JsonPointData {
                displacement,
                displacement_magnitude: None,
            },
            cell_data: JsonCellData { axial_stress },
        };

        let file = File::create(path)?;
        let mut w = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut w, &data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        w.flush()?;
        Ok(())
    }

    /// Writes enhanced JSON with both undeformed and deformed geometry.
    pub fn write_truss2_enhanced<P: AsRef<Path>>(
        &self,
        path: P,
        model: &Model<Truss2>,
        u: &[f64],
        config: &VizConfig,
    ) -> std::io::Result<()> {
        // Undeformed geometry
        let undeformed_points: Vec<[f64; 3]> = model.nodes.iter().map(|n| n.as_array()).collect();
        let cells: Vec<[usize; 2]> = model.elements.iter().map(|e| [e.n1, e.n2]).collect();

        // Deformed geometry
        let mut deformed_points = Vec::with_capacity(model.nodes.len());
        for nid in 0..model.nodes.len() {
            let node = model.nodes[nid];
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
            deformed_points.push([
                node.x + config.deformation_scale * ux,
                node.y + config.deformation_scale * uy,
                node.z + config.deformation_scale * uz,
            ]);
        }

        // Displacement data with magnitudes
        let mut displacement = Vec::with_capacity(model.nodes.len());
        let mut displacement_magnitude = Vec::with_capacity(model.nodes.len());
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
            displacement.push([ux, uy, uz]);
            displacement_magnitude.push((ux * ux + uy * uy + uz * uz).sqrt());
        }

        // Stress data with statistics
        let axial_stress: Vec<f64> = model
            .elements
            .iter()
            .map(|e| e.axial_stress(model, u))
            .collect();

        let max_stress = axial_stress
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b.abs()));
        let min_stress = axial_stress
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));

        // Max displacement
        let max_disp = displacement_magnitude.iter().copied().fold(0.0_f64, f64::max);

        let point_data = JsonPointData {
            displacement: displacement.clone(),
            displacement_magnitude: Some(displacement_magnitude),
        };

        let data = JsonTruss2Enhanced {
            undeformed: JsonGeometry {
                points: undeformed_points,
                cells: cells.clone(),
                point_data: point_data.clone(),
            },
            deformed: JsonGeometry {
                points: deformed_points,
                cells,
                point_data,
            },
            cell_data: JsonCellData { axial_stress },
            metadata: JsonMetadata {
                deformation_scale: config.deformation_scale,
                num_nodes: model.nodes.len(),
                num_elements: model.elements.len(),
                max_displacement: max_disp,
                max_stress,
                min_stress,
            },
        };

        let file = File::create(path)?;
        let mut w = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut w, &data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        w.flush()?;
        Ok(())
    }
}

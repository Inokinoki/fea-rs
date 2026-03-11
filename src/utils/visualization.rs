//! Visualization export utilities.

use serde::Serialize;

/// Configuration for visualization export.
#[derive(Debug, Clone)]
pub struct VizConfig {
    /// Scale factor for deformed shape visualization.
    pub deformation_scale: f64,
    /// Whether to include original geometry.
    pub show_undeformed: bool,
}

impl Default for VizConfig {
    fn default() -> Self {
        Self {
            deformation_scale: 1.0,
            show_undeformed: true,
        }
    }
}

/// A minimal unstructured grid for VTK legacy export.
#[derive(Debug, Clone)]
pub struct VtkMesh {
    pub points: Vec<[f64; 3]>,
    pub cells: Vec<Vec<usize>>,
    pub cell_types: Vec<u8>,
}

impl VtkMesh {
    /// Creates a VTK mesh from node coordinates and element connectivity.
    pub fn new(points: Vec<[f64; 3]>, cells: Vec<[usize; 2]>) -> Self {
        Self {
            points,
            cells: cells.iter().map(|c| c.to_vec()).collect(),
            cell_types: vec![3; cells.len()], // VTK_LINE
        }
    }
}

/// JSON geometry for web visualization.
#[derive(Debug, Clone, Serialize)]
pub struct JsonGeometry {
    pub points: Vec<[f64; 3]>,
    pub cells: Vec<[usize; 2]>,
}

/// JSON output for visualization.
#[derive(Debug, Clone, Serialize)]
pub struct JsonOutput {
    pub geometry: JsonGeometry,
    pub displacements: Vec<[f64; 3]>,
    pub stress: Vec<f64>,
}

impl JsonOutput {
    /// Creates a new JSON output.
    pub fn new(
        points: Vec<[f64; 3]>,
        cells: Vec<[usize; 2]>,
        displacements: Vec<[f64; 3]>,
        stress: Vec<f64>,
    ) -> Self {
        Self {
            geometry: JsonGeometry { points, cells },
            displacements,
            stress,
        }
    }
}

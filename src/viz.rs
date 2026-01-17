use crate::core::{Dof, Model};
use crate::elements::Truss2;
use serde::Serialize;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

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
}

/// Writes VTK legacy (`.vtk`) unstructured grids (ASCII).
#[derive(Debug, Default)]
pub struct VtkLegacyWriter;

impl VtkLegacyWriter {
    pub fn new() -> Self {
        Self
    }

    /// Writes a truss model to a VTK legacy file.
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
}

#[derive(Debug, Clone, Serialize)]
pub struct JsonCellData {
    pub axial_stress: Vec<f64>,
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
            point_data: JsonPointData { displacement },
            cell_data: JsonCellData { axial_stress },
        };

        let file = File::create(path)?;
        let mut w = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut w, &data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        w.flush()?;
        Ok(())
    }
}

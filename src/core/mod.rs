//! Core data structures for finite element analysis.
//!
//! This module provides:
//! - Node and coordinate types
//! - Degrees of freedom (DOF) definitions
//! - Model structure for assembling FEA problems

use std::collections::BTreeMap;

/// A node identifier.
pub type NodeId = usize;

/// Degrees of freedom supported by the library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dof {
    /// Translation in X direction.
    Ux,
    /// Translation in Y direction.
    Uy,
    /// Translation in Z direction.
    Uz,
}

impl Dof {
    /// Returns the index of this DOF in a 3D system (0-2).
    pub fn index_in_3d(self) -> usize {
        match self {
            Dof::Ux => 0,
            Dof::Uy => 1,
            Dof::Uz => 2,
        }
    }
}

/// A 3D node coordinate. Use `z = 0.0` for 2D models.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Node {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Node {
    /// Creates a new 2D node.
    pub fn new_2d(x: f64, y: f64) -> Self {
        Self { x, y, z: 0.0 }
    }

    /// Creates a new 3D node.
    pub fn new_3d(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Returns the node coordinates as an array.
    pub fn as_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
}

mod boundary;
mod material;

pub use boundary::{BoundaryCondition, Load};
pub use material::{Material, Section, STEEL_A36, STAINLESS_STEEL_304, ALUMINUM_6061_T6, ALUMINUM_7075_T6, TITANIUM_TI6AL4V};

/// A finite element model: nodes, elements, loads, and boundary conditions.
///
/// The library currently focuses on truss-like problems where each node has
/// displacement DOFs in X/Y/Z.
#[derive(Debug)]
pub struct Model<E> {
    pub nodes: Vec<Node>,
    pub elements: Vec<E>,
    pub loads: Vec<Load>,
    pub bcs: Vec<BoundaryCondition>,
    pub materials: Vec<Material>,
    pub sections: Vec<Section>,
    dof_map: BTreeMap<(NodeId, Dof), usize>,
}

impl<E> Default for Model<E> {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            elements: Vec::new(),
            loads: Vec::new(),
            bcs: Vec::new(),
            materials: Vec::new(),
            sections: Vec::new(),
            dof_map: BTreeMap::new(),
        }
    }
}

impl<E> Model<E> {
    /// Creates a new empty model.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a node and returns its `NodeId`.
    pub fn add_node(&mut self, node: Node) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(node);
        id
    }

    /// Adds an element to the model.
    pub fn add_element(&mut self, element: E) {
        self.elements.push(element);
    }

    /// Adds a nodal load to the model.
    pub fn add_load(&mut self, load: Load) {
        self.loads.push(load);
    }

    /// Adds a boundary condition to the model.
    pub fn add_bc(&mut self, bc: BoundaryCondition) {
        self.bcs.push(bc);
    }

    /// Adds a material and returns its index.
    pub fn add_material(&mut self, material: Material) -> usize {
        let id = self.materials.len();
        self.materials.push(material);
        id
    }

    /// Adds a section and returns its index.
    pub fn add_section(&mut self, section: Section) -> usize {
        let id = self.sections.len();
        self.sections.push(section);
        id
    }

    /// Builds a consistent DOF map for all nodes/DOFs that appear.
    ///
    /// For now we register Ux/Uy/Uz for every node, which keeps the
    /// mapping stable and simple.
    pub fn build_dofs_3d(&mut self) -> usize {
        self.dof_map.clear();
        let mut next = 0usize;
        for node in 0..self.nodes.len() {
            for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
                self.dof_map.insert((node, dof), next);
                next += 1;
            }
        }
        next
    }

    /// Returns the global DOF index for a node/DOF pair.
    pub fn dof_index(&self, node: NodeId, dof: Dof) -> Option<usize> {
        self.dof_map.get(&(node, dof)).copied()
    }

    /// Returns the number of DOFs in the model.
    pub fn ndofs(&self) -> usize {
        self.dof_map.len()
    }
}

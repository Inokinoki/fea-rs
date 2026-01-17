use std::collections::BTreeMap;

/// A node identifier.
pub type NodeId = usize;

/// Degrees of freedom supported by the library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dof {
    Ux,
    Uy,
    Uz,
}

impl Dof {
    pub fn index_in_3d(self) -> usize {
        match self {
            Dof::Ux => 0,
            Dof::Uy => 1,
            Dof::Uz => 2,
        }
    }
}

/// A nodal boundary condition (Dirichlet/essential condition).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundaryCondition {
    pub node: NodeId,
    pub dof: Dof,
    pub value: f64,
}

/// A nodal load.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Load {
    pub node: NodeId,
    pub dof: Dof,
    pub value: f64,
}

/// A 3D node coordinate. Use `z = 0.0` for 2D models.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Node {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Node {
    pub fn new_2d(x: f64, y: f64) -> Self {
        Self { x, y, z: 0.0 }
    }

    pub fn new_3d(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn as_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
}

/// A finite element model: nodes, elements, loads, and boundary conditions.
///
/// The library currently focuses on truss-like problems where each node has
/// displacement DOFs in X/Y/Z.
#[derive(Debug, Default)]
pub struct Model<E> {
    pub nodes: Vec<Node>,
    pub elements: Vec<E>,
    pub loads: Vec<Load>,
    pub bcs: Vec<BoundaryCondition>,
    dof_map: BTreeMap<(NodeId, Dof), usize>,
}

impl<E> Model<E> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a node and returns its `NodeId`.
    pub fn add_node(&mut self, node: Node) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(node);
        id
    }

    pub fn add_element(&mut self, element: E) {
        self.elements.push(element);
    }

    pub fn add_load(&mut self, load: Load) {
        self.loads.push(load);
    }

    pub fn add_bc(&mut self, bc: BoundaryCondition) {
        self.bcs.push(bc);
    }

    /// Builds a consistent DOF map for all nodes/DOFs that appear.
    ///
    /// For now we always register Ux/Uy/Uz for every node, which keeps the
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

    pub fn dof_index(&self, node: NodeId, dof: Dof) -> Option<usize> {
        self.dof_map.get(&(node, dof)).copied()
    }
}

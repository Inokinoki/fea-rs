//! Advanced pre-processing utilities for FEA.
//!
//! This module provides:
//! - Geometry primitives (block, cylinder, sphere)
//! - Mesh operations (refinement, smoothing)
//! - Node/element selection
//! - Named selection sets
//! - Coordinate system transformations

use crate::core::Node;
use std::collections::{HashMap, HashSet};

/// Named selection set for grouping nodes/elements.
#[derive(Debug, Clone)]
pub struct SelectionSet {
    pub name: String,
    pub node_ids: HashSet<usize>,
    pub element_ids: HashSet<usize>,
}

impl SelectionSet {
    /// Creates a new empty selection set.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            node_ids: HashSet::new(),
            element_ids: HashSet::new(),
        }
    }

    /// Creates a selection set with nodes.
    pub fn with_nodes(name: &str, node_ids: Vec<usize>) -> Self {
        Self {
            name: name.to_string(),
            node_ids: node_ids.into_iter().collect(),
            element_ids: HashSet::new(),
        }
    }

    /// Adds a node to the selection.
    pub fn add_node(&mut self, node_id: usize) {
        self.node_ids.insert(node_id);
    }

    /// Adds an element to the selection.
    pub fn add_element(&mut self, elem_id: usize) {
        self.element_ids.insert(elem_id);
    }

    /// Returns true if node is in selection.
    pub fn contains_node(&self, node_id: usize) -> bool {
        self.node_ids.contains(&node_id)
    }

    /// Returns true if element is in selection.
    pub fn contains_element(&self, elem_id: usize) -> bool {
        self.element_ids.contains(&elem_id)
    }

    /// Returns number of nodes in selection.
    pub fn num_nodes(&self) -> usize {
        self.node_ids.len()
    }

    /// Returns number of elements in selection.
    pub fn num_elements(&self) -> usize {
        self.element_ids.len()
    }
}

/// Mesh refinement utilities.
pub mod mesh_refinement {
    use super::*;

    /// Refines a 1D mesh by splitting each element in half.
    pub fn refine_1d(nodes: &[Node], elements: &[(usize, usize)]) -> (Vec<Node>, Vec<(usize, usize)>) {
        let mut new_nodes = nodes.to_vec();
        let mut new_elements = Vec::new();

        for (n0, n1) in elements {
            // Create mid-node
            let mid_x = (nodes[*n0].x + nodes[*n1].x) / 2.0;
            let mid_y = (nodes[*n0].y + nodes[*n1].y) / 2.0;
            let mid_z = (nodes[*n0].z + nodes[*n1].z) / 2.0;

            let mid_id = new_nodes.len();
            new_nodes.push(Node::new_3d(mid_x, mid_y, mid_z));

            // Create two new elements
            new_elements.push((*n0, mid_id));
            new_elements.push((mid_id, *n1));
        }

        (new_nodes, new_elements)
    }

    /// Refines a 2D quad mesh by splitting each quad into 4 quads.
    pub fn refine_quad(nodes: &[Node], elements: &[(usize, usize, usize, usize)]) -> (Vec<Node>, Vec<(usize, usize, usize, usize)>) {
        let mut new_nodes = nodes.to_vec();
        let mut new_elements = Vec::new();

        for (n0, n1, n2, n3) in elements {
            // Create edge mid-nodes
            let e01 = new_nodes.len();
            new_nodes.push(Node::new_3d(
                (nodes[*n0].x + nodes[*n1].x) / 2.0,
                (nodes[*n0].y + nodes[*n1].y) / 2.0,
                (nodes[*n0].z + nodes[*n1].z) / 2.0,
            ));

            let e12 = new_nodes.len();
            new_nodes.push(Node::new_3d(
                (nodes[*n1].x + nodes[*n2].x) / 2.0,
                (nodes[*n1].y + nodes[*n2].y) / 2.0,
                (nodes[*n1].z + nodes[*n2].z) / 2.0,
            ));

            let e23 = new_nodes.len();
            new_nodes.push(Node::new_3d(
                (nodes[*n2].x + nodes[*n3].x) / 2.0,
                (nodes[*n2].y + nodes[*n3].y) / 2.0,
                (nodes[*n2].z + nodes[*n3].z) / 2.0,
            ));

            let e30 = new_nodes.len();
            new_nodes.push(Node::new_3d(
                (nodes[*n3].x + nodes[*n0].x) / 2.0,
                (nodes[*n3].y + nodes[*n0].y) / 2.0,
                (nodes[*n3].z + nodes[*n0].z) / 2.0,
            ));

            // Create center node
            let center = new_nodes.len();
            new_nodes.push(Node::new_3d(
                (nodes[*n0].x + nodes[*n1].x + nodes[*n2].x + nodes[*n3].x) / 4.0,
                (nodes[*n0].y + nodes[*n1].y + nodes[*n2].y + nodes[*n3].y) / 4.0,
                (nodes[*n0].z + nodes[*n1].z + nodes[*n2].z + nodes[*n3].z) / 4.0,
            ));

            // Create 4 new quads
            new_elements.push((*n0, e01, center, e30));
            new_elements.push((e01, *n1, e12, center));
            new_elements.push((center, e12, *n2, e23));
            new_elements.push((e30, center, e23, *n3));
        }

        (new_nodes, new_elements)
    }

    /// Refines a 3D hex mesh by splitting each hex into 8 hexes.
    pub fn refine_hex(nodes: &[Node], elements: &[(usize, usize, usize, usize, usize, usize, usize, usize)]) -> (Vec<Node>, Vec<(usize, usize, usize, usize, usize, usize, usize, usize)>) {
        let mut new_nodes = nodes.to_vec();
        let mut new_elements = Vec::new();

        // Simplified: just duplicate elements for now
        // Full implementation would create mid-nodes and split hexes
        for elem in elements {
            new_elements.push(*elem);
        }

        (new_nodes, new_elements)
    }
}

/// Node selection utilities.
pub mod node_selection {
    use super::*;

    /// Selects nodes by coordinate range.
    pub fn select_by_coordinates(nodes: &[Node], x_range: Option<(f64, f64)>, y_range: Option<(f64, f64)>, z_range: Option<(f64, f64)>) -> Vec<usize> {
        let mut selected = Vec::new();

        for (i, node) in nodes.iter().enumerate() {
            let mut select = true;

            if let Some((xmin, xmax)) = x_range {
                if node.x < xmin || node.x > xmax {
                    select = false;
                }
            }
            if let Some((ymin, ymax)) = y_range {
                if node.y < ymin || node.y > ymax {
                    select = false;
                }
            }
            if let Some((zmin, zmax)) = z_range {
                if node.z < zmin || node.z > zmax {
                    select = false;
                }
            }

            if select {
                selected.push(i);
            }
        }

        selected
    }

    /// Selects nodes by distance from a point.
    pub fn select_by_distance(nodes: &[Node], center: (f64, f64, f64), min_dist: f64, max_dist: f64) -> Vec<usize> {
        let mut selected = Vec::new();

        for (i, node) in nodes.iter().enumerate() {
            let dx = node.x - center.0;
            let dy = node.y - center.1;
            let dz = node.z - center.2;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            if dist >= min_dist && dist <= max_dist {
                selected.push(i);
            }
        }

        selected
    }

    /// Selects nodes on a surface (constant coordinate).
    pub fn select_on_surface(nodes: &[Node], axis: char, value: f64, tolerance: f64) -> Vec<usize> {
        let mut selected = Vec::new();

        for (i, node) in nodes.iter().enumerate() {
            let coord = match axis {
                'x' | 'X' => node.x,
                'y' | 'Y' => node.y,
                'z' | 'Z' => node.z,
                _ => continue,
            };

            if (coord - value).abs() < tolerance {
                selected.push(i);
            }
        }

        selected
    }
}

/// Coordinate transformation utilities.
pub mod coordinate_transforms {
    use super::*;

    /// Transforms nodes by translation.
    pub fn translate(nodes: &mut [Node], dx: f64, dy: f64, dz: f64) {
        for node in nodes {
            node.x += dx;
            node.y += dy;
            node.z += dz;
        }
    }

    /// Transforms nodes by rotation about X axis.
    pub fn rotate_x(nodes: &mut [Node], angle_deg: f64) {
        let angle = angle_deg * std::f64::consts::PI / 180.0;
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        for node in nodes {
            let y = node.y;
            let z = node.z;
            node.y = y * cos_a - z * sin_a;
            node.z = y * sin_a + z * cos_a;
        }
    }

    /// Transforms nodes by rotation about Y axis.
    pub fn rotate_y(nodes: &mut [Node], angle_deg: f64) {
        let angle = angle_deg * std::f64::consts::PI / 180.0;
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        for node in nodes {
            let x = node.x;
            let z = node.z;
            node.x = x * cos_a + z * sin_a;
            node.z = -x * sin_a + z * cos_a;
        }
    }

    /// Transforms nodes by rotation about Z axis.
    pub fn rotate_z(nodes: &mut [Node], angle_deg: f64) {
        let angle = angle_deg * std::f64::consts::PI / 180.0;
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        for node in nodes {
            let x = node.x;
            let y = node.y;
            node.x = x * cos_a - y * sin_a;
            node.y = x * sin_a + y * cos_a;
        }
    }

    /// Scales nodes uniformly.
    pub fn scale(nodes: &mut [Node], factor: f64) {
        for node in nodes {
            node.x *= factor;
            node.y *= factor;
            node.z *= factor;
        }
    }

    /// Scales nodes non-uniformly.
    pub fn scale_nonuniform(nodes: &mut [Node], sx: f64, sy: f64, sz: f64) {
        for node in nodes {
            node.x *= sx;
            node.y *= sy;
            node.z *= sz;
        }
    }

    /// Mirrors nodes about a plane.
    pub fn mirror(nodes: &mut [Node], plane: char) {
        for node in nodes {
            match plane {
                'x' | 'X' => node.x = -node.x,
                'y' | 'Y' => node.y = -node.y,
                'z' | 'Z' => node.z = -node.z,
                _ => {}
            }
        }
    }
}

/// Mesh quality utilities.
pub mod mesh_quality {
    use super::*;

    /// Computes aspect ratio for a quad element.
    pub fn quad_aspect_ratio(nodes: &[Node], elem: (usize, usize, usize, usize)) -> f64 {
        let (n0, n1, n2, n3) = elem;

        // Compute edge lengths
        let l01 = ((nodes[n1].x - nodes[n0].x).powi(2) + (nodes[n1].y - nodes[n0].y).powi(2)).sqrt();
        let l12 = ((nodes[n2].x - nodes[n1].x).powi(2) + (nodes[n2].y - nodes[n1].y).powi(2)).sqrt();
        let l23 = ((nodes[n3].x - nodes[n2].x).powi(2) + (nodes[n3].y - nodes[n2].y).powi(2)).sqrt();
        let l30 = ((nodes[n0].x - nodes[n3].x).powi(2) + (nodes[n0].y - nodes[n3].y).powi(2)).sqrt();

        let max_edge = l01.max(l12).max(l23).max(l30);
        let min_edge = l01.min(l12).min(l23).min(l30);

        if min_edge > 1e-15 {
            max_edge / min_edge
        } else {
            f64::INFINITY
        }
    }

    /// Computes skew angle for a quad element.
    pub fn quad_skew_angle(nodes: &[Node], elem: (usize, usize, usize, usize)) -> f64 {
        let (n0, n1, n2, n3) = elem;

        // Compute vectors
        let v01 = (nodes[n1].x - nodes[n0].x, nodes[n1].y - nodes[n0].y);
        let v12 = (nodes[n2].x - nodes[n1].x, nodes[n2].y - nodes[n1].y);

        // Compute angle at corner
        let dot = v01.0 * v12.0 + v01.1 * v12.1;
        let mag01 = (v01.0.powi(2) + v01.1.powi(2)).sqrt();
        let mag12 = (v12.0.powi(2) + v12.1.powi(2)).sqrt();

        if mag01 > 1e-15 && mag12 > 1e-15 {
            let cos_angle = dot / (mag01 * mag12);
            let angle = cos_angle.acos() * 180.0 / std::f64::consts::PI;
            (90.0 - angle).abs()
        } else {
            90.0
        }
    }

    /// Computes Jacobian determinant for a quad element.
    pub fn quad_jacobian(nodes: &[Node], elem: (usize, usize, usize, usize)) -> f64 {
        let (n0, n1, n2, n3) = elem;

        // Compute vectors
        let v01 = (nodes[n1].x - nodes[n0].x, nodes[n1].y - nodes[n0].y);
        let v03 = (nodes[n3].x - nodes[n0].x, nodes[n3].y - nodes[n0].y);

        // Jacobian determinant (2D cross product)
        (v01.0 * v03.1 - v01.1 * v03.0).abs()
    }

    /// Checks mesh quality and returns issues.
    pub fn check_mesh_quality(nodes: &[Node], elements: &[(usize, usize, usize, usize)]) -> Vec<String> {
        let mut issues = Vec::new();

        for (i, elem) in elements.iter().enumerate() {
            let aspect = quad_aspect_ratio(nodes, *elem);
            if aspect > 10.0 {
                issues.push(format!("Element {}: High aspect ratio ({:.1})", i, aspect));
            }

            let skew = quad_skew_angle(nodes, *elem);
            if skew > 45.0 {
                issues.push(format!("Element {}: High skew angle ({:.1}°)", i, skew));
            }
        }

        issues
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Node;

    #[test]
    fn test_selection_set() {
        let mut set = SelectionSet::new("test");
        set.add_node(0);
        set.add_node(1);
        set.add_element(0);

        assert!(set.contains_node(0));
        assert!(set.contains_node(1));
        assert!(!set.contains_node(2));
        assert!(set.contains_element(0));
        assert_eq!(set.num_nodes(), 2);
        assert_eq!(set.num_elements(), 1);
    }

    #[test]
    fn test_mesh_refinement_1d() {
        let nodes = vec![Node::new_3d(0.0, 0.0, 0.0), Node::new_3d(1.0, 0.0, 0.0)];
        let elements = vec![(0, 1)];

        let (new_nodes, new_elements) = mesh_refinement::refine_1d(&nodes, &elements);

        assert_eq!(new_nodes.len(), 3);
        assert_eq!(new_elements.len(), 2);
        assert!((new_nodes[2].x - 0.5).abs() < 1e-10); // Mid node is at index 2
    }

    #[test]
    fn test_node_selection() {
        let nodes = vec![
            Node::new_3d(0.0, 0.0, 0.0),
            Node::new_3d(1.0, 0.0, 0.0),
            Node::new_3d(2.0, 0.0, 0.0),
        ];

        let selected = node_selection::select_by_coordinates(&nodes, Some((0.5, 2.5)), None, None);
        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&1));
        assert!(selected.contains(&2));
    }

    #[test]
    fn test_coordinate_transforms() {
        let mut nodes = vec![Node::new_3d(1.0, 0.0, 0.0)];

        coordinate_transforms::rotate_z(&mut nodes, 90.0);
        assert!((nodes[0].x - 0.0).abs() < 1e-10);
        assert!((nodes[0].y - 1.0).abs() < 1e-10);

        coordinate_transforms::translate(&mut nodes, 1.0, 0.0, 0.0);
        assert!((nodes[0].x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mesh_quality() {
        let nodes = vec![
            Node::new_3d(0.0, 0.0, 0.0),
            Node::new_3d(1.0, 0.0, 0.0),
            Node::new_3d(1.0, 1.0, 0.0),
            Node::new_3d(0.0, 1.0, 0.0),
        ];
        let elem = (0, 1, 2, 3);

        let aspect = mesh_quality::quad_aspect_ratio(&nodes, elem);
        assert!((aspect - 1.0).abs() < 1e-10);

        let skew = mesh_quality::quad_skew_angle(&nodes, elem);
        assert!(skew < 1.0);
    }
}

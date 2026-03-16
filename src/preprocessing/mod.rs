//! Pre-processing module for FEA models.
//!
//! This module provides:
//! - Mesh generation utilities (1D, 2D, 3D)
//! - Geometry import/export (STL, STEP-like)
//! - Boundary condition helpers
//! - Model assembly utilities
//! - Material assignment helpers

use crate::core::{Node, NodeId, BoundaryCondition, Dof};
use crate::elements::Truss2;
use crate::materials::Material;
use crate::Model;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// Mesh generation utilities.
pub mod mesh_generation {
    use super::*;

    /// Generates a 1D bar mesh.
    pub fn generate_bar_1d(
        length: f64,
        num_elements: usize,
        area: f64,
    ) -> (Vec<Node>, Vec<(usize, usize)>) {
        let dx = length / num_elements as f64;
        let mut nodes = Vec::with_capacity(num_elements + 1);
        let mut elements = Vec::with_capacity(num_elements);

        // Generate nodes
        for i in 0..=num_elements {
            nodes.push(Node::new_3d(i as f64 * dx, 0.0, 0.0));
        }

        // Generate elements
        for i in 0..num_elements {
            elements.push((i, i + 1));
        }

        (nodes, elements)
    }

    /// Generates a 2D rectangular mesh (quad elements).
    pub fn generate_rect_2d(
        width: f64,
        height: f64,
        nx: usize,
        ny: usize,
    ) -> (Vec<Node>, Vec<[usize; 4]>) {
        let dx = width / nx as f64;
        let dy = height / ny as f64;
        let mut nodes = Vec::with_capacity((nx + 1) * (ny + 1));
        let mut elements = Vec::with_capacity(nx * ny);

        // Generate nodes
        for j in 0..=ny {
            for i in 0..=nx {
                nodes.push(Node::new_2d(i as f64 * dx, j as f64 * dy));
            }
        }

        // Generate quad elements
        for j in 0..ny {
            for i in 0..nx {
                let n0 = j * (nx + 1) + i;
                let n1 = n0 + 1;
                let n2 = n0 + nx + 2;
                let n3 = n0 + nx + 1;
                elements.push([n0, n1, n2, n3]);
            }
        }

        (nodes, elements)
    }

    /// Generates a 2D triangular mesh from rectangular mesh.
    pub fn generate_tri_2d_from_rect(
        width: f64,
        height: f64,
        nx: usize,
        ny: usize,
    ) -> (Vec<Node>, Vec<[usize; 3]>) {
        let (nodes, quads) = generate_rect_2d(width, height, nx, ny);
        let mut triangles = Vec::with_capacity(quads.len() * 2);

        // Split each quad into 2 triangles
        for quad in quads {
            triangles.push([quad[0], quad[1], quad[2]]);
            triangles.push([quad[0], quad[2], quad[3]]);
        }

        (nodes, triangles)
    }

    /// Generates a 3D rectangular mesh (hex elements).
    pub fn generate_box_3d(
        width: f64,
        height: f64,
        depth: f64,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> (Vec<Node>, Vec<[usize; 8]>) {
        let dx = width / nx as f64;
        let dy = height / ny as f64;
        let dz = depth / nz as f64;
        let mut nodes = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1));
        let mut elements = Vec::with_capacity(nx * ny * nz);

        // Generate nodes
        for k in 0..=nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    nodes.push(Node::new_3d(
                        i as f64 * dx,
                        j as f64 * dy,
                        k as f64 * dz,
                    ));
                }
            }
        }

        // Generate hex elements
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let n0 = k * (nx + 1) * (ny + 1) + j * (nx + 1) + i;
                    let n1 = n0 + 1;
                    let n2 = n0 + nx + 2;
                    let n3 = n0 + nx + 1;
                    let n4 = n0 + (nx + 1) * (ny + 1);
                    let n5 = n4 + 1;
                    let n6 = n4 + nx + 2;
                    let n7 = n4 + nx + 1;
                    elements.push([n0, n1, n2, n3, n4, n5, n6, n7]);
                }
            }
        }

        (nodes, elements)
    }

    /// Generates a tetrahedral mesh from hex mesh.
    pub fn generate_tet_3d_from_hex(
        width: f64,
        height: f64,
        depth: f64,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> (Vec<Node>, Vec<[usize; 4]>) {
        let (nodes, hexes) = generate_box_3d(width, height, depth, nx, ny, nz);
        let mut tets = Vec::with_capacity(hexes.len() * 5);

        // Split each hex into 5 tetrahedra
        for hex in hexes {
            tets.push([hex[0], hex[1], hex[3], hex[4]]);
            tets.push([hex[1], hex[2], hex[3], hex[4]]);
            tets.push([hex[1], hex[2], hex[6], hex[4]]);
            tets.push([hex[4], hex[6], hex[2], hex[5]]);
            tets.push([hex[4], hex[7], hex[6], hex[2]]);
        }

        (nodes, tets)
    }

    /// Generates a cylindrical mesh.
    pub fn generate_cylinder_3d(
        radius: f64,
        length: f64,
        n_radial: usize,
        n_circum: usize,
        n_axial: usize,
    ) -> (Vec<Node>, Vec<[usize; 8]>) {
        let mut nodes = Vec::new();
        let mut elements = Vec::new();

        let dr = radius / n_radial as f64;
        let dtheta = 2.0 * std::f64::consts::PI / n_circum as f64;
        let dz = length / n_axial as f64;

        // Generate nodes
        for k in 0..=n_axial {
            let z = k as f64 * dz;
            for j in 0..n_circum {
                let theta = j as f64 * dtheta;
                for i in 0..=n_radial {
                    let r = i as f64 * dr;
                    let x = r * theta.cos();
                    let y = r * theta.sin();
                    nodes.push(Node::new_3d(x, y, z));
                }
            }
        }

        // Generate hexahedral elements
        let n_nodes_radial = n_radial + 1;
        let n_nodes_circum = n_circum;
        let n_nodes_axial = n_axial + 1;

        for k in 0..n_axial {
            for j in 0..n_circum {
                for i in 0..n_radial {
                    // Node indices for current hexahedron
                    let n0 = k * (n_nodes_circum * n_nodes_radial) + j * n_nodes_radial + i;
                    let n1 = n0 + 1; // radial direction
                    let n2 = ((j + 1) % n_nodes_circum) * n_nodes_radial + i;
                    let n3 = n2 + 1; // radial direction at j+1

                    let n4 = (k + 1) * (n_nodes_circum * n_nodes_radial) + j * n_nodes_radial + i;
                    let n5 = n4 + 1; // radial direction
                    let n6 = ((j + 1) % n_nodes_circum) * n_nodes_radial + i;
                    let n7 = n6 + 1; // radial direction at j+1
                    let n6 = (k + 1) * (n_nodes_circum * n_nodes_radial) + ((j + 1) % n_nodes_circum) * n_nodes_radial + i;
                    let n7 = n6 + 1;

                    // Adjust for proper hex connectivity
                    let base = k * (n_nodes_circum * n_nodes_radial) + j * n_nodes_radial;
                    let base_top = (k + 1) * (n_nodes_circum * n_nodes_radial) + j * n_nodes_radial;
                    let j_next = (j + 1) % n_nodes_circum;

                    let n = |ii: usize, jj: usize, kk: usize| -> usize {
                        let j_wrap = (j + jj) % n_nodes_circum;
                        kk * (n_nodes_circum * n_nodes_radial) + j_wrap * n_nodes_radial + ii
                    };

                    if i < n_radial {
                        elements.push([
                            n(i, 0, k),
                            n(i + 1, 0, k),
                            n(i + 1, 1, k),
                            n(i, 1, k),
                            n(i, 0, k + 1),
                            n(i + 1, 0, k + 1),
                            n(i + 1, 1, k + 1),
                            n(i, 1, k + 1),
                        ]);
                    }
                }
            }
        }

        (nodes, elements)
    }

    /// Generates a structured mesh around a circle (2D).
    pub fn generate_plate_with_hole_2d(
        width: f64,
        height: f64,
        hole_radius: f64,
        nx: usize,
        ny: usize,
    ) -> (Vec<Node>, Vec<[usize; 4]>) {
        let mut nodes = Vec::new();
        let mut elements = Vec::new();

        let dx = width / nx as f64;
        let dy = height / ny as f64;
        let cx = width / 2.0;
        let cy = height / 2.0;

        // Generate nodes, skipping those inside hole
        let mut node_map: Vec<Vec<Option<usize>>> = vec![vec![None; nx + 1]; ny + 1];
        let mut node_count = 0;

        for j in 0..=ny {
            for i in 0..=nx {
                let x = i as f64 * dx;
                let y = j as f64 * dy;

                // Check if outside hole
                let dist_sq = (x - cx).powi(2) + (y - cy).powi(2);
                if dist_sq >= hole_radius.powi(2) {
                    node_map[j][i] = Some(node_count);
                    nodes.push(Node::new_2d(x, y));
                    node_count += 1;
                }
            }
        }

        // Generate quadrilateral elements
        for j in 0..ny {
            for i in 0..nx {
                // Check if all four corners exist
                let n0 = node_map[j][i];
                let n1 = node_map[j][i + 1];
                let n2 = node_map[j + 1][i + 1];
                let n3 = node_map[j + 1][i];

                if let (Some(n0), Some(n1), Some(n2), Some(n3)) = (n0, n1, n2, n3) {
                    elements.push([n0, n1, n2, n3]);
                }
            }
        }

        (nodes, elements)
    }
}

/// Mesh quality tools.
pub mod mesh_quality {
    use super::*;

    /// Mesh quality metrics result.
    #[derive(Debug, Clone)]
    pub struct MeshQualityReport {
        /// Minimum aspect ratio.
        pub min_aspect_ratio: f64,
        /// Maximum aspect ratio.
        pub max_aspect_ratio: f64,
        /// Average aspect ratio.
        pub avg_aspect_ratio: f64,
        /// Minimum Jacobian determinant.
        pub min_jacobian: f64,
        /// Maximum Jacobian determinant.
        pub max_jacobian: f64,
        /// Average Jacobian determinant.
        pub avg_jacobian: f64,
        /// Number of elements with aspect ratio > 10.
        pub high_aspect_ratio_count: usize,
        /// Number of elements with Jacobian < 0.5.
        pub distorted_element_count: usize,
        /// Overall quality score (0-1, 1 = perfect).
        pub quality_score: f64,
        /// Pass/fail status.
        pub passed: bool,
    }

    impl MeshQualityReport {
        /// Creates a new mesh quality report.
        pub fn new(
            min_ar: f64,
            max_ar: f64,
            avg_ar: f64,
            min_jac: f64,
            max_jac: f64,
            avg_jac: f64,
            high_ar_count: usize,
            distorted_count: usize,
        ) -> Self {
            // Quality score based on aspect ratio and Jacobian
            let ar_score = (1.0 / max_ar).min(1.0);
            let jac_score = min_jac.min(1.0);
            let quality_score = (ar_score + jac_score) / 2.0;

            // Pass if quality score > 0.5 and no severely distorted elements
            let passed = quality_score > 0.5 && distorted_count == 0;

            Self {
                min_aspect_ratio: min_ar,
                max_aspect_ratio: max_ar,
                avg_aspect_ratio: avg_ar,
                min_jacobian: min_jac,
                max_jacobian: max_jac,
                avg_jacobian: avg_jac,
                high_aspect_ratio_count: high_ar_count,
                distorted_element_count: distorted_count,
                quality_score,
                passed,
            }
        }

        /// Returns a summary of the quality report.
        pub fn summary(&self) -> String {
            format!(
                "Mesh Quality Report\n\
                 ───────────────────\n\
                 Aspect Ratio: min={:.3}, max={:.3}, avg={:.3}\n\
                 Jacobian: min={:.3}, max={:.3}, avg={:.3}\n\
                 High AR elements: {}\n\
                 Distorted elements: {}\n\
                 Quality Score: {:.2}/1.00\n\
                 Status: {}",
                self.min_aspect_ratio, self.max_aspect_ratio, self.avg_aspect_ratio,
                self.min_jacobian, self.max_jacobian, self.avg_jacobian,
                self.high_aspect_ratio_count,
                self.distorted_element_count,
                self.quality_score,
                if self.passed { "PASS" } else { "FAIL" }
            )
        }
    }

    /// Computes aspect ratio for a quad element.
    pub fn quad_aspect_ratio(nodes: &[Node; 4]) -> f64 {
        // Compute edge lengths
        let edge_lengths: Vec<f64> = (0..4)
            .map(|i| {
                let j = (i + 1) % 4;
                let dx = nodes[j].x - nodes[i].x;
                let dy = nodes[j].y - nodes[i].y;
                let dz = nodes[j].z - nodes[i].z;
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .collect();

        // Aspect ratio = max edge / min edge
        let max_edge = edge_lengths.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_edge = edge_lengths.iter().cloned().fold(f64::INFINITY, f64::min);

        if min_edge > 1e-15 {
            max_edge / min_edge
        } else {
            f64::INFINITY
        }
    }

    /// Computes aspect ratio for a hex element.
    pub fn hex_aspect_ratio(nodes: &[Node; 8]) -> f64 {
        // Compute all edge lengths
        let edges: [(usize, usize); 12] = [
            (0, 1), (1, 2), (2, 3), (3, 0), // Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4), // Top face
            (0, 4), (1, 5), (2, 6), (3, 7), // Vertical edges
        ];

        let edge_lengths: Vec<f64> = edges
            .iter()
            .map(|&(i, j)| {
                let dx = nodes[j].x - nodes[i].x;
                let dy = nodes[j].y - nodes[i].y;
                let dz = nodes[j].z - nodes[i].z;
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .collect();

        let max_edge = edge_lengths.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_edge = edge_lengths.iter().cloned().fold(f64::INFINITY, f64::min);

        if min_edge > 1e-15 {
            max_edge / min_edge
        } else {
            f64::INFINITY
        }
    }

    /// Computes aspect ratio for a tet element.
    pub fn tet_aspect_ratio(nodes: &[Node; 4]) -> f64 {
        // Compute all edge lengths
        let edges: [(usize, usize); 6] = [
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
        ];

        let edge_lengths: Vec<f64> = edges
            .iter()
            .map(|&(i, j)| {
                let dx = nodes[j].x - nodes[i].x;
                let dy = nodes[j].y - nodes[i].y;
                let dz = nodes[j].z - nodes[i].z;
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .collect();

        let max_edge = edge_lengths.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_edge = edge_lengths.iter().cloned().fold(f64::INFINITY, f64::min);

        if min_edge > 1e-15 {
            max_edge / min_edge
        } else {
            f64::INFINITY
        }
    }

    /// Computes Jacobian determinant for a quad element at natural coordinates.
    pub fn quad_jacobian(nodes: &[Node; 4], xi: f64, eta: f64) -> f64 {
        // Shape function derivatives for 4-node quad
        let dndx = [
            -(1.0 - eta) / 4.0,
            (1.0 - eta) / 4.0,
            (1.0 + eta) / 4.0,
            -(1.0 + eta) / 4.0,
        ];
        let dndy = [
            -(1.0 - xi) / 4.0,
            -(1.0 + xi) / 4.0,
            (1.0 + xi) / 4.0,
            (1.0 - xi) / 4.0,
        ];

        // Jacobian matrix components
        let mut j00 = 0.0;
        let mut j01 = 0.0;
        let mut j10 = 0.0;
        let mut j11 = 0.0;

        for i in 0..4 {
            j00 += dndx[i] * nodes[i].x;
            j01 += dndx[i] * nodes[i].y;
            j10 += dndy[i] * nodes[i].x;
            j11 += dndy[i] * nodes[i].y;
        }

        // Include z-component for 3D
        if nodes.iter().any(|n| n.z.abs() > 1e-15) {
            let mut j02 = 0.0;
            let mut j12 = 0.0;
            for i in 0..4 {
                j02 += dndx[i] * nodes[i].z;
                j12 += dndy[i] * nodes[i].z;
            }
            // For 2D plane in 3D, compute magnitude of cross product
            let j20 = j01 * j12 - j02 * j11;
            let j21 = j02 * j10 - j00 * j12;
            let j22 = j00 * j11 - j01 * j10;
            (j20 * j20 + j21 * j21 + j22 * j22).sqrt()
        } else {
            // 2D case
            (j00 * j11 - j01 * j10).abs()
        }
    }

    /// Computes Jacobian determinant for a hex element at natural coordinates.
    pub fn hex_jacobian(nodes: &[Node; 8], xi: f64, eta: f64, zeta: f64) -> f64 {
        // Shape function derivatives for 8-node hex
        let dndx = [
            -(1.0 - eta) * (1.0 - zeta) / 8.0,
            (1.0 - eta) * (1.0 - zeta) / 8.0,
            (1.0 + eta) * (1.0 - zeta) / 8.0,
            -(1.0 + eta) * (1.0 - zeta) / 8.0,
            -(1.0 - eta) * (1.0 + zeta) / 8.0,
            (1.0 - eta) * (1.0 + zeta) / 8.0,
            (1.0 + eta) * (1.0 + zeta) / 8.0,
            -(1.0 + eta) * (1.0 + zeta) / 8.0,
        ];
        let dndy = [
            -(1.0 - xi) * (1.0 - zeta) / 8.0,
            -(1.0 + xi) * (1.0 - zeta) / 8.0,
            (1.0 + xi) * (1.0 - zeta) / 8.0,
            (1.0 - xi) * (1.0 - zeta) / 8.0,
            -(1.0 - xi) * (1.0 + zeta) / 8.0,
            -(1.0 + xi) * (1.0 + zeta) / 8.0,
            (1.0 + xi) * (1.0 + zeta) / 8.0,
            (1.0 - xi) * (1.0 + zeta) / 8.0,
        ];
        let dndz = [
            -(1.0 - xi) * (1.0 - eta) / 8.0,
            -(1.0 + xi) * (1.0 - eta) / 8.0,
            -(1.0 + xi) * (1.0 + eta) / 8.0,
            -(1.0 - xi) * (1.0 + eta) / 8.0,
            (1.0 - xi) * (1.0 - eta) / 8.0,
            (1.0 + xi) * (1.0 - eta) / 8.0,
            (1.0 + xi) * (1.0 + eta) / 8.0,
            (1.0 - xi) * (1.0 + eta) / 8.0,
        ];

        // Jacobian matrix components
        let mut j = [[0.0; 3]; 3];

        for i in 0..8 {
            j[0][0] += dndx[i] * nodes[i].x;
            j[0][1] += dndx[i] * nodes[i].y;
            j[0][2] += dndx[i] * nodes[i].z;
            j[1][0] += dndy[i] * nodes[i].x;
            j[1][1] += dndy[i] * nodes[i].y;
            j[1][2] += dndy[i] * nodes[i].z;
            j[2][0] += dndz[i] * nodes[i].x;
            j[2][1] += dndz[i] * nodes[i].y;
            j[2][2] += dndz[i] * nodes[i].z;
        }

        // Determinant of 3x3 matrix
        (j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
            - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
            + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]))
        .abs()
    }

    /// Checks mesh quality for quad elements.
    pub fn check_quad_mesh(nodes: &[Node], elements: &[[usize; 4]]) -> MeshQualityReport {
        if elements.is_empty() {
            return MeshQualityReport::new(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0, 0);
        }

        let mut min_ar = f64::INFINITY;
        let mut max_ar = 0.0_f64;
        let mut sum_ar = 0.0_f64;
        let mut min_jac = f64::INFINITY;
        let mut max_jac = 0.0_f64;
        let mut sum_jac = 0.0_f64;
        let mut high_ar_count = 0;
        let mut distorted_count = 0;

        for elem in elements {
            let elem_nodes = [
                nodes[elem[0]],
                nodes[elem[1]],
                nodes[elem[2]],
                nodes[elem[3]],
            ];

            let ar = quad_aspect_ratio(&elem_nodes);
            let jac = quad_jacobian(&elem_nodes, 0.0, 0.0); // Evaluate at center

            min_ar = min_ar.min(ar);
            max_ar = max_ar.max(ar);
            sum_ar += ar;

            min_jac = min_jac.min(jac);
            max_jac = max_jac.max(jac);
            sum_jac += jac;

            if ar > 10.0 {
                high_ar_count += 1;
            }
            if jac < 0.5 {
                distorted_count += 1;
            }
        }

        let n = elements.len() as f64;
        MeshQualityReport::new(
            min_ar, max_ar, sum_ar / n,
            min_jac, max_jac, sum_jac / n,
            high_ar_count, distorted_count,
        )
    }

    /// Checks mesh quality for hex elements.
    pub fn check_hex_mesh(nodes: &[Node], elements: &[[usize; 8]]) -> MeshQualityReport {
        if elements.is_empty() {
            return MeshQualityReport::new(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0, 0);
        }

        let mut min_ar = f64::INFINITY;
        let mut max_ar = 0.0_f64;
        let mut sum_ar = 0.0_f64;
        let mut min_jac = f64::INFINITY;
        let mut max_jac = 0.0_f64;
        let mut sum_jac = 0.0_f64;
        let mut high_ar_count = 0;
        let mut distorted_count = 0;

        // Integration points for Jacobian check
        let integration_points = [
            (-0.577, -0.577, -0.577),
            (0.577, -0.577, -0.577),
            (0.577, 0.577, -0.577),
            (-0.577, 0.577, -0.577),
            (-0.577, -0.577, 0.577),
            (0.577, -0.577, 0.577),
            (0.577, 0.577, 0.577),
            (-0.577, 0.577, 0.577),
        ];

        for elem in elements {
            let elem_nodes: [Node; 8] = [
                nodes[elem[0]], nodes[elem[1]], nodes[elem[2]], nodes[elem[3]],
                nodes[elem[4]], nodes[elem[5]], nodes[elem[6]], nodes[elem[7]],
            ];

            let ar = hex_aspect_ratio(&elem_nodes);
            min_ar = min_ar.min(ar);
            max_ar = max_ar.max(ar);
            sum_ar += ar;

            // Check Jacobian at integration points
            let mut elem_min_jac = f64::INFINITY;
            for (xi, eta, zeta) in integration_points.iter() {
                let jac = hex_jacobian(&elem_nodes, *xi, *eta, *zeta);
                elem_min_jac = elem_min_jac.min(jac);
                sum_jac += jac;
            }

            min_jac = min_jac.min(elem_min_jac);
            max_jac = max_jac.max(elem_min_jac);

            if ar > 10.0 {
                high_ar_count += 1;
            }
            if elem_min_jac < 0.5 {
                distorted_count += 1;
            }
        }

        let n = (elements.len() * integration_points.len()) as f64;
        MeshQualityReport::new(
            min_ar, max_ar, sum_ar / elements.len() as f64,
            min_jac, max_jac, sum_jac / n,
            high_ar_count, distorted_count,
        )
    }

    /// Checks mesh quality for tet elements.
    pub fn check_tet_mesh(nodes: &[Node], elements: &[[usize; 4]]) -> MeshQualityReport {
        if elements.is_empty() {
            return MeshQualityReport::new(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0, 0);
        }

        let mut min_ar = f64::INFINITY;
        let mut max_ar = 0.0_f64;
        let mut sum_ar = 0.0_f64;
        let mut min_jac = f64::INFINITY;
        let mut max_jac = 0.0_f64;
        let mut sum_jac = 0.0_f64;
        let mut high_ar_count = 0;
        let mut distorted_count = 0;

        for elem in elements {
            let elem_nodes = [
                nodes[elem[0]],
                nodes[elem[1]],
                nodes[elem[2]],
                nodes[elem[3]],
            ];

            let ar = tet_aspect_ratio(&elem_nodes);
            min_ar = min_ar.min(ar);
            max_ar = max_ar.max(ar);
            sum_ar += ar;

            // Jacobian for tet = 6 * volume
            let jac = tet_jacobian(&elem_nodes);
            min_jac = min_jac.min(jac);
            max_jac = max_jac.max(jac);
            sum_jac += jac;

            if ar > 10.0 {
                high_ar_count += 1;
            }
            if jac < 0.5 {
                distorted_count += 1;
            }
        }

        let n = elements.len() as f64;
        MeshQualityReport::new(
            min_ar, max_ar, sum_ar / n,
            min_jac, max_jac, sum_jac / n,
            high_ar_count, distorted_count,
        )
    }

    /// Computes Jacobian (6 * volume) for a tet element.
    pub fn tet_jacobian(nodes: &[Node; 4]) -> f64 {
        let v1 = [
            nodes[1].x - nodes[0].x,
            nodes[1].y - nodes[0].y,
            nodes[1].z - nodes[0].z,
        ];
        let v2 = [
            nodes[2].x - nodes[0].x,
            nodes[2].y - nodes[0].y,
            nodes[2].z - nodes[0].z,
        ];
        let v3 = [
            nodes[3].x - nodes[0].x,
            nodes[3].y - nodes[0].y,
            nodes[3].z - nodes[0].z,
        ];

        // Scalar triple product = 6 * volume
        let cross_yz = v2[1] * v3[2] - v2[2] * v3[1];
        let cross_xz = v2[2] * v3[0] - v2[0] * v3[2];
        let cross_xy = v2[0] * v3[1] - v2[1] * v3[0];

        (v1[0] * cross_yz + v1[1] * cross_xz + v1[2] * cross_xy).abs()
    }

    /// Checks warpage for quad elements.
    pub fn quad_warpage(nodes: &[Node; 4]) -> f64 {
        // Warpage = maximum deviation from plane
        // Compute plane from first 3 nodes
        let v1 = [
            nodes[1].x - nodes[0].x,
            nodes[1].y - nodes[0].y,
            nodes[1].z - nodes[0].z,
        ];
        let v2 = [
            nodes[2].x - nodes[0].x,
            nodes[2].y - nodes[0].y,
            nodes[2].z - nodes[0].z,
        ];

        // Normal vector
        let nx = v1[1] * v2[2] - v1[2] * v2[1];
        let ny = v1[2] * v2[0] - v1[0] * v2[2];
        let nz = v1[0] * v2[1] - v1[1] * v2[0];

        let norm = (nx * nx + ny * ny + nz * nz).sqrt();
        if norm < 1e-15 {
            return 0.0;
        }

        let (nx, ny, nz) = (nx / norm, ny / norm, nz / norm);

        // Distance of 4th node from plane
        let d = (nodes[3].x - nodes[0].x) * nx
            + (nodes[3].y - nodes[0].y) * ny
            + (nodes[3].z - nodes[0].z) * nz;

        d.abs()
    }
}

/// STL file import/export.
pub mod stl_io {
    use super::*;
    use std::str::FromStr;

    /// STL facet structure.
    #[derive(Debug, Clone)]
    pub struct StlFacet {
        pub normal: [f64; 3],
        pub vertices: [[f64; 3]; 3],
    }

    /// Imports ASCII STL file.
    pub fn import_ascii_stl<P: AsRef<Path>>(path: P) -> anyhow::Result<Vec<StlFacet>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut facets = Vec::new();

        let mut lines = reader.lines();
        while let Some(Ok(line)) = lines.next() {
            let trimmed = line.trim();
            if trimmed.starts_with("facet normal") {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 5 {
                    let normal = [
                        f64::from_str(parts[2]).unwrap_or(0.0),
                        f64::from_str(parts[3]).unwrap_or(0.0),
                        f64::from_str(parts[4]).unwrap_or(0.0),
                    ];

                    let mut vertices = [[0.0; 3]; 3];
                    for i in 0..3 {
                        if let Some(Ok(vertex_line)) = lines.next() {
                            let parts: Vec<&str> = vertex_line.trim().split_whitespace().collect();
                            if parts.len() >= 4 {
                                vertices[i] = [
                                    f64::from_str(parts[1]).unwrap_or(0.0),
                                    f64::from_str(parts[2]).unwrap_or(0.0),
                                    f64::from_str(parts[3]).unwrap_or(0.0),
                                ];
                            }
                        }
                    }

                    facets.push(StlFacet { normal, vertices });

                    // Skip endloop and endfacet
                    lines.next();
                    lines.next();
                }
            }
        }

        Ok(facets)
    }

    /// Exports to ASCII STL file.
    pub fn export_ascii_stl<P: AsRef<Path>>(
        path: P,
        facets: &[StlFacet],
    ) -> anyhow::Result<()> {
        let mut file = File::create(path)?;

        writeln!(file, "solid exported")?;
        for facet in facets {
            writeln!(
                file,
                "  facet normal {} {} {}",
                facet.normal[0], facet.normal[1], facet.normal[2]
            )?;
            writeln!(file, "    outer loop")?;
            for vertex in &facet.vertices {
                writeln!(
                    file,
                    "      vertex {} {} {}",
                    vertex[0], vertex[1], vertex[2]
                )?;
            }
            writeln!(file, "    endloop")?;
            writeln!(file, "  endfacet")?;
        }
        writeln!(file, "endsolid exported")?;

        Ok(())
    }

    /// Converts STL facets to mesh nodes and elements.
    pub fn stl_to_mesh(facets: &[StlFacet]) -> (Vec<Node>, Vec<[usize; 3]>) {
        let mut nodes = Vec::new();
        let mut elements = Vec::new();
        let mut node_map: HashMap<[i32; 3], usize> = HashMap::new();

        // Tolerance for node merging
        let tol = 1e-6;

        for facet in facets {
            let mut facet_nodes = [0usize; 3];

            for (i, vertex) in facet.vertices.iter().enumerate() {
                // Quantize coordinates for merging
                let key = [
                    (vertex[0] / tol) as i32,
                    (vertex[1] / tol) as i32,
                    (vertex[2] / tol) as i32,
                ];

                let node_id = match node_map.get(&key) {
                    Some(&id) => id,
                    None => {
                        let id = nodes.len();
                        nodes.push(Node::new_3d(vertex[0], vertex[1], vertex[2]));
                        node_map.insert(key, id);
                        id
                    }
                };

                facet_nodes[i] = node_id;
            }

            elements.push(facet_nodes);
        }

        (nodes, elements)
    }
}

/// VTK file export for visualization.
pub mod vtk_io {
    use super::*;

    /// Exports mesh to VTK format.
    pub fn export_vtk_mesh<P: AsRef<Path>>(
        path: P,
        nodes: &[Node],
        elements: &[(usize, usize)],
    ) -> anyhow::Result<()> {
        let mut file = File::create(path)?;

        writeln!(file, "# vtk DataFile Version 3.0")?;
        writeln!(file, "FEA Mesh")?;
        writeln!(file, "ASCII")?;
        writeln!(file, "DATASET UNSTRUCTURED_GRID")?;
        writeln!(file)?;

        // Points
        writeln!(file, "POINTS {} float", nodes.len())?;
        for node in nodes {
            let coords = [node.x, node.y, node.z];
            writeln!(file, "{} {} {}", coords[0], coords[1], coords[2])?;
        }
        writeln!(file)?;

        // Cells (truss elements as lines)
        writeln!(file, "CELLS {} {}", elements.len(), elements.len() * 3)?;
        for (n0, n1) in elements {
            writeln!(file, "2 {} {}", n0, n1)?;
        }
        writeln!(file)?;

        // Cell types (3 = line)
        writeln!(file, "CELL_TYPES {}", elements.len())?;
        for _ in elements {
            writeln!(file, "3")?;
        }

        Ok(())
    }

    /// Exports displacement results to VTK format.
    pub fn export_vtk_displacements<P: AsRef<Path>>(
        path: P,
        nodes: &[Node],
        displacements: &[f64],
    ) -> anyhow::Result<()> {
        let mut file = File::create(path)?;

        writeln!(file, "# vtk DataFile Version 3.0")?;
        writeln!(file, "FEA Displacement Results")?;
        writeln!(file, "ASCII")?;
        writeln!(file, "DATASET UNSTRUCTURED_GRID")?;
        writeln!(file)?;

        // Points (deformed)
        writeln!(file, "POINTS {} float", nodes.len())?;
        for (i, node) in nodes.iter().enumerate() {
            let orig = [node.x, node.y, node.z];

            let dof_per_node = 3; // Assume 3D
            let dx = if i * dof_per_node < displacements.len() {
                displacements[i * dof_per_node]
            } else {
                0.0
            };
            let dy = if i * dof_per_node + 1 < displacements.len() {
                displacements[i * dof_per_node + 1]
            } else {
                0.0
            };
            let dz = if i * dof_per_node + 2 < displacements.len() {
                displacements[i * dof_per_node + 2]
            } else {
                0.0
            };

            writeln!(
                file,
                "{} {} {}",
                orig[0] + dx, orig[1] + dy, orig[2] + dz
            )?;
        }
        writeln!(file)?;

        // Displacement vectors
        writeln!(file, "POINT_DATA {}", nodes.len())?;
        writeln!(file, "VECTORS displacements float")?;
        for (i, _node) in nodes.iter().enumerate() {
            let dof_per_node = 3; // Assume 3D
            let dx = if i * dof_per_node < displacements.len() {
                displacements[i * dof_per_node]
            } else {
                0.0
            };
            let dy = if i * dof_per_node + 1 < displacements.len() {
                displacements[i * dof_per_node + 1]
            } else {
                0.0
            };
            let dz = if i * dof_per_node + 2 < displacements.len() {
                displacements[i * dof_per_node + 2]
            } else {
                0.0
            };
            writeln!(file, "{} {} {}", dx, dy, dz)?;
        }

        // Displacement magnitude
        writeln!(file, "SCALARS displacement_magnitude float")?;
        writeln!(file, "LOOKUP_TABLE default")?;
        for (i, _node) in nodes.iter().enumerate() {
            let dof_per_node = 3;
            let dx = if i * dof_per_node < displacements.len() {
                displacements[i * dof_per_node]
            } else {
                0.0
            };
            let dy = if i * dof_per_node + 1 < displacements.len() {
                displacements[i * dof_per_node + 1]
            } else {
                0.0
            };
            let dz = if i * dof_per_node + 2 < displacements.len() {
                displacements[i * dof_per_node + 2]
            } else {
                0.0
            };
            let mag = (dx * dx + dy * dy + dz * dz).sqrt();
            writeln!(file, "{}", mag)?;
        }

        Ok(())
    }

    /// Exports scalar field to VTK format.
    pub fn export_vtk_scalar<P: AsRef<Path>, T: AsRef<[f64]>>(
        path: P,
        nodes: &[Node],
        scalar_name: &str,
        values: T,
    ) -> anyhow::Result<()> {
        let values = values.as_ref();
        let mut file = File::create(path)?;

        writeln!(file, "# vtk DataFile Version 3.0")?;
        writeln!(file, "FEA {} Results", scalar_name)?;
        writeln!(file, "ASCII")?;
        writeln!(file, "DATASET UNSTRUCTURED_GRID")?;
        writeln!(file)?;

        // Points
        writeln!(file, "POINTS {} float", nodes.len())?;
        for node in nodes {
            let coords = [node.x, node.y, node.z];
            writeln!(file, "{} {} {}", coords[0], coords[1], coords[2])?;
        }
        writeln!(file)?;

        // Scalar data
        writeln!(file, "POINT_DATA {}", nodes.len())?;
        writeln!(file, "SCALARS {} float", scalar_name)?;
        writeln!(file, "LOOKUP_TABLE default")?;
        for &value in values {
            writeln!(file, "{}", value)?;
        }

        Ok(())
    }
}

/// Boundary condition helpers.
pub mod bc_helpers {
    use super::*;

    /// Applies fixed boundary conditions to all DOFs of specified nodes.
    pub fn fix_all_dofs<E>(model: &mut Model<E>, node_ids: &[usize]) {
        for &node_id in node_ids {
            for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
                model.add_bc(BoundaryCondition::fixed(node_id, dof));
            }
        }
    }

    /// Applies fixed boundary conditions to a face of a 3D model.
    pub fn fix_face_3d<E>(model: &mut Model<E>, x: Option<f64>, y: Option<f64>, z: Option<f64>, tolerance: f64) {
        let mut nodes_to_fix = Vec::new();

        for (node_id, node) in model.nodes.iter().enumerate() {
            let coords = (node.x, node.y, node.z);

            let mut should_fix = false;
            if let Some(x_val) = x {
                if (coords.0 - x_val).abs() < tolerance {
                    should_fix = true;
                }
            }
            if let Some(y_val) = y {
                if (coords.1 - y_val).abs() < tolerance {
                    should_fix = true;
                }
            }
            if let Some(z_val) = z {
                if (coords.2 - z_val).abs() < tolerance {
                    should_fix = true;
                }
            }

            if should_fix {
                nodes_to_fix.push(node_id);
            }
        }

        for node_id in nodes_to_fix {
            fix_all_dofs(model, &[node_id]);
        }
    }

    /// Applies a distributed load to a face.
    pub fn apply_distributed_load<E>(
        model: &mut Model<E>,
        direction: Dof,
        magnitude: f64,
        x: Option<f64>,
        y: Option<f64>,
        z: Option<f64>,
        tolerance: f64,
    ) {
        let mut nodes_to_load = Vec::new();

        for (node_id, node) in model.nodes.iter().enumerate() {
            let coords = (node.x, node.y, node.z);

            let mut should_apply = false;
            if let Some(x_val) = x {
                if (coords.0 - x_val).abs() < tolerance {
                    should_apply = true;
                }
            }
            if let Some(y_val) = y {
                if (coords.1 - y_val).abs() < tolerance {
                    should_apply = true;
                }
            }
            if let Some(z_val) = z {
                if (coords.2 - z_val).abs() < tolerance {
                    should_apply = true;
                }
            }

            if should_apply {
                nodes_to_load.push(node_id);
            }
        }

        for node_id in nodes_to_load {
            model.add_load(crate::core::Load::new(node_id, direction, magnitude));
        }
    }

    /// Applies symmetry boundary conditions.
    pub fn apply_symmetry_x<E>(model: &mut Model<E>, x: f64, tolerance: f64) {
        let mut nodes_to_fix = Vec::new();

        for (node_id, node) in model.nodes.iter().enumerate() {
            let node_x = node.x;

            if (node_x - x).abs() < tolerance {
                nodes_to_fix.push(node_id);
            }
        }

        for node_id in nodes_to_fix {
            model.add_bc(BoundaryCondition::fixed(node_id, Dof::Ux));
        }
    }
}

/// Material assignment helpers.
pub mod material_helpers {
    use super::*;

    /// Creates a steel material.
    pub fn steel_a36() -> Material {
        Material {
            name: "A36 Steel",
            young_modulus: 200e9,
            poisson_ratio: 0.26,
            density: 7850.0,
            yield_strength: 250e6,
        }
    }

    /// Creates an aluminum material.
    pub fn aluminum_6061() -> Material {
        Material {
            name: "6061 Aluminum",
            young_modulus: 68.9e9,
            poisson_ratio: 0.33,
            density: 2700.0,
            yield_strength: 276e6,
        }
    }

    /// Creates a titanium material.
    pub fn titanium_ti64() -> Material {
        Material {
            name: "Ti-6Al-4V",
            young_modulus: 113.8e9,
            poisson_ratio: 0.342,
            density: 4430.0,
            yield_strength: 880e6,
        }
    }

    /// Creates a concrete material.
    pub fn concrete_normal() -> Material {
        Material {
            name: "Normal Concrete",
            young_modulus: 25e9,
            poisson_ratio: 0.2,
            density: 2400.0,
            yield_strength: 30e6,
        }
    }
}

/// Advanced pre-processing utilities.
pub mod advanced;

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_generation::*;

    #[test]
    fn test_bar_1d_generation() {
        let (nodes, elements) = generate_bar_1d(10.0, 5, 0.01);
        assert_eq!(nodes.len(), 6);
        assert_eq!(elements.len(), 5);
    }

    #[test]
    fn test_rect_2d_generation() {
        let (nodes, elements) = generate_rect_2d(1.0, 1.0, 4, 4);
        assert_eq!(nodes.len(), 25);
        assert_eq!(elements.len(), 16);
    }

    #[test]
    fn test_tri_2d_generation() {
        let (nodes, elements) = generate_tri_2d_from_rect(1.0, 1.0, 4, 4);
        assert_eq!(nodes.len(), 25);
        assert_eq!(elements.len(), 32); // 16 quads * 2 triangles
    }

    #[test]
    fn test_box_3d_generation() {
        let (nodes, elements) = generate_box_3d(1.0, 1.0, 1.0, 4, 4, 4);
        assert_eq!(nodes.len(), 125);
        assert_eq!(elements.len(), 64);
    }

    #[test]
    fn test_tet_3d_generation() {
        let (nodes, elements) = generate_tet_3d_from_hex(1.0, 1.0, 1.0, 2, 2, 2);
        assert_eq!(nodes.len(), 27);
        assert_eq!(elements.len(), 40); // 8 hexes * 5 tets
    }

    #[test]
    fn test_bc_helpers() {
        use bc_helpers::*;

        let mut model: Model<Truss2> = Model::new();
        model.add_node(Node::new_3d(0.0, 0.0, 0.0));
        model.add_node(Node::new_3d(1.0, 0.0, 0.0));

        fix_all_dofs(&mut model, &[0]);

        // Should have 3 BCs (Ux, Uy, Uz fixed)
        assert_eq!(model.bcs.len(), 3);
    }

    #[test]
    fn test_material_helpers() {
        use material_helpers::*;

        let steel = steel_a36();
        assert!((steel.young_modulus - 200e9).abs() < 1e6);
        assert!((steel.density - 7850.0).abs() < 1.0);
    }
}

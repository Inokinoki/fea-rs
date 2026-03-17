//! Plate elements for 2D structural analysis.
#![allow(unused_variables)]
//!
//! This module provides:
//! - Plate4: 4-node quadrilateral plate element (plane stress/strain)
//! - Plate8: 8-node serendipity plate element (plane stress/strain)
//! - KirchhoffPlate4: 4-node Kirchhoff thin plate bending element
//! - MindlinPlate4: 4-node Mindlin thick plate bending element

use super::{Element, ElementContext};
use crate::core::NodeId;
use nalgebra::{DMatrix, Matrix2, Matrix3};

/// A 4-node quadrilateral plate element for plane stress/strain analysis.
///
/// Each node has 2 DOFs (ux, uy).
#[derive(Debug, Clone)]
pub struct Plate4 {
    pub nodes: [NodeId; 4],
    pub thickness: f64,
    pub plane_strain: bool,
}

impl Plate4 {
    /// Creates a new Plate4 element.
    pub fn new(n1: NodeId, n2: NodeId, n3: NodeId, n4: NodeId, thickness: f64) -> Self {
        Self {
            nodes: [n1, n2, n3, n4],
            thickness,
            plane_strain: false,
        }
    }

    /// Sets whether this is a plane strain element (default is plane stress).
    pub fn with_plane_strain(mut self, plane_strain: bool) -> Self {
        self.plane_strain = plane_strain;
        self
    }

    /// Computes the element stiffness matrix using 2x2 Gauss quadrature.
    fn compute_stiffness(&self, ctx: &ElementContext) -> DMatrix<f64> {
        let e = ctx.material.young_modulus;
        let nu = ctx.material.poisson_ratio;
        let t = self.thickness;

        // Constitutive matrix (plane stress or plane strain)
        let d = if self.plane_strain {
            let factor = e / ((1.0 + nu) * (1.0 - 2.0 * nu));
            DMatrix::from_row_slice(3, 3, &[
                1.0 - nu, nu, 0.0,
                nu, 1.0 - nu, 0.0,
                0.0, 0.0, (1.0 - 2.0 * nu) / 2.0,
            ]) * factor
        } else {
            // Plane stress
            let factor = e / (1.0 - nu * nu);
            DMatrix::from_row_slice(3, 3, &[
                1.0, nu, 0.0,
                nu, 1.0, 0.0,
                0.0, 0.0, (1.0 - nu) / 2.0,
            ]) * factor
        };

        // 2x2 Gauss quadrature
        let gauss_points = [
            (-1.0 / 3.0f64.sqrt(), -1.0 / 3.0f64.sqrt(), 1.0),
            (1.0 / 3.0f64.sqrt(), -1.0 / 3.0f64.sqrt(), 1.0),
            (1.0 / 3.0f64.sqrt(), 1.0 / 3.0f64.sqrt(), 1.0),
            (-1.0 / 3.0f64.sqrt(), 1.0 / 3.0f64.sqrt(), 1.0),
        ];

        let mut ke = DMatrix::zeros(8, 8);

        for (xi, eta, w) in &gauss_points {
            // Compute shape function derivatives
            let (dndx, dndy) = self.shape_function_derivatives(*xi, *eta);

            // Compute Jacobian
            let mut j = DMatrix::zeros(2, 2);
            for i in 0..4 {
                let node = ctx.nodes[self.nodes[i]];
                j[(0, 0)] += dndx[i] * node.x;
                j[(0, 1)] += dndx[i] * node.y;
                j[(1, 0)] += dndy[i] * node.x;
                j[(1, 1)] += dndy[i] * node.y;
            }

            let det_j = j.determinant();
            if det_j <= 0.0 {
                continue; // Invalid element
            }

            let j_inv = j.try_inverse().unwrap_or(DMatrix::identity(2, 2));

            // Compute B matrix (strain-displacement)
            let b = self.compute_b_matrix(&j_inv, &dndx, &dndy);

            // K = integral(B^T * D * B * t * det(J) * w)
            let bt = b.transpose();
            let dbt = &d * &b;
            let integrand = &bt * &dbt * (t * det_j * w);

            ke += integrand;
        }

        ke
    }

    /// Computes shape function derivatives with respect to natural coordinates.
    fn shape_function_derivatives(&self, xi: f64, eta: f64) -> (Vec<f64>, Vec<f64>) {
        let dndx = vec![
            -0.25 * (1.0 - eta),  // Node 1
            0.25 * (1.0 - eta),   // Node 2
            0.25 * (1.0 + eta),   // Node 3
            -0.25 * (1.0 + eta),  // Node 4
        ];

        let dndy = vec![
            -0.25 * (1.0 - xi),   // Node 1
            -0.25 * (1.0 + xi),   // Node 2
            0.25 * (1.0 + xi),    // Node 3
            0.25 * (1.0 - xi),    // Node 4
        ];

        (dndx, dndy)
    }

    /// Computes the B matrix (strain-displacement matrix).
    fn compute_b_matrix(&self, j_inv: &DMatrix<f64>, dndx: &[f64], dndy: &[f64]) -> DMatrix<f64> {
        let mut b = DMatrix::zeros(3, 8);

        for i in 0..4 {
            // Compute derivatives in global coordinates
            let dndx_global = j_inv[(0, 0)] * dndx[i] + j_inv[(0, 1)] * dndy[i];
            let dndy_global = j_inv[(1, 0)] * dndx[i] + j_inv[(1, 1)] * dndy[i];

            let col = i * 2;
            b[(0, col)] = dndx_global;        // dN/dx for ux
            b[(1, col + 1)] = dndy_global;    // dN/dy for uy
            b[(2, col)] = dndy_global;        // dN/dy for ux
            b[(2, col + 1)] = dndx_global;    // dN/dx for uy
        }

        b
    }
}

impl Element for Plate4 {
    fn node_ids(&self) -> Vec<NodeId> {
        self.nodes.to_vec()
    }

    fn ndofs(&self) -> usize {
        8 // 4 nodes * 2 DOFs each (ux, uy)
    }

    fn stiffness(&self, ctx: &ElementContext) -> DMatrix<f64> {
        self.compute_stiffness(ctx)
    }

    fn mass(&self, ctx: &ElementContext) -> Option<DMatrix<f64>> {
        let rho = ctx.material.density;
        let t = self.thickness;

        // Lumped mass matrix using 2x2 Gauss quadrature
        let gauss_points = [
            (-1.0 / 3.0f64.sqrt(), -1.0 / 3.0f64.sqrt(), 1.0),
            (1.0 / 3.0f64.sqrt(), -1.0 / 3.0f64.sqrt(), 1.0),
            (1.0 / 3.0f64.sqrt(), 1.0 / 3.0f64.sqrt(), 1.0),
            (-1.0 / 3.0f64.sqrt(), 1.0 / 3.0f64.sqrt(), 1.0),
        ];

        let mut total_mass = 0.0;
        for (xi, eta, w) in &gauss_points {
            let det_j = self.jacobian_determinant(*xi, *eta, ctx);
            if det_j > 0.0 {
                total_mass += rho * t * det_j * w;
            }
        }

        // Lumped mass: distribute equally to each DOF
        let mut me = DMatrix::zeros(8, 8);
        let mass_per_dof = total_mass / 8.0;
        for i in 0..8 {
            me[(i, i)] = mass_per_dof;
        }

        Some(me)
    }
}

impl Plate4 {
    /// Computes the Jacobian determinant at a given natural coordinate.
    fn jacobian_determinant(&self, xi: f64, eta: f64, ctx: &ElementContext) -> f64 {
        let (dndx, dndy) = self.shape_function_derivatives(xi, eta);

        let mut j = DMatrix::zeros(2, 2);
        for i in 0..4 {
            let node = ctx.nodes[self.nodes[i]];
            j[(0, 0)] += dndx[i] * node.x;
            j[(0, 1)] += dndx[i] * node.y;
            j[(1, 0)] += dndy[i] * node.x;
            j[(1, 1)] += dndy[i] * node.y;
        }

        j.determinant()
    }
}

/// An 8-node serendipity plate element for plane stress/strain analysis.
///
/// Each node has 2 DOFs (ux, uy). Mid-side nodes provide quadratic interpolation.
#[derive(Debug, Clone)]
pub struct Plate8 {
    pub nodes: [NodeId; 8],
    pub thickness: f64,
    pub plane_strain: bool,
}

impl Plate8 {
    /// Creates a new Plate8 element.
    pub fn new(nodes: [NodeId; 8], thickness: f64) -> Self {
        Self {
            nodes,
            thickness,
            plane_strain: false,
        }
    }

    /// Sets whether this is a plane strain element.
    pub fn with_plane_strain(mut self, plane_strain: bool) -> Self {
        self.plane_strain = plane_strain;
        self
    }

    /// Computes shape functions and their derivatives for 8-node serendipity element.
    fn shape_functions(&self, xi: f64, eta: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut n = vec![0.0; 8];
        let mut dndx = vec![0.0; 8];
        let mut dndy = vec![0.0; 8];

        // Corner nodes (1-4)
        for (i, xi_i, eta_i) in [(0, -1.0, -1.0), (1, 1.0, -1.0), (2, 1.0, 1.0), (3, -1.0, 1.0)] {
            let xi_eta = xi * xi_i;
            let eta_eta = eta * eta_i;
            n[i] = 0.25 * (1.0 + xi_eta) * (1.0 + eta_eta) * (xi_eta + eta_eta - 1.0);
            dndx[i] = 0.25 * xi_i * (1.0 + eta_eta) * (2.0 * xi_eta + eta_eta);
            dndy[i] = 0.25 * eta_i * (1.0 + xi_eta) * (2.0 * eta_eta + xi_eta);
        }

        // Mid-side nodes (5-8)
        // Node 5: xi = 0, eta = -1
        n[4] = 0.5 * (1.0 - xi * xi) * (1.0 - eta);
        dndx[4] = -xi * (1.0 - eta);
        dndy[4] = -0.5 * (1.0 - xi * xi);

        // Node 6: xi = 1, eta = 0
        n[5] = 0.5 * (1.0 + xi) * (1.0 - eta * eta);
        dndx[5] = 0.5 * (1.0 - eta * eta);
        dndy[5] = -eta * (1.0 + xi);

        // Node 7: xi = 0, eta = 1
        n[6] = 0.5 * (1.0 - xi * xi) * (1.0 + eta);
        dndx[6] = -xi * (1.0 + eta);
        dndy[6] = 0.5 * (1.0 - xi * xi);

        // Node 8: xi = -1, eta = 0
        n[7] = 0.5 * (1.0 - xi) * (1.0 - eta * eta);
        dndx[7] = -0.5 * (1.0 - eta * eta);
        dndy[7] = -eta * (1.0 - xi);

        (n, dndx, dndy)
    }

    /// Computes the element stiffness matrix using 3x3 Gauss quadrature.
    fn compute_stiffness(&self, ctx: &ElementContext) -> DMatrix<f64> {
        let e = ctx.material.young_modulus;
        let nu = ctx.material.poisson_ratio;
        let t = self.thickness;

        // Constitutive matrix
        let d = if self.plane_strain {
            let factor = e / ((1.0 + nu) * (1.0 - 2.0 * nu));
            DMatrix::from_row_slice(3, 3, &[
                1.0 - nu, nu, 0.0,
                nu, 1.0 - nu, 0.0,
                0.0, 0.0, (1.0 - 2.0 * nu) / 2.0,
            ]) * factor
        } else {
            let factor = e / (1.0 - nu * nu);
            DMatrix::from_row_slice(3, 3, &[
                1.0, nu, 0.0,
                nu, 1.0, 0.0,
                0.0, 0.0, (1.0 - nu) / 2.0,
            ]) * factor
        };

        // 3x3 Gauss quadrature points
        let sqrt_3_5: f64 = (3.0_f64 / 5.0_f64).sqrt();
        let gauss_points = [
            (-sqrt_3_5, -sqrt_3_5, 5.0 / 9.0),
            (0.0, -sqrt_3_5, 8.0 / 9.0),
            (sqrt_3_5, -sqrt_3_5, 5.0 / 9.0),
            (-sqrt_3_5, 0.0, 8.0 / 9.0),
            (0.0, 0.0, 64.0 / 81.0),
            (sqrt_3_5, 0.0, 8.0 / 9.0),
            (-sqrt_3_5, sqrt_3_5, 5.0 / 9.0),
            (0.0, sqrt_3_5, 8.0 / 9.0),
            (sqrt_3_5, sqrt_3_5, 5.0 / 9.0),
        ];

        let mut ke = DMatrix::zeros(16, 16);

        for (xi, eta, w) in &gauss_points {
            let (_, dndx, dndy) = self.shape_functions(*xi, *eta);

            // Compute Jacobian
            let mut j = DMatrix::zeros(2, 2);
            for i in 0..8 {
                let node = ctx.nodes[self.nodes[i]];
                j[(0, 0)] += dndx[i] * node.x;
                j[(0, 1)] += dndx[i] * node.y;
                j[(1, 0)] += dndy[i] * node.x;
                j[(1, 1)] += dndy[i] * node.y;
            }

            let det_j = j.determinant();
            if det_j <= 0.0 {
                continue;
            }

            let j_inv = j.try_inverse().unwrap_or(DMatrix::identity(2, 2));
            let b = self.compute_b_matrix(&j_inv, &dndx, &dndy);

            let bt = b.transpose();
            let dbt = &d * &b;
            let integrand = &bt * &dbt * (t * det_j * w);

            ke += integrand;
        }

        ke
    }

    /// Computes the B matrix for Plate8.
    fn compute_b_matrix(&self, j_inv: &DMatrix<f64>, dndx: &[f64], dndy: &[f64]) -> DMatrix<f64> {
        let mut b = DMatrix::zeros(3, 16);

        for i in 0..8 {
            let dndx_global = j_inv[(0, 0)] * dndx[i] + j_inv[(0, 1)] * dndy[i];
            let dndy_global = j_inv[(1, 0)] * dndx[i] + j_inv[(1, 1)] * dndy[i];

            let col = i * 2;
            b[(0, col)] = dndx_global;
            b[(1, col + 1)] = dndy_global;
            b[(2, col)] = dndy_global;
            b[(2, col + 1)] = dndx_global;
        }

        b
    }
}

impl Element for Plate8 {
    fn node_ids(&self) -> Vec<NodeId> {
        self.nodes.to_vec()
    }

    fn ndofs(&self) -> usize {
        16 // 8 nodes * 2 DOFs each
    }

    fn stiffness(&self, ctx: &ElementContext) -> DMatrix<f64> {
        self.compute_stiffness(ctx)
    }

    fn mass(&self, ctx: &ElementContext) -> Option<DMatrix<f64>> {
        let rho = ctx.material.density;
        let t = self.thickness;

        // Approximate area using corner nodes only
        let n0 = ctx.nodes[self.nodes[0]];
        let n1 = ctx.nodes[self.nodes[1]];
        let n2 = ctx.nodes[self.nodes[2]];

        // Shoelace formula for quadrilateral area
        let area = 0.5 * ((n1.x - n0.x) * (n2.y - n0.y) - (n2.x - n0.x) * (n1.y - n0.y)).abs();
        let total_mass = rho * t * area;

        // Lumped mass: distribute equally to each DOF
        let mut me = DMatrix::zeros(16, 16);
        let mass_per_dof = total_mass / 16.0;
        for i in 0..16 {
            me[(i, i)] = mass_per_dof;
        }

        Some(me)
    }
}

// ============================================================================
// Plate Bending Elements
// ============================================================================

/// DOFs per node for plate bending (w, θx, θy)
pub const PLATE_BENDING_DOFS: usize = 3;

/// Kirchhoff-Love thin plate bending element (4-node quadrilateral).
#[derive(Debug, Clone)]
pub struct KirchhoffPlate4 {
    pub nodes: [NodeId; 4],
    pub id: usize,
    pub thickness: f64,
}

impl KirchhoffPlate4 {
    pub fn new(n0: NodeId, n1: NodeId, n2: NodeId, n3: NodeId, thickness: f64) -> Self {
        Self { nodes: [n0, n1, n2, n3], id: 0, thickness }
    }

    fn constitutive_matrix(&self, e: f64, nu: f64) -> Matrix3<f64> {
        let d = e * self.thickness.powi(3) / (12.0 * (1.0 - nu * nu));
        Matrix3::from_row_slice(&[d, d * nu, 0.0, d * nu, d, 0.0, 0.0, 0.0, d * (1.0 - nu) / 2.0])
    }

    fn get_dimensions(&self, ctx: &ElementContext) -> (f64, f64) {
        let n0 = ctx.nodes[self.nodes[0]];
        let n1 = ctx.nodes[self.nodes[1]];
        let n3 = ctx.nodes[self.nodes[3]];
        ((n1.x - n0.x).abs().max(0.01), (n3.y - n0.y).abs().max(0.01))
    }
}

impl Element for KirchhoffPlate4 {
    fn node_ids(&self) -> Vec<NodeId> { self.nodes.to_vec() }
    fn ndofs(&self) -> usize { 4 * PLATE_BENDING_DOFS }

    fn stiffness(&self, ctx: &ElementContext) -> DMatrix<f64> {
        let e = ctx.material.young_modulus;
        let nu = ctx.material.poisson_ratio;
        let d = self.constitutive_matrix(e, nu);
        let (x, y) = self.get_dimensions(ctx);
        let area = x * y;

        let gp = 1.0 / 3.0_f64.sqrt();
        let gauss_points = [(-gp, -gp, 1.0), (gp, -gp, 1.0), (gp, gp, 1.0), (-gp, gp, 1.0)];

        let ndofs = self.ndofs();
        let mut ke = DMatrix::zeros(ndofs, ndofs);

        for (xi, eta, w) in &gauss_points {
            let mut b = DMatrix::zeros(3, ndofs);
            for i in 0..4 {
                let dof = i * PLATE_BENDING_DOFS;
                b[(0, dof + 1)] = -xi / x;
                b[(1, dof + 2)] = eta / y;
                b[(2, dof + 1)] = -eta / (2.0 * y);
                b[(2, dof + 2)] = xi / (2.0 * x);
            }
            ke += b.transpose() * &d * &b * (w * area / 4.0);
        }
        ke
    }

    fn mass(&self, ctx: &ElementContext) -> Option<DMatrix<f64>> {
        let rho = ctx.material.density;
        let (x, y) = self.get_dimensions(ctx);
        let total_mass = rho * self.thickness * x * y;

        let ndofs = self.ndofs();
        let mut me = DMatrix::zeros(ndofs, ndofs);
        let mass_per_node = total_mass / 4.0;
        for i in 0..4 {
            me[(i * PLATE_BENDING_DOFS, i * PLATE_BENDING_DOFS)] = mass_per_node;
        }
        Some(me)
    }
}

/// Mindlin-Reissner thick plate bending element.
#[derive(Debug, Clone)]
pub struct MindlinPlate4 {
    pub nodes: [NodeId; 4],
    pub id: usize,
    pub thickness: f64,
    pub shear_correction: f64,
}

impl MindlinPlate4 {
    pub fn new(n0: NodeId, n1: NodeId, n2: NodeId, n3: NodeId, thickness: f64) -> Self {
        Self { nodes: [n0, n1, n2, n3], id: 0, thickness, shear_correction: 5.0 / 6.0 }
    }

    pub fn with_shear_correction(n0: NodeId, n1: NodeId, n2: NodeId, n3: NodeId, thickness: f64, k: f64) -> Self {
        Self { nodes: [n0, n1, n2, n3], id: 0, thickness, shear_correction: k.clamp(0.5, 1.0) }
    }

    fn bending_matrix(&self, e: f64, nu: f64) -> Matrix3<f64> {
        let d = e * self.thickness.powi(3) / (12.0 * (1.0 - nu * nu));
        Matrix3::from_row_slice(&[d, d * nu, 0.0, d * nu, d, 0.0, 0.0, 0.0, d * (1.0 - nu) / 2.0])
    }

    fn shear_matrix(&self, e: f64, nu: f64) -> Matrix2<f64> {
        let g = e / (2.0 * (1.0 + nu));
        let gs = self.shear_correction * g * self.thickness;
        Matrix2::new(gs, 0.0, 0.0, gs)
    }

    fn get_dimensions(&self, ctx: &ElementContext) -> (f64, f64) {
        let n0 = ctx.nodes[self.nodes[0]];
        let n1 = ctx.nodes[self.nodes[1]];
        let n3 = ctx.nodes[self.nodes[3]];
        ((n1.x - n0.x).abs().max(0.01), (n3.y - n0.y).abs().max(0.01))
    }
}

impl Element for MindlinPlate4 {
    fn node_ids(&self) -> Vec<NodeId> { self.nodes.to_vec() }
    fn ndofs(&self) -> usize { 4 * PLATE_BENDING_DOFS }

    fn stiffness(&self, ctx: &ElementContext) -> DMatrix<f64> {
        let e = ctx.material.young_modulus;
        let nu = ctx.material.poisson_ratio;
        let db = self.bending_matrix(e, nu);
        let ds = self.shear_matrix(e, nu);
        let (x, y) = self.get_dimensions(ctx);
        let area = x * y;

        let gp = 1.0 / 3.0_f64.sqrt();
        let gauss_points = [(-gp, -gp, 1.0), (gp, -gp, 1.0), (gp, gp, 1.0), (-gp, gp, 1.0)];

        let ndofs = self.ndofs();
        let mut ke = DMatrix::zeros(ndofs, ndofs);

        for (xi, eta, w) in &gauss_points {
            let mut bb = DMatrix::zeros(3, ndofs);
            let mut bs = DMatrix::zeros(2, ndofs);
            for i in 0..4 {
                let dof = i * PLATE_BENDING_DOFS;
                let xinv = 1.0 / x;
                let yinv = 1.0 / y;
                bb[(0, dof + 1)] = if *xi > 0.0 { xinv } else { -xinv };
                bb[(1, dof + 2)] = if *eta > 0.0 { -yinv } else { yinv };
                bs[(0, dof)] = if *xi > 0.0 { xinv } else { -xinv };
                bs[(1, dof)] = if *eta > 0.0 { yinv } else { -yinv };
            }
            ke += bb.transpose() * &db * &bb * (w * area / 4.0);
            ke += bs.transpose() * &ds * &bs * (w * area / 4.0);
        }
        ke
    }

    fn mass(&self, ctx: &ElementContext) -> Option<DMatrix<f64>> {
        let rho = ctx.material.density;
        let (x, y) = self.get_dimensions(ctx);
        let total_mass = rho * self.thickness * x * y;

        let ndofs = self.ndofs();
        let mut me = DMatrix::zeros(ndofs, ndofs);
        let mass_per_node = total_mass / 4.0;
        for i in 0..4 {
            me[(i * PLATE_BENDING_DOFS, i * PLATE_BENDING_DOFS)] = mass_per_node;
        }
        Some(me)
    }
}

/// Plate bending stress result.
#[derive(Debug, Clone)]
pub struct PlateBendingStress {
    pub moments: [f64; 3],
    pub sigma_x_top: f64,
    pub sigma_x_bottom: f64,
    pub sigma_y_top: f64,
    pub sigma_y_bottom: f64,
}

impl PlateBendingStress {
    pub fn from_moments(moments: &[f64; 3], thickness: f64) -> Self {
        let c = thickness / 2.0;
        let i = thickness.powi(3) / 12.0;
        Self {
            moments: *moments,
            sigma_x_top: -moments[0] * c / i,
            sigma_x_bottom: moments[0] * c / i,
            sigma_y_top: -moments[1] * c / i,
            sigma_y_bottom: moments[1] * c / i,
        }
    }
}

#[cfg(test)]
mod plate_bending_tests {
    use super::*;
    use crate::core::{Material, Node, Section};

    #[test]
    fn test_kirchhoff_plate_creation() {
        let elem = KirchhoffPlate4::new(0, 1, 2, 3, 0.01);
        assert_eq!(elem.ndofs(), 12);
        assert_eq!(elem.node_ids().len(), 4);
        assert!((elem.thickness - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_mindlin_plate_creation() {
        let elem = MindlinPlate4::new(0, 1, 2, 3, 0.01);
        assert_eq!(elem.ndofs(), 12);
        assert!((elem.shear_correction - 5.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_mindlin_custom_shear() {
        let elem = MindlinPlate4::with_shear_correction(0, 1, 2, 3, 0.01, 0.85);
        assert!((elem.shear_correction - 0.85).abs() < 1e-10);
    }

    #[test]
    fn test_plate_bending_stress() {
        let moments = [100.0, 50.0, 25.0];
        let stress = PlateBendingStress::from_moments(&moments, 0.01);
        assert!(stress.sigma_x_top.is_finite());
        assert!(stress.sigma_x_top * stress.sigma_x_bottom < 0.0);
    }
}

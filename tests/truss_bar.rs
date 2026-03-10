use approx::assert_relative_eq;
use fea::prelude::*;
use fea::core::Node;
use fea::modal::{ModalSolver, ModalConfig, MassFormulation};
use fea::beam::{Beam2D, BeamModel, assemble_beam_stiffness};

// =============================================================================
// Unit Tests for Core Module
// =============================================================================

#[test]
fn test_dof_index_in_3d() {
    assert_eq!(Dof::Ux.index_in_3d(), 0);
    assert_eq!(Dof::Uy.index_in_3d(), 1);
    assert_eq!(Dof::Uz.index_in_3d(), 2);
}

#[test]
fn test_node_new_2d() {
    let node = Node::new_2d(3.0, 4.0);
    assert_eq!(node.x, 3.0);
    assert_eq!(node.y, 4.0);
    assert_eq!(node.z, 0.0);
}

#[test]
fn test_node_new_3d() {
    let node = Node::new_3d(1.0, 2.0, 3.0);
    assert_eq!(node.x, 1.0);
    assert_eq!(node.y, 2.0);
    assert_eq!(node.z, 3.0);
}

#[test]
fn test_node_as_array() {
    let node = Node::new_3d(1.0, 2.0, 3.0);
    let arr = node.as_array();
    assert_eq!(arr, [1.0, 2.0, 3.0]);
}

#[test]
fn test_model_default_and_new() {
    let model: Model<Truss2> = Model::default();
    assert!(model.nodes.is_empty());
    assert!(model.elements.is_empty());
    assert!(model.loads.is_empty());
    assert!(model.bcs.is_empty());

    let model: Model<Truss2> = Model::new();
    assert!(model.nodes.is_empty());
}

#[test]
fn test_model_add_node() {
    let mut model: Model<Truss2> = Model::new();
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(1.0, 0.0));
    assert_eq!(n0, 0);
    assert_eq!(n1, 1);
    assert_eq!(model.nodes.len(), 2);
}

#[test]
fn test_model_add_element() {
    let mut model = Model::new();
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(1.0, 0.0));
    model.add_element(Truss2::new(n0, n1, 210e9, 1e-4));
    assert_eq!(model.elements.len(), 1);
}

#[test]
fn test_model_add_load() {
    let mut model: Model<Truss2> = Model::new();
    model.add_load(Load {
        node: 0,
        dof: Dof::Ux,
        value: 100.0,
    });
    assert_eq!(model.loads.len(), 1);
}

#[test]
fn test_model_add_bc() {
    let mut model: Model<Truss2> = Model::new();
    model.add_bc(BoundaryCondition {
        node: 0,
        dof: Dof::Ux,
        value: 0.0,
    });
    assert_eq!(model.bcs.len(), 1);
}

#[test]
fn test_model_build_dofs_3d() {
    let mut model: Model<Truss2> = Model::new();
    model.add_node(Node::new_2d(0.0, 0.0));
    model.add_node(Node::new_2d(1.0, 0.0));
    let ndof = model.build_dofs_3d();
    assert_eq!(ndof, 6); // 2 nodes * 3 DOFs each

    // Verify DOF mapping exists
    assert!(model.dof_index(0, Dof::Ux).is_some());
    assert!(model.dof_index(0, Dof::Uy).is_some());
    assert!(model.dof_index(0, Dof::Uz).is_some());
    assert!(model.dof_index(1, Dof::Ux).is_some());
    assert!(model.dof_index(1, Dof::Uy).is_some());
    assert!(model.dof_index(1, Dof::Uz).is_some());
}

#[test]
fn test_model_dof_index_ordering() {
    let mut model: Model<Truss2> = Model::new();
    model.add_node(Node::new_2d(0.0, 0.0));
    model.add_node(Node::new_2d(1.0, 0.0));
    model.build_dofs_3d();

    // Check consistent ordering
    assert_eq!(model.dof_index(0, Dof::Ux), Some(0));
    assert_eq!(model.dof_index(0, Dof::Uy), Some(1));
    assert_eq!(model.dof_index(0, Dof::Uz), Some(2));
    assert_eq!(model.dof_index(1, Dof::Ux), Some(3));
    assert_eq!(model.dof_index(1, Dof::Uy), Some(4));
    assert_eq!(model.dof_index(1, Dof::Uz), Some(5));
}

// =============================================================================
// Unit Tests for Elements Module
// =============================================================================

#[test]
fn test_truss2_new() {
    let e = 210e9;
    let a = 1e-4;
    let elem = Truss2::new(0, 1, e, a);
    assert_eq!(elem.n1, 0);
    assert_eq!(elem.n2, 1);
    assert_eq!(elem.e, e);
    assert_eq!(elem.a, a);
}

#[test]
fn test_truss2_node_ids() {
    let elem = Truss2::new(5, 10, 210e9, 1e-4);
    assert_eq!(elem.node_ids(), vec![5, 10]);
}

#[test]
fn test_truss2_length_and_dir_1d() {
    let mut model = Model::new();
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(2.0, 0.0));
    let elem = Truss2::new(n0, n1, 210e9, 1e-4);

    let (length, dir) = elem.length_and_dir(&model);
    assert_relative_eq!(length, 2.0, epsilon = 1e-12);
    assert_relative_eq!(dir[0], 1.0, epsilon = 1e-12);
    assert_relative_eq!(dir[1], 0.0, epsilon = 1e-12);
    assert_relative_eq!(dir[2], 0.0, epsilon = 1e-12);
}

#[test]
fn test_truss2_length_and_dir_2d_diagonal() {
    let mut model = Model::new();
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(3.0, 4.0));
    let elem = Truss2::new(n0, n1, 210e9, 1e-4);

    let (length, dir) = elem.length_and_dir(&model);
    assert_relative_eq!(length, 5.0, epsilon = 1e-12);
    assert_relative_eq!(dir[0], 0.6, epsilon = 1e-12);
    assert_relative_eq!(dir[1], 0.8, epsilon = 1e-12);
    assert_relative_eq!(dir[2], 0.0, epsilon = 1e-12);
}

#[test]
fn test_truss2_length_and_dir_3d() {
    let mut model = Model::new();
    let n0 = model.add_node(Node::new_3d(0.0, 0.0, 0.0));
    let n1 = model.add_node(Node::new_3d(1.0, 2.0, 2.0));
    let elem = Truss2::new(n0, n1, 210e9, 1e-4);

    let (length, dir) = elem.length_and_dir(&model);
    let expected_len = 3.0; // sqrt(1 + 4 + 4) = 3
    assert_relative_eq!(length, expected_len, epsilon = 1e-12);
    assert_relative_eq!(dir[0], 1.0 / 3.0, epsilon = 1e-12);
    assert_relative_eq!(dir[1], 2.0 / 3.0, epsilon = 1e-12);
    assert_relative_eq!(dir[2], 2.0 / 3.0, epsilon = 1e-12);
}

#[test]
fn test_truss2_length_zero() {
    let mut model = Model::new();
    let n0 = model.add_node(Node::new_2d(1.0, 1.0));
    let n1 = model.add_node(Node::new_2d(1.0, 1.0));
    let elem = Truss2::new(n0, n1, 210e9, 1e-4);

    let (length, dir) = elem.length_and_dir(&model);
    assert_eq!(length, 0.0);
    assert_eq!(dir, [0.0, 0.0, 0.0]);
}

#[test]
fn test_truss2_stiffness_1d() {
    let mut model = Model::new();
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(1.0, 0.0));
    let e = 210e9;
    let a = 1e-4;
    let elem = Truss2::new(n0, n1, e, a);

    model.build_dofs_3d();
    let k = elem.stiffness(&model);

    // For a 1D truss along X, k_00 = k_22 = EA/L, k_03 = k_30 = -EA/L
    let k_axial = e * a / 1.0;

    assert_relative_eq!(k[(0, 0)], k_axial, epsilon = 1e-6);
    assert_relative_eq!(k[(3, 3)], k_axial, epsilon = 1e-6);
    assert_relative_eq!(k[(0, 3)], -k_axial, epsilon = 1e-6);
    assert_relative_eq!(k[(3, 0)], -k_axial, epsilon = 1e-6);

    // Y and Z DOFs should have zero stiffness for pure X-aligned element
    assert_relative_eq!(k[(1, 1)], 0.0, epsilon = 1e-10);
    assert_relative_eq!(k[(4, 4)], 0.0, epsilon = 1e-10);
}

#[test]
fn test_truss2_stiffness_symmetry() {
    let mut model = Model::new();
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(1.0, 1.0));
    let elem = Truss2::new(n0, n1, 210e9, 1e-4);

    model.build_dofs_3d();
    let k = elem.stiffness(&model);

    // Check symmetry
    for i in 0..6 {
        for j in 0..6 {
            assert_relative_eq!(k[(i, j)], k[(j, i)], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_truss2_axial_stress_tension() {
    let mut model = Model::new();
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(2.0, 0.0));
    let e = 210e9;
    let a = 1e-4;
    let elem = Truss2::new(n0, n1, e, a);

    model.build_dofs_3d();

    // Apply displacement: stretch by 0.001 m
    let mut u = vec![0.0; 6];
    u[model.dof_index(n1, Dof::Ux).unwrap()] = 0.001;

    let stress = elem.axial_stress(&model, &u);
    let expected_stress = e * (0.001 / 2.0); // E * strain = E * delta_L / L

    assert_relative_eq!(stress, expected_stress, max_relative = 1e-10);
}

#[test]
fn test_truss2_axial_stress_compression() {
    let mut model = Model::new();
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(2.0, 0.0));
    let e = 210e9;
    let elem = Truss2::new(n0, n1, e, 1e-4);

    model.build_dofs_3d();

    // Apply displacement: compress by 0.001 m
    let mut u = vec![0.0; 6];
    u[model.dof_index(n1, Dof::Ux).unwrap()] = -0.001;

    let stress = elem.axial_stress(&model, &u);
    let expected_stress = -e * (0.001 / 2.0); // Negative = compression

    assert_relative_eq!(stress, expected_stress, max_relative = 1e-10);
}

#[test]
fn test_truss2_axial_stress_zero_displacement() {
    let mut model = Model::new();
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(2.0, 0.0));
    let elem = Truss2::new(n0, n1, 210e9, 1e-4);

    model.build_dofs_3d();
    let u = vec![0.0; 6];

    let stress = elem.axial_stress(&model, &u);
    assert_eq!(stress, 0.0);
}

#[test]
fn test_truss2_axial_stress_perpendicular_displacement() {
    let mut model = Model::new();
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(2.0, 0.0));
    let elem = Truss2::new(n0, n1, 210e9, 1e-4);

    model.build_dofs_3d();

    // Apply perpendicular (Y) displacement - should produce no axial stress
    // for small displacements (linear theory)
    let mut u = vec![0.0; 6];
    u[model.dof_index(n1, Dof::Uy).unwrap()] = 0.001;

    let stress = elem.axial_stress(&model, &u);
    assert_relative_eq!(stress, 0.0, epsilon = 1e-10);
}

// =============================================================================
// Integration Tests for Solver
// =============================================================================

#[test]
fn truss_bar_matches_closed_form_displacement() -> anyhow::Result<()> {
    let e = 210e9;
    let a = 1.0e-4;
    let l = 2.0;
    let f = 10_000.0;

    let mut model = Model::<Truss2>::new();
    let n0 = model.add_node(fea::core::Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(fea::core::Node::new_2d(l, 0.0));
    model.add_element(Truss2::new(n0, n1, e, a));

    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition {
            node: n0,
            dof,
            value: 0.0,
        });
    }
    for dof in [Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition {
            node: n1,
            dof,
            value: 0.0,
        });
    }
    model.add_load(Load {
        node: n1,
        dof: Dof::Ux,
        value: f,
    });

    let result = LinearStaticSolver::new().solve_truss2(&mut model)?;
    let ux1 = model
        .dof_index(n1, Dof::Ux)
        .and_then(|i| result.u.get(i))
        .copied()
        .unwrap_or(0.0);

    let expected = f * l / (a * e);
    assert_relative_eq!(ux1, expected, max_relative = 1e-9, epsilon = 1e-15);
    Ok(())
}

#[test]
fn test_solver_reaction_forces() -> anyhow::Result<()> {
    let e = 210e9;
    let a = 1.0e-4;
    let l = 2.0;
    let f = 10_000.0;

    let mut model = Model::new();
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(l, 0.0));
    model.add_element(Truss2::new(n0, n1, e, a));

    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition {
            node: n0,
            dof,
            value: 0.0,
        });
    }
    for dof in [Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition {
            node: n1,
            dof,
            value: 0.0,
        });
    }
    model.add_load(Load {
        node: n1,
        dof: Dof::Ux,
        value: f,
    });

    let solver = LinearStaticSolver::new();
    let result = solver.solve_truss2(&mut model)?;

    // Reaction at fixed end should equal applied load (equilibrium)
    let rx0 = result.reactions.get(&0).copied().unwrap_or(0.0);
    assert_relative_eq!(rx0, -f, max_relative = 1e-10);

    Ok(())
}

#[test]
fn test_multi_element_truss_series() -> anyhow::Result<()> {
    // Three elements in series
    let e = 210e9;
    let a = 1.0e-4;
    let l = 1.0; // Each element length
    let f = 10_000.0;

    let mut model = Model::new();

    // Create 4 nodes
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(l, 0.0));
    let n2 = model.add_node(Node::new_2d(2.0 * l, 0.0));
    let n3 = model.add_node(Node::new_2d(3.0 * l, 0.0));

    // Create 3 elements
    model.add_element(Truss2::new(n0, n1, e, a));
    model.add_element(Truss2::new(n1, n2, e, a));
    model.add_element(Truss2::new(n2, n3, e, a));

    // Fix n0
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition {
            node: n0,
            dof,
            value: 0.0,
        });
    }

    // Constrain Y/Z for all other nodes
    for &n in &[n1, n2, n3] {
        for dof in [Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition {
                node: n,
                dof,
                value: 0.0,
            });
        }
    }

    // Apply load at n3
    model.add_load(Load {
        node: n3,
        dof: Dof::Ux,
        value: f,
    });

    let solver = LinearStaticSolver::new();
    let result = solver.solve_truss2(&mut model)?;

    // Total displacement should be f * (3L) / (A * E)
    let total_length = 3.0 * l;
    let expected_u3 = f * total_length / (a * e);

    let u3 = model
        .dof_index(n3, Dof::Ux)
        .and_then(|i| result.u.get(i))
        .copied()
        .unwrap_or(0.0);

    assert_relative_eq!(u3, expected_u3, max_relative = 1e-9);

    Ok(())
}

#[test]
fn test_truss_2d_triangle() -> anyhow::Result<()> {
    // Simple triangular truss - both bottom nodes fully fixed (cantilevered triangle)
    let e = 210e9;
    let a = 1.0e-4;

    let mut model = Model::new();

    // Right triangle: (0,0), (2,0), (0,1) - non-collinear points
    // Using explicit 2D constructor with z=0
    let n0 = model.add_node(Node { x: 0.0, y: 0.0, z: 0.0 });
    let n1 = model.add_node(Node { x: 2.0, y: 0.0, z: 0.0 });
    let n2 = model.add_node(Node { x: 0.0, y: 1.0, z: 0.0 });

    // Three elements forming triangle
    model.add_element(Truss2::new(n0, n1, e, a));
    model.add_element(Truss2::new(n1, n2, e, a));
    model.add_element(Truss2::new(n0, n2, e, a)); // Changed order for consistent connectivity

    // Fix n0 completely (pin support)
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition {
            node: n0,
            dof,
            value: 0.0,
        });
    }

    // Fix n1 completely
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition {
            node: n1,
            dof,
            value: 0.0,
        });
    }

    // Fix n2 in Z to prevent out-of-plane motion
    model.add_bc(BoundaryCondition {
        node: n2,
        dof: Dof::Uz,
        value: 0.0,
    });

    // Apply horizontal load at n2 (free node in X)
    model.add_load(Load {
        node: n2,
        dof: Dof::Ux,
        value: 1000.0,
    });

    let solver = LinearStaticSolver::new();
    let result = solver.solve_truss2(&mut model)?;

    // Verify solution exists and is finite
    for &disp in &result.u {
        assert!(disp.is_finite());
    }

    // Verify n2 moves in direction of load (positive X displacement)
    let n2_ux = model
        .dof_index(n2, Dof::Ux)
        .and_then(|i| result.u.get(i))
        .copied()
        .unwrap_or(0.0);
    assert!(n2_ux > 0.0, "Node n2 should move in direction of load");

    Ok(())
}

#[test]
fn test_solver_all_dofs_constrained() -> anyhow::Result<()> {
    // When all DOFs are constrained, solver should return without error
    let mut model = Model::new();
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(1.0, 0.0));
    model.add_element(Truss2::new(n0, n1, 210e9, 1e-4));

    // Constrain all DOFs at both nodes
    for &n in &[n0, n1] {
        for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition {
                node: n,
                dof,
                value: 0.0,
            });
        }
    }

    let solver = LinearStaticSolver::new();
    let result = solver.solve_truss2(&mut model)?;

    // All displacements should be zero
    for &disp in &result.u {
        assert_relative_eq!(disp, 0.0, epsilon = 1e-15);
    }

    Ok(())
}

// =============================================================================
// Visualization Tests
// =============================================================================

#[test]
fn test_vtk_mesh_from_truss2() {
    let mut model = Model::new();
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(1.0, 0.0));
    model.add_element(Truss2::new(n0, n1, 210e9, 1e-4));

    let mesh = VtkMesh::from_truss2(&model);

    assert_eq!(mesh.points.len(), 2);
    assert_eq!(mesh.cells.len(), 1);
    assert_eq!(mesh.cell_types.len(), 1);
    assert_eq!(mesh.cell_types[0], 3); // VTK_LINE

    assert_relative_eq!(mesh.points[0][0], 0.0, epsilon = 1e-12);
    assert_relative_eq!(mesh.points[1][0], 1.0, epsilon = 1e-12);
}

#[test]
fn test_vtk_writer_file_creation() -> anyhow::Result<()> {
    let mut model = Model::new();
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(1.0, 0.0));
    model.add_element(Truss2::new(n0, n1, 210e9, 1e-4));
    model.build_dofs_3d();

    let u = vec![0.0; 6];

    let temp_path = "/tmp/test_vtk_output.vtk";
    let writer = VtkLegacyWriter::new();
    writer.write_truss2(temp_path, &model, &u)?;

    // Verify file was created
    assert!(std::path::Path::new(temp_path).exists());

    // Cleanup
    std::fs::remove_file(temp_path)?;

    Ok(())
}

#[test]
fn test_vtk_writer_contains_required_sections() -> anyhow::Result<()> {
    let mut model = Model::new();
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(1.0, 0.0));
    model.add_element(Truss2::new(n0, n1, 210e9, 1e-4));
    model.build_dofs_3d();

    let u = vec![0.001; 6];

    let temp_path = "/tmp/test_vtk_sections.vtk";
    let writer = VtkLegacyWriter::new();
    writer.write_truss2(temp_path, &model, &u)?;

    let content = std::fs::read_to_string(temp_path)?;

    // Verify required VTK sections
    assert!(content.contains("# vtk DataFile"));
    assert!(content.contains("DATASET UNSTRUCTURED_GRID"));
    assert!(content.contains("POINTS"));
    assert!(content.contains("CELLS"));
    assert!(content.contains("CELL_TYPES"));
    assert!(content.contains("POINT_DATA"));
    assert!(content.contains("VECTORS displacement"));
    assert!(content.contains("CELL_DATA"));
    assert!(content.contains("SCALARS axial_stress"));

    std::fs::remove_file(temp_path)?;

    Ok(())
}

#[test]
fn test_json_writer_file_creation() -> anyhow::Result<()> {
    let mut model = Model::new();
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(1.0, 0.0));
    model.add_element(Truss2::new(n0, n1, 210e9, 1e-4));
    model.build_dofs_3d();

    let u = vec![0.0; 6];

    let temp_path = "/tmp/test_json_output.json";
    let writer = JsonWriter::new();
    writer.write_truss2(temp_path, &model, &u)?;

    // Verify file was created
    assert!(std::path::Path::new(temp_path).exists());

    // Verify valid JSON
    let content = std::fs::read_to_string(temp_path)?;
    let _: serde_json::Value = serde_json::from_str(&content)?;

    // Cleanup
    std::fs::remove_file(temp_path)?;

    Ok(())
}

#[test]
fn test_json_writer_contains_required_fields() -> anyhow::Result<()> {
    let mut model = Model::new();
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(1.0, 0.0));
    model.add_element(Truss2::new(n0, n1, 210e9, 1e-4));
    model.build_dofs_3d();

    let u = vec![0.001; 6];

    let temp_path = "/tmp/test_json_fields.json";
    let writer = JsonWriter::new();
    writer.write_truss2(temp_path, &model, &u)?;

    let content = std::fs::read_to_string(temp_path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;

    // Verify required fields
    assert!(json.get("points").is_some());
    assert!(json.get("cells").is_some());
    assert!(json.get("point_data").is_some());
    assert!(json.get("cell_data").is_some());
    assert!(json["point_data"].get("displacement").is_some());
    assert!(json["cell_data"].get("axial_stress").is_some());

    std::fs::remove_file(temp_path)?;

    Ok(())
}

#[test]
fn test_json_writer_displacement_values() -> anyhow::Result<()> {
    let mut model = Model::new();
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(1.0, 0.0));
    model.add_element(Truss2::new(n0, n1, 210e9, 1e-4));
    model.build_dofs_3d();

    // Set known displacements
    let mut u = vec![0.0; 6];
    u[3] = 0.001; // n1 Ux
    u[4] = 0.002; // n1 Uy

    let temp_path = "/tmp/test_json_disp.json";
    let writer = JsonWriter::new();
    writer.write_truss2(temp_path, &model, &u)?;

    let content = std::fs::read_to_string(temp_path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;

    let disp = &json["point_data"]["displacement"];
    // Node 0: [0, 0, 0]
    assert_relative_eq!(disp[0][0].as_f64().unwrap(), 0.0, epsilon = 1e-15);
    assert_relative_eq!(disp[0][1].as_f64().unwrap(), 0.0, epsilon = 1e-15);
    // Node 1: [0.001, 0.002, 0]
    assert_relative_eq!(disp[1][0].as_f64().unwrap(), 0.001, epsilon = 1e-15);
    assert_relative_eq!(disp[1][1].as_f64().unwrap(), 0.002, epsilon = 1e-15);

    std::fs::remove_file(temp_path)?;

    Ok(())
}

// =============================================================================
// Modal Analysis Tests
// =============================================================================

#[test]
fn test_modal_analysis_simple_truss() -> anyhow::Result<()> {
    let e = 210e9;
    let a = 1.0e-4;
    let l = 1.0;

    let mut model = Model::<Truss2>::new();
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(l, 0.0));
    model.add_element(Truss2::new(n0, n1, e, a));

    // Fix n0 completely
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition {
            node: n0,
            dof,
            value: 0.0,
        });
    }
    // Constrain n1 in Y and Z
    for dof in [Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition {
            node: n1,
            dof,
            value: 0.0,
        });
    }

    let config = ModalConfig {
        num_modes: 1,
        mass_formulation: MassFormulation::Lumped,
        max_iterations: 100,
        tolerance: 1e-8,
    };
    let solver = ModalSolver::with_config(config);
    let result = solver.analyze_truss2(&mut model)?;

    // Should find at least one mode
    assert!(!result.frequencies.is_empty());
    assert!(result.frequencies[0] > 0.0, "Natural frequency should be positive");

    Ok(())
}

#[test]
fn test_modal_analysis_multi_element() -> anyhow::Result<()> {
    let e = 210e9;
    let a = 1.0e-4;
    let l = 0.5;

    let mut model = Model::<Truss2>::new();

    // Create 3-element bar
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(l, 0.0));
    let n2 = model.add_node(Node::new_2d(2.0 * l, 0.0));
    let n3 = model.add_node(Node::new_2d(3.0 * l, 0.0));

    model.add_element(Truss2::new(n0, n1, e, a));
    model.add_element(Truss2::new(n1, n2, e, a));
    model.add_element(Truss2::new(n2, n3, e, a));

    // Fix n0
    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition {
            node: n0,
            dof,
            value: 0.0,
        });
    }
    // Constrain Y/Z for other nodes
    for &n in &[n1, n2, n3] {
        for dof in [Dof::Uy, Dof::Uz] {
            model.add_bc(BoundaryCondition {
                node: n,
                dof,
                value: 0.0,
            });
        }
    }

    let solver = ModalSolver::new();
    let result = solver.analyze_truss2(&mut model)?;

    // Should find modes
    assert!(!result.frequencies.is_empty());

    // Frequencies should increase for higher modes
    for i in 1..result.frequencies.len() {
        assert!(result.frequencies[i] >= result.frequencies[i - 1]);
    }

    Ok(())
}

#[test]
fn test_modal_analysis_consistent_mass() -> anyhow::Result<()> {
    let e = 210e9;
    let a = 1.0e-4;
    let l = 1.0;

    let mut model = Model::<Truss2>::new();
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(l, 0.0));
    model.add_element(Truss2::new(n0, n1, e, a));

    for dof in [Dof::Ux, Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition {
            node: n0,
            dof,
            value: 0.0,
        });
    }
    for dof in [Dof::Uy, Dof::Uz] {
        model.add_bc(BoundaryCondition {
            node: n1,
            dof,
            value: 0.0,
        });
    }

    let config = ModalConfig {
        num_modes: 1,
        mass_formulation: MassFormulation::Consistent,
        ..Default::default()
    };
    let solver = ModalSolver::with_config(config);
    let result = solver.analyze_truss2(&mut model)?;

    assert!(!result.frequencies.is_empty());
    assert!(result.frequencies[0] > 0.0);

    Ok(())
}

// =============================================================================
// Beam Element Tests
// =============================================================================

#[test]
fn test_beam_cantilever_static() {
    // Cantilever beam with point load at tip
    let mut model = BeamModel::new();
    let n0 = model.add_node([0.0, 0.0]);
    let n1 = model.add_node([1.0, 0.0]);

    let e = 210e9;
    let a = 1e-4;
    let i = 1e-8;
    let l = 1.0;

    model.add_element(Beam2D::new(n0, n1, e, a, i));

    // Assemble stiffness (6 DOFs total)
    let k = assemble_beam_stiffness(&model, 6);

    // Check that axial stiffness is correct: EA/L
    let expected_axial = e * a / l;
    assert_relative_eq!(k[(0, 0)], expected_axial, max_relative = 1e-10);

    // Check bending stiffness: 12EI/L^3
    let expected_bending = 12.0 * e * i / (l * l * l);
    assert_relative_eq!(k[(1, 1)], expected_bending, max_relative = 1e-10);

    // Check rotational stiffness: 4EI/L
    let expected_rotation = 4.0 * e * i / l;
    assert_relative_eq!(k[(2, 2)], expected_rotation, max_relative = 1e-10);
}

#[test]
fn test_beam_simply_supported() {
    // Simply supported beam
    let mut model = BeamModel::new();
    let n0 = model.add_node([0.0, 0.0]);
    let n1 = model.add_node([2.0, 0.0]);
    let n2 = model.add_node([4.0, 0.0]);

    let e = 210e9;
    let a = 1e-4;
    let i = 1e-8;

    model.add_element(Beam2D::new(n0, n1, e, a, i));
    model.add_element(Beam2D::new(n1, n2, e, a, i));

    let k = assemble_beam_stiffness(&model, 9); // 3 nodes * 3 DOFs

    assert_eq!(k.nrows(), 9);
    assert_eq!(k.ncols(), 9);

    // Stiffness matrix should be symmetric
    for i in 0..9 {
        for j in 0..9 {
            assert_relative_eq!(k[(i, j)], k[(j, i)], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_beam_element_length_various_orientations() {
    // Test horizontal beam
    let mut model = BeamModel::new();
    let n0 = model.add_node([0.0, 0.0]);
    let n1 = model.add_node([3.0, 0.0]);
    let beam = Beam2D::new(n0, n1, 210e9, 1e-4, 1e-8);
    let (len, dir) = beam.length_and_dir(&model);
    assert_relative_eq!(len, 3.0, epsilon = 1e-12);
    assert_relative_eq!(dir[0], 1.0, epsilon = 1e-12);

    // Test vertical beam
    let mut model = BeamModel::new();
    let n0 = model.add_node([0.0, 0.0]);
    let n1 = model.add_node([0.0, 4.0]);
    let beam = Beam2D::new(n0, n1, 210e9, 1e-4, 1e-8);
    let (len, dir) = beam.length_and_dir(&model);
    assert_relative_eq!(len, 4.0, epsilon = 1e-12);
    assert_relative_eq!(dir[1], 1.0, epsilon = 1e-12);

    // Test diagonal beam (3-4-5 triangle)
    let mut model = BeamModel::new();
    let n0 = model.add_node([0.0, 0.0]);
    let n1 = model.add_node([3.0, 4.0]);
    let beam = Beam2D::new(n0, n1, 210e9, 1e-4, 1e-8);
    let (len, dir) = beam.length_and_dir(&model);
    assert_relative_eq!(len, 5.0, epsilon = 1e-12);
    assert_relative_eq!(dir[0], 0.6, epsilon = 1e-12);
    assert_relative_eq!(dir[1], 0.8, epsilon = 1e-12);
}

// =============================================================================
// Visualization Enhancement Tests
// =============================================================================

#[test]
fn test_vtk_writer_includes_displacement_magnitude() -> anyhow::Result<()> {
    let mut model = Model::new();
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(1.0, 0.0));
    model.add_element(Truss2::new(n0, n1, 210e9, 1e-4));
    model.build_dofs_3d();

    // Set known displacements
    let mut u = vec![0.0; 6];
    u[3] = 0.003; // n1 Ux
    u[4] = 0.004; // n1 Uy

    let temp_path = "/tmp/test_vtk_disp_mag.vtk";
    let writer = VtkLegacyWriter::new();
    writer.write_truss2(temp_path, &model, &u)?;

    let content = std::fs::read_to_string(temp_path)?;

    // Verify displacement_magnitude section exists
    assert!(content.contains("displacement_magnitude"), "VTK should include displacement_magnitude");

    // Verify magnitude values are present
    assert!(content.contains("0.005"), "Should contain magnitude 0.005 (sqrt(0.003^2 + 0.004^2))");

    std::fs::remove_file(temp_path)?;
    Ok(())
}

#[test]
fn test_json_enhanced_includes_displacement_magnitude() -> anyhow::Result<()> {
    let mut model = Model::new();
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(1.0, 0.0));
    model.add_element(Truss2::new(n0, n1, 210e9, 1e-4));
    model.build_dofs_3d();

    let mut u = vec![0.0; 6];
    u[3] = 0.003; // n1 Ux
    u[4] = 0.004; // n1 Uy

    let temp_path = "/tmp/test_json_enhanced_mag.json";
    let writer = JsonWriter::new();
    let config = VizConfig::default();
    writer.write_truss2_enhanced(temp_path, &model, &u, &config)?;

    let content = std::fs::read_to_string(temp_path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;

    // Verify displacement_magnitude field exists
    assert!(json["undeformed"]["point_data"].get("displacement_magnitude").is_some());
    assert!(json["deformed"]["point_data"].get("displacement_magnitude").is_some());

    // Verify magnitude values
    let mag = &json["undeformed"]["point_data"]["displacement_magnitude"];
    assert_relative_eq!(mag[0].as_f64().unwrap(), 0.0, epsilon = 1e-15);
    assert_relative_eq!(mag[1].as_f64().unwrap(), 0.005, epsilon = 1e-15); // sqrt(0.003^2 + 0.004^2)

    std::fs::remove_file(temp_path)?;
    Ok(())
}

#[test]
fn test_viz_config_auto_scale() {
    let mut model = Model::new();
    let n0 = model.add_node(Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(Node::new_2d(2.0, 0.0));
    model.add_element(Truss2::new(n0, n1, 210e9, 1e-4));
    model.build_dofs_3d();

    let u = vec![0.001; 6]; // All DOFs have 0.001 displacement

    let config = VizConfig::with_auto_scale(&model, &u, 0.1);

    // Scale should be computed to make max displacement 10% of model size
    assert!(config.deformation_scale > 0.0);
    assert!(config.deformation_scale < 1000.0); // Reasonable bounds
}

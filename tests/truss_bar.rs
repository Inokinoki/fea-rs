use approx::assert_relative_eq;
use fea::prelude::*;

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

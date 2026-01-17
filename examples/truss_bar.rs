use fea::prelude::*;

fn main() -> anyhow::Result<()> {
    // Simple 2-node bar along X.
    //
    // Expected displacement (node 1, Ux): u = F*L/(A*E)
    let e = 210e9;
    let a = 1.0e-4;
    let l = 2.0;
    let f = 10_000.0;

    let mut model = Model::<Truss2>::new();
    let n0 = model.add_node(fea::core::Node::new_2d(0.0, 0.0));
    let n1 = model.add_node(fea::core::Node::new_2d(l, 0.0));
    model.add_element(Truss2::new(n0, n1, e, a));

    // Fix node 0, and constrain Y/Z on node 1 for a stable truss system.
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

    let ux1 = model
        .dof_index(n1, Dof::Ux)
        .and_then(|i| result.u.get(i))
        .copied()
        .unwrap_or(0.0);
    let expected = f * l / (a * e);
    println!("u(node1, Ux) = {ux1:e} m, expected {expected:e} m");

    std::fs::create_dir_all("out")?;
    std::fs::create_dir_all("web")?;

    VtkLegacyWriter::new().write_truss2("out/truss_bar.vtk", &model, &result.u)?;
    JsonWriter::new().write_truss2("web/model.json", &model, &result.u)?;

    println!("Wrote `out/truss_bar.vtk` (ParaView) and `web/model.json` (browser viewer).");
    Ok(())
}

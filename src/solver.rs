use crate::core::{BoundaryCondition, Dof, Load, Model, NodeId};
use crate::elements_legacy::{ElementOld as Element, Truss2};
use nalgebra::{DMatrix, DVector};
use std::collections::{BTreeMap, BTreeSet};

/// Result of a linear static solve.
#[derive(Debug, Clone)]
pub struct LinearStaticResult {
    /// Global displacement vector (size = n_dofs).
    pub u: Vec<f64>,
    /// Global nodal force vector (size = n_dofs).
    pub f: Vec<f64>,
    /// Reaction forces computed at constrained DOFs (map of global dof index -> value).
    pub reactions: BTreeMap<usize, f64>,
}

/// Linear static solver for small/medium models (dense global stiffness).
#[derive(Debug, Default)]
pub struct LinearStaticSolver;

impl LinearStaticSolver {
    pub fn new() -> Self {
        Self
    }

    /// Solves a truss/bar model using a dense stiffness matrix and Dirichlet elimination.
    ///
    /// Notes:
    /// - This uses a 3D DOF layout per node (Ux, Uy, Uz). For 2D models, keep `z=0`
    ///   and constrain `Uz` as needed.
    pub fn solve_truss2(&self, model: &mut Model<Truss2>) -> anyhow::Result<LinearStaticResult> {
        let ndof = model.build_dofs_3d();
        let k = assemble_global_stiffness_truss2(model, ndof);
        let f = assemble_global_load_vector(model, ndof);

        let (u, reactions) = solve_with_dirichlet(&k, &f, &model.bcs, model)?;
        Ok(LinearStaticResult {
            u,
            f: f.data.as_vec().clone(),
            reactions,
        })
    }
}

fn assemble_global_stiffness_truss2(model: &Model<Truss2>, ndof: usize) -> DMatrix<f64> {
    let mut k = DMatrix::<f64>::zeros(ndof, ndof);

    for e in &model.elements {
        let ke = e.stiffness(model);

        let idx = element_dof_indices_truss2(model, e.n1, e.n2);
        for (a, &ia) in idx.iter().enumerate() {
            for (b, &ib) in idx.iter().enumerate() {
                k[(ia, ib)] += ke[(a, b)];
            }
        }
    }

    k
}

fn assemble_global_load_vector(model: &Model<Truss2>, ndof: usize) -> DVector<f64> {
    let mut f = DVector::<f64>::zeros(ndof);
    for Load { node, dof, value } in &model.loads {
        if let Some(i) = model.dof_index(*node, *dof) {
            f[i] += *value;
        }
    }
    f
}

fn element_dof_indices_truss2(model: &Model<Truss2>, n1: NodeId, n2: NodeId) -> [usize; 6] {
    let dof = |nid: NodeId, d: Dof| model.dof_index(nid, d).unwrap();
    [
        dof(n1, Dof::Ux),
        dof(n1, Dof::Uy),
        dof(n1, Dof::Uz),
        dof(n2, Dof::Ux),
        dof(n2, Dof::Uy),
        dof(n2, Dof::Uz),
    ]
}

fn solve_with_dirichlet(
    k: &DMatrix<f64>,
    f: &DVector<f64>,
    bcs: &[BoundaryCondition],
    model: &Model<Truss2>,
) -> anyhow::Result<(Vec<f64>, BTreeMap<usize, f64>)> {
    let ndof = k.nrows();
    if ndof != k.ncols() || ndof != f.len() {
        anyhow::bail!(
            "dimension mismatch: K is {}x{}, f is {}",
            k.nrows(),
            k.ncols(),
            f.len()
        );
    }

    // Constrained dofs and their prescribed values.
    let mut prescribed: BTreeMap<usize, f64> = BTreeMap::new();
    for bc in bcs {
        if let Some(i) = model.dof_index(bc.node, bc.dof) {
            prescribed.insert(i, bc.value);
        }
    }

    let constrained: BTreeSet<usize> = prescribed.keys().copied().collect();
    let free: Vec<usize> = (0..ndof).filter(|i| !constrained.contains(i)).collect();

    // If everything is constrained, just return the prescribed vector.
    let mut u_full = vec![0.0f64; ndof];
    for (i, v) in &prescribed {
        u_full[*i] = *v;
    }

    if free.is_empty() {
        let reactions = compute_reactions(k, f, &u_full, &constrained);
        return Ok((u_full, reactions));
    }

    // Build reduced system: K_ff * u_f = f_f - K_fc * u_c
    let nfree = free.len();
    let mut k_ff = DMatrix::<f64>::zeros(nfree, nfree);
    let mut rhs = DVector::<f64>::zeros(nfree);

    for (row_pos, &i) in free.iter().enumerate() {
        rhs[row_pos] = f[i];
        for (col_pos, &j) in free.iter().enumerate() {
            k_ff[(row_pos, col_pos)] = k[(i, j)];
        }
        for (&c, &uc) in &prescribed {
            rhs[row_pos] -= k[(i, c)] * uc;
        }
    }

    // Solve using LU (works for general dense matrices; SPD models will also succeed).
    let u_free = k_ff
        .lu()
        .solve(&rhs)
        .ok_or_else(|| anyhow::anyhow!("failed to solve: matrix is singular"))?;

    for (pos, &i) in free.iter().enumerate() {
        u_full[i] = u_free[pos];
    }

    let reactions = compute_reactions(k, f, &u_full, &constrained);
    Ok((u_full, reactions))
}

fn compute_reactions(
    k: &DMatrix<f64>,
    f: &DVector<f64>,
    u: &[f64],
    constrained: &BTreeSet<usize>,
) -> BTreeMap<usize, f64> {
    let ndof = u.len();
    let uvec = DVector::<f64>::from_column_slice(u);
    let r = k * uvec - f;
    let mut out = BTreeMap::new();
    for &i in constrained {
        if i < ndof {
            out.insert(i, r[i]);
        }
    }
    out
}

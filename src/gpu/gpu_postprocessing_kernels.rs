//! GPU Kernels for Pre/Post-Processing Operations.
//!
//! This module provides CUDA/OpenCL kernels for:
//! - Nodal coordinate transformations
//! - Stress/strain visualization
//! - Contour plot generation
//! - Animation frame generation
//! - Result export operations

/// CUDA kernel for nodal coordinate transformation (rotation)
pub const CUDA_NODE_ROTATE_KERNEL: &str = r#"
extern "C" __global__
void node_rotate(const int num_nodes,
                 const double* input_coords,
                 double* output_coords,
                 const double cos_angle,
                 const double sin_angle,
                 const int axis) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    double x = input_coords[node * 3];
    double y = input_coords[node * 3 + 1];
    double z = input_coords[node * 3 + 2];

    if (axis == 0) { // Rotate about X
        output_coords[node * 3] = x;
        output_coords[node * 3 + 1] = y * cos_angle - z * sin_angle;
        output_coords[node * 3 + 2] = y * sin_angle + z * cos_angle;
    } else if (axis == 1) { // Rotate about Y
        output_coords[node * 3] = x * cos_angle + z * sin_angle;
        output_coords[node * 3 + 1] = y;
        output_coords[node * 3 + 2] = -x * sin_angle + z * cos_angle;
    } else if (axis == 2) { // Rotate about Z
        output_coords[node * 3] = x * cos_angle - y * sin_angle;
        output_coords[node * 3 + 1] = x * sin_angle + y * cos_angle;
        output_coords[node * 3 + 2] = z;
    }
}
"#;

/// CUDA kernel for nodal scaling transformation
pub const CUDA_NODE_SCALE_KERNEL: &str = r#"
extern "C" __global__
void node_scale(const int num_nodes,
                const double* input_coords,
                double* output_coords,
                const double scale_x,
                const double scale_y,
                const double scale_z) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    output_coords[node * 3] = input_coords[node * 3] * scale_x;
    output_coords[node * 3 + 1] = input_coords[node * 3 + 1] * scale_y;
    output_coords[node * 3 + 2] = input_coords[node * 3 + 2] * scale_z;
}
"#;

/// CUDA kernel for stress tensor to principal stress conversion
pub const CUDA_STRESS_TO_PRINCIPAL_KERNEL: &str = r#"
extern "C" __global__
void stress_to_principal(const int num_points,
                         const double* stress_xx,
                         const double* stress_yy,
                         const double* stress_zz,
                         const double* stress_xy,
                         const double* stress_yz,
                         const double* stress_xz,
                         double* sigma_1,
                         double* sigma_2,
                         double* sigma_3) {
    int pt = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt >= num_points) return;

    // Compute invariants
    double sxx = stress_xx[pt];
    double syy = stress_yy[pt];
    double szz = stress_zz[pt];
    double sxy = stress_xy[pt];
    double syz = stress_yz[pt];
    double sxz = stress_xz[pt];

    double I1 = sxx + syy + szz;
    double I2 = sxx*syy + syy*szz + szz*sxx - sxy*sxy - syz*syz - sxz*sxz;
    double I3 = sxx*syy*szz + 2*sxy*syz*sxz - sxx*syz*syz - syy*sxz*sxz - szz*sxy*sxy;

    // Cardano's formula for cubic roots (simplified)
    double p = I2 - I1*I1/3.0;
    double q = 2*I1*I1*I1/27.0 - I1*I2/3.0 + I3;

    double discriminant = q*q/4.0 + p*p*p/27.0;

    if (discriminant >= 0) {
        double u = cbrt(-q/2.0 + sqrt(discriminant));
        double v = cbrt(-q/2.0 - sqrt(discriminant));
        sigma_1[pt] = u + v - I1/3.0;
        sigma_2[pt] = -(u + v)/2.0 - I1/3.0;
        sigma_3[pt] = -(u + v)/2.0 - I1/3.0;
    } else {
        double r = sqrt(-p*p*p/27.0);
        double theta = acos(-q/(2.0*r)) / 3.0;
        sigma_1[pt] = 2*r*cos(theta) - I1/3.0;
        sigma_2[pt] = 2*r*cos(theta + 2*3.14159265358979/3.0) - I1/3.0;
        sigma_3[pt] = 2*r*cos(theta + 4*3.14159265358979/3.0) - I1/3.0;
    }
}
"#;

/// CUDA kernel for strain tensor to von Mises strain
pub const CUDA_STRAIN_TO_VON_MISES_KERNEL: &str = r#"
extern "C" __global__
void strain_to_von_mises(const int num_points,
                         const double* exx,
                         const double* eyy,
                         const double* ezz,
                         const double* exy,
                         const double* eyz,
                         const double* exz,
                         double* von_mises_strain) {
    int pt = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt >= num_points) return;

    double ex = exx[pt];
    double ey = eyy[pt];
    double ez = ezz[pt];
    double exy_val = exy[pt];
    double eyz_val = eyz[pt];
    double exz_val = exz[pt];

    // Von Mises strain
    von_mises_strain[pt] = sqrt(
        2.0/9.0 * ((ex-ey)*(ex-ey) + (ey-ez)*(ey-ez) + (ez-ex)*(ez-ex)) +
        4.0/3.0 * (exy_val*exy_val + eyz_val*eyz_val + exz_val*exz_val)
    );
}
"#;

/// CUDA kernel for displacement magnitude
pub const CUDA_DISPLACEMENT_MAGNITUDE_KERNEL: &str = r#"
extern "C" __global__
void displacement_magnitude(const int num_nodes,
                            const double* ux,
                            const double* uy,
                            const double* uz,
                            double* magnitude) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    magnitude[node] = sqrt(ux[node]*ux[node] + uy[node]*uy[node] + uz[node]*uz[node]);
}
"#;

/// CUDA kernel for reaction force computation
pub const CUDA_REACTION_FORCE_KERNEL: &str = r#"
extern "C" __global__
void reaction_force(const int num_constrained_dofs,
                    const double* stiffness,
                    const double* displacements,
                    const int* constraint_indices,
                    double* reactions) {
    int dof = blockIdx.x * blockDim.x + threadIdx.x;
    if (dof >= num_constrained_dofs) return;

    int global_dof = constraint_indices[dof];
    double reaction = 0.0;

    // Sum K_ij * u_j for all j
    for (int j = 0; j < n; j++) {
        reaction += stiffness[global_dof * n + j] * displacements[j];
    }

    reactions[dof] = reaction;
}
"#;

/// CUDA kernel for element force recovery
pub const CUDA_ELEMENT_FORCE_KERNEL: &str = r#"
extern "C" __global__
void element_force(const int num_elements,
                   const int dofs_per_elem,
                   const double* element_matrices,
                   const double* element_displacements,
                   double* element_forces) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem >= num_elements) return;

    int offset = elem * dofs_per_elem * dofs_per_elem;
    int disp_offset = elem * dofs_per_elem;
    int force_offset = elem * dofs_per_elem;

    for (int i = 0; i < dofs_per_elem; i++) {
        double force = 0.0;
        for (int j = 0; j < dofs_per_elem; j++) {
            force += element_matrices[offset + i * dofs_per_elem + j] *
                     element_displacements[disp_offset + j];
        }
        element_forces[force_offset + i] = force;
    }
}
"#;

/// CUDA kernel for contour level assignment
pub const CUDA_CONTOUR_LEVEL_KERNEL: &str = r#"
extern "C" __global__
void contour_level(const int num_values,
                   const double* values,
                   const double min_val,
                   const double max_val,
                   const int num_levels,
                   int* level_indices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_values) return;

    double range = max_val - min_val;
    double normalized = (values[idx] - min_val) / range;
    level_indices[idx] = (int)(normalized * (num_levels - 1));
}
"#;

/// CUDA kernel for RGB color mapping (jet colormap)
pub const CUDA_COLOR_MAP_KERNEL: &str = r#"
extern "C" __global__
void color_map_jet(const int num_values,
                    const double* values,
                    const double min_val,
                    const double max_val,
                    unsigned char* r,
                    unsigned char* g,
                    unsigned char* b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_values) return;

    double t = (values[idx] - min_val) / (max_val - min_val);
    t = fmax(0.0, fmin(1.0, t));

    // Jet colormap
    r[idx] = (unsigned char)(255.0 * fmax(0.0, fmin(1.0, 4.0*t - 1.5)));
    g[idx] = (unsigned char)(255.0 * fmax(0.0, fmin(1.0, 2.0 - fabs(4.0*(t - 0.5)))));
    b[idx] = (unsigned char)(255.0 * fmax(0.0, fmin(1.0, 1.5 - 4.0*(1.0 - t))));
}
"#;

/// CUDA kernel for mesh quality computation (aspect ratio)
pub const CUDA_MESH_QUALITY_KERNEL: &str = r#"
extern "C" __global__
void mesh_aspect_ratio(const int num_elements,
                       const int nodes_per_elem,
                       const double* node_coords,
                       const int* connectivity,
                       double* aspect_ratios) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem >= num_elements) return;

    double max_edge = 0.0;
    double min_edge = 1e30;

    for (int i = 0; i < nodes_per_elem; i++) {
        int node_i = connectivity[elem * nodes_per_elem + i];
        int node_j = connectivity[elem * nodes_per_elem + ((i + 1) % nodes_per_elem)];

        double dx = node_coords[node_i * 3] - node_coords[node_j * 3];
        double dy = node_coords[node_i * 3 + 1] - node_coords[node_j * 3 + 1];
        double dz = node_coords[node_i * 3 + 2] - node_coords[node_j * 3 + 2];

        double edge_len = sqrt(dx*dx + dy*dy + dz*dz);
        max_edge = fmax(max_edge, edge_len);
        min_edge = fmin(min_edge, edge_len);
    }

    aspect_ratios[elem] = (min_edge > 1e-15) ? max_edge / min_edge : 1e30;
}
"#;

/// OpenCL kernel for nodal transformations
pub const OPENCL_NODE_TRANSFORM_KERNEL: &str = r#"
__kernel void node_transform(const int num_nodes,
                             __global const double* input_coords,
                             __global double* output_coords,
                             const double scale,
                             const double tx,
                             const double ty,
                             const double tz) {
    int node = get_global_id(0);
    if (node >= num_nodes) return;

    output_coords[node * 3] = input_coords[node * 3] * scale + tx;
    output_coords[node * 3 + 1] = input_coords[node * 3 + 1] * scale + ty;
    output_coords[node * 3 + 2] = input_coords[node * 3 + 2] * scale + tz;
}
"#;

/// OpenCL kernel for stress visualization
pub const OPENCL_STRESS_VISUALIZATION_KERNEL: &str = r#"
__kernel void stress_visualization(const int num_points,
                                   __global const double* stress_values,
                                   const double min_stress,
                                   const double max_stress,
                                   __global uchar* colors) {
    int pt = get_global_id(0);
    if (pt >= num_points) return;

    double t = (stress_values[pt] - min_stress) / (max_stress - min_stress);
    t = fmax(0.0, fmin(1.0, t));

    // Jet colormap
    colors[pt * 4] = (uchar)(255.0 * fmax(0.0, fmin(1.0, 4.0*t - 1.5)));
    colors[pt * 4 + 1] = (uchar)(255.0 * fmax(0.0, fmin(1.0, 2.0 - fabs(4.0*(t - 0.5)))));
    colors[pt * 4 + 2] = (uchar)(255.0 * fmax(0.0, fmin(1.0, 1.5 - 4.0*(1.0 - t))));
    colors[pt * 4 + 3] = 255; // Alpha
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_postprocessing_cuda_kernels_exist() {
        assert!(!CUDA_NODE_ROTATE_KERNEL.is_empty());
        assert!(!CUDA_NODE_SCALE_KERNEL.is_empty());
        assert!(!CUDA_STRESS_TO_PRINCIPAL_KERNEL.is_empty());
        assert!(!CUDA_STRAIN_TO_VON_MISES_KERNEL.is_empty());
        assert!(!CUDA_DISPLACEMENT_MAGNITUDE_KERNEL.is_empty());
        assert!(!CUDA_REACTION_FORCE_KERNEL.is_empty());
        assert!(!CUDA_ELEMENT_FORCE_KERNEL.is_empty());
        assert!(!CUDA_CONTOUR_LEVEL_KERNEL.is_empty());
        assert!(!CUDA_COLOR_MAP_KERNEL.is_empty());
        assert!(!CUDA_MESH_QUALITY_KERNEL.is_empty());
    }

    #[test]
    fn test_postprocessing_opencl_kernels_exist() {
        assert!(!OPENCL_NODE_TRANSFORM_KERNEL.is_empty());
        assert!(!OPENCL_STRESS_VISUALIZATION_KERNEL.is_empty());
    }

    #[test]
    fn test_kernel_syntax() {
        // Check for CUDA patterns
        assert!(CUDA_COLOR_MAP_KERNEL.contains("__global__"));
        assert!(CUDA_COLOR_MAP_KERNEL.contains("fmax"));
        assert!(CUDA_COLOR_MAP_KERNEL.contains("fmin"));

        // Check for OpenCL patterns
        assert!(OPENCL_STRESS_VISUALIZATION_KERNEL.contains("__kernel"));
        assert!(OPENCL_STRESS_VISUALIZATION_KERNEL.contains("get_global_id"));
    }

    #[test]
    fn test_kernel_documentation() {
        // Verify kernels have proper naming
        assert!(CUDA_NODE_ROTATE_KERNEL.contains("node_rotate"));
        assert!(CUDA_COLOR_MAP_KERNEL.contains("color_map_jet"));
    }
}

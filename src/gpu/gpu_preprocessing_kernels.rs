//! Advanced GPU Pre-processing Kernels.
//!
//! This module provides CUDA/OpenCL kernels for pre-processing:
//! - Mesh generation on GPU
//! - Node coordinate transformations
//! - Element quality computation
//! - Boundary condition application
//! - Material property assignment

/// CUDA kernel for 1D mesh generation
pub const CUDA_MESH_1D_KERNEL: &str = r#"
extern "C" __global__
void mesh_1d(const int num_elements,
             const double start_x,
             const double end_x,
             double* node_coords) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    int total_nodes = num_elements + 1;

    if (node >= total_nodes) return;

    double dx = (end_x - start_x) / num_elements;
    node_coords[node * 3] = start_x + node * dx;
    node_coords[node * 3 + 1] = 0.0;
    node_coords[node * 3 + 2] = 0.0;
}
"#;

/// CUDA kernel for 2D structured quad mesh generation
pub const CUDA_MESH_2D_QUAD_KERNEL: &str = r#"
extern "C" __global__
void mesh_2d_quad(const int nx,
                  const int ny,
                  const double start_x,
                  const double start_y,
                  const double end_x,
                  const double end_y,
                  double* node_coords) {
    int node_x = blockIdx.x * blockDim.x + threadIdx.x;
    int node_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (node_x > nx || node_y > ny) return;

    int node_id = node_y * (nx + 1) + node_x;
    double dx = (end_x - start_x) / nx;
    double dy = (end_y - start_y) / ny;

    node_coords[node_id * 3] = start_x + node_x * dx;
    node_coords[node_id * 3 + 1] = start_y + node_y * dy;
    node_coords[node_id * 3 + 2] = 0.0;
}
"#;

/// CUDA kernel for 3D structured hex mesh generation
pub const CUDA_MESH_3D_HEX_KERNEL: &str = r#"
extern "C" __global__
void mesh_3d_hex(const int nx,
                 const int ny,
                 const int nz,
                 const double start_x,
                 const double start_y,
                 const double start_z,
                 const double end_x,
                 const double end_y,
                 const double end_z,
                 double* node_coords) {
    int node_x = blockIdx.x * blockDim.x + threadIdx.x;
    int node_y = blockIdx.y * blockDim.y + threadIdx.y;
    int node_z = blockIdx.z * blockDim.z + threadIdx.z;

    if (node_x > nx || node_y > ny || node_z > nz) return;

    int node_id = node_z * (nx + 1) * (ny + 1) + node_y * (nx + 1) + node_x;
    double dx = (end_x - start_x) / nx;
    double dy = (end_y - start_y) / ny;
    double dz = (end_z - start_z) / nz;

    node_coords[node_id * 3] = start_x + node_x * dx;
    node_coords[node_id * 3 + 1] = start_y + node_y * dy;
    node_coords[node_id * 3 + 2] = start_z + node_z * dz;
}
"#;

/// CUDA kernel for nodal rotation about arbitrary axis
pub const CUDA_NODE_ROTATE_AXIS_KERNEL: &str = r#"
extern "C" __global__
void node_rotate_axis(const int num_nodes,
                      const double* input_coords,
                      double* output_coords,
                      const double axis_x,
                      const double axis_y,
                      const double axis_z,
                      const double angle_rad) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    double x = input_coords[node * 3];
    double y = input_coords[node * 3 + 1];
    double z = input_coords[node * 3 + 2];

    // Rodrigues' rotation formula
    double c = cos(angle_rad);
    double s = sin(angle_rad);
    double t = 1.0 - c;

    // Normalize axis
    double len = sqrt(axis_x*axis_x + axis_y*axis_y + axis_z*axis_z);
    double ux = axis_x / len;
    double uy = axis_y / len;
    double uz = axis_z / len;

    // Rotation matrix
    double r00 = t*ux*ux + c;
    double r01 = t*ux*uy - s*uz;
    double r02 = t*ux*uz + s*uy;
    double r10 = t*ux*uy + s*uz;
    double r11 = t*uy*uy + c;
    double r12 = t*uy*uz - s*ux;
    double r20 = t*ux*uz - s*uy;
    double r21 = t*uy*uz + s*ux;
    double r22 = t*uz*uz + c;

    output_coords[node * 3] = r00*x + r01*y + r02*z;
    output_coords[node * 3 + 1] = r10*x + r11*y + r12*z;
    output_coords[node * 3 + 2] = r20*x + r21*y + r22*z;
}
"#;

/// CUDA kernel for mesh quality (aspect ratio)
pub const CUDA_MESH_ASPECT_RATIO_KERNEL: &str = r#"
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

/// CUDA kernel for mesh quality (skew angle)
pub const CUDA_MESH_SKEW_KERNEL: &str = r#"
extern "C" __global__
void mesh_skew_angle(const int num_elements,
                     const double* node_coords,
                     const int* connectivity,
                     double* skew_angles) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem >= num_elements) return;

    // For quad elements (4 nodes)
    int n0 = connectivity[elem * 4];
    int n1 = connectivity[elem * 4 + 1];
    int n2 = connectivity[elem * 4 + 2];
    int n3 = connectivity[elem * 4 + 3];

    // Compute vectors at corner 0
    double v01_x = node_coords[n1 * 3] - node_coords[n0 * 3];
    double v01_y = node_coords[n1 * 3 + 1] - node_coords[n0 * 3 + 1];
    double v03_x = node_coords[n3 * 3] - node_coords[n0 * 3];
    double v03_y = node_coords[n3 * 3 + 1] - node_coords[n0 * 3 + 1];

    // Compute angle
    double dot = v01_x * v03_x + v01_y * v03_y;
    double mag01 = sqrt(v01_x*v01_x + v01_y*v01_y);
    double mag03 = sqrt(v03_x*v03_x + v03_y*v03_y);

    double cos_angle = dot / (mag01 * mag03);
    cos_angle = fmax(-1.0, fmin(1.0, cos_angle));
    double angle = acos(cos_angle) * 180.0 / 3.14159265358979;

    // Skew = |90 - angle|
    skew_angles[elem] = fabs(90.0 - angle);
}
"#;

/// CUDA kernel for applying boundary conditions
pub const CUDA_APPLY_BC_KERNEL: &str = r#"
extern "C" __global__
void apply_bc(const int num_constraints,
              const int* constraint_indices,
              const double* constraint_values,
              double* global_rhs,
              double* global_matrix,
              const int n_dofs) {
    int constraint = blockIdx.x * blockDim.x + threadIdx.x;
    if (constraint >= num_constraints) return;

    int dof = constraint_indices[constraint];
    double value = constraint_values[constraint];

    // Set RHS
    global_rhs[dof] = value;

    // Zero out row and column
    for (int j = 0; j < n_dofs; j++) {
        global_matrix[dof * n_dofs + j] = 0.0;
        global_matrix[j * n_dofs + dof] = 0.0;
    }

    // Set diagonal to 1
    global_matrix[dof * n_dofs + dof] = 1.0;
}
"#;

/// CUDA kernel for material property assignment
pub const CUDA_MATERIAL_ASSIGN_KERNEL: &str = r#"
extern "C" __global__
void material_assign(const int num_elements,
                     const int* element_material_ids,
                     const double* material_E,
                     const double* material_nu,
                     const double* material_rho,
                     double* element_E,
                     double* element_nu,
                     double* element_rho) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem >= num_elements) return;

    int mat_id = element_material_ids[elem];
    element_E[elem] = material_E[mat_id];
    element_nu[elem] = material_nu[mat_id];
    element_rho[elem] = material_rho[mat_id];
}
"#;

/// CUDA kernel for element volume computation
pub const CUDA_ELEMENT_VOLUME_KERNEL: &str = r#"
extern "C" __global__
void element_volume(const int num_elements,
                    const int element_type,
                    const double* node_coords,
                    const int* connectivity,
                    double* volumes) {
    int elem = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem >= num_elements) return;

    if (element_type == 0) { // Truss (2 nodes)
        int n0 = connectivity[elem * 2];
        int n1 = connectivity[elem * 2 + 1];

        double dx = node_coords[n1 * 3] - node_coords[n0 * 3];
        double dy = node_coords[n1 * 3 + 1] - node_coords[n0 * 3 + 1];
        double dz = node_coords[n1 * 3 + 2] - node_coords[n0 * 3 + 2];

        volumes[elem] = sqrt(dx*dx + dy*dy + dz*dz);
    } else if (element_type == 1) { // Quad (4 nodes)
        // Compute area using cross product
        int n0 = connectivity[elem * 4];
        int n1 = connectivity[elem * 4 + 1];
        int n2 = connectivity[elem * 4 + 2];

        double v01_x = node_coords[n1 * 3] - node_coords[n0 * 3];
        double v01_y = node_coords[n1 * 3 + 1] - node_coords[n0 * 3 + 1];
        double v02_x = node_coords[n2 * 3] - node_coords[n0 * 3];
        double v02_y = node_coords[n2 * 3 + 1] - node_coords[n0 * 3 + 1];

        // Triangle 1 area
        double area1 = 0.5 * fabs(v01_x * v02_y - v01_y * v02_x);

        // Triangle 2 (n0, n2, n3)
        int n3 = connectivity[elem * 4 + 3];
        double v03_x = node_coords[n3 * 3] - node_coords[n0 * 3];
        double v03_y = node_coords[n3 * 3 + 1] - node_coords[n0 * 3 + 1];
        double area2 = 0.5 * fabs(v02_x * v03_y - v02_y * v03_x);

        volumes[elem] = area1 + area2;
    }
}
"#;

/// OpenCL kernel for mesh generation
pub const OPENCL_MESH_GENERATION_KERNEL: &str = r#"
__kernel void mesh_generation(const int nx,
                              const int ny,
                              const int nz,
                              const double dx,
                              const double dy,
                              const double dz,
                              __global double* node_coords) {
    int node_x = get_global_id(0);
    int node_y = get_global_id(1);
    int node_z = get_global_id(2);

    if (node_x > nx || node_y > ny || node_z > nz) return;

    int node_id = node_z * (nx + 1) * (ny + 1) + node_y * (nx + 1) + node_x;

    node_coords[node_id * 3] = node_x * dx;
    node_coords[node_id * 3 + 1] = node_y * dy;
    node_coords[node_id * 3 + 2] = node_z * dz;
}
"#;

/// OpenCL kernel for coordinate transformation
pub const OPENCL_COORD_TRANSFORM_KERNEL: &str = r#"
__kernel void coord_transform(const int num_nodes,
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

/// OpenCL kernel for mesh quality
pub const OPENCL_MESH_QUALITY_KERNEL: &str = r#"
__kernel void mesh_quality(const int num_elements,
                           __global const double* node_coords,
                           __global const int* connectivity,
                           __global double* aspect_ratios,
                           __global double* skew_angles) {
    int elem = get_global_id(0);
    if (elem >= num_elements) return;

    // Compute aspect ratio
    int n0 = connectivity[elem * 4];
    int n1 = connectivity[elem * 4 + 1];

    double dx = node_coords[n1 * 3] - node_coords[n0 * 3];
    double dy = node_coords[n1 * 3 + 1] - node_coords[n0 * 3 + 1];
    double dz = node_coords[n1 * 3 + 2] - node_coords[n0 * 3 + 2];

    double edge_len = sqrt(dx*dx + dy*dy + dz*dz);
    aspect_ratios[elem] = edge_len;

    // Simplified skew calculation
    skew_angles[elem] = 0.0;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_preprocessing_kernels_exist() {
        assert!(!CUDA_MESH_1D_KERNEL.is_empty());
        assert!(!CUDA_MESH_2D_QUAD_KERNEL.is_empty());
        assert!(!CUDA_MESH_3D_HEX_KERNEL.is_empty());
        assert!(!CUDA_NODE_ROTATE_AXIS_KERNEL.is_empty());
        assert!(!CUDA_MESH_ASPECT_RATIO_KERNEL.is_empty());
        assert!(!CUDA_MESH_SKEW_KERNEL.is_empty());
        assert!(!CUDA_APPLY_BC_KERNEL.is_empty());
        assert!(!CUDA_MATERIAL_ASSIGN_KERNEL.is_empty());
        assert!(!CUDA_ELEMENT_VOLUME_KERNEL.is_empty());
    }

    #[test]
    fn test_opencl_preprocessing_kernels_exist() {
        assert!(!OPENCL_MESH_GENERATION_KERNEL.is_empty());
        assert!(!OPENCL_COORD_TRANSFORM_KERNEL.is_empty());
        assert!(!OPENCL_MESH_QUALITY_KERNEL.is_empty());
    }

    #[test]
    fn test_kernel_syntax() {
        // CUDA syntax
        assert!(CUDA_MESH_3D_HEX_KERNEL.contains("__global__"));
        assert!(CUDA_MESH_3D_HEX_KERNEL.contains("blockIdx"));
        assert!(CUDA_MESH_3D_HEX_KERNEL.contains("threadIdx"));

        // OpenCL syntax
        assert!(OPENCL_MESH_GENERATION_KERNEL.contains("__kernel"));
        assert!(OPENCL_MESH_GENERATION_KERNEL.contains("get_global_id"));
    }

    #[test]
    fn test_kernel_documentation() {
        assert!(CUDA_MESH_1D_KERNEL.contains("mesh_1d"));
        assert!(CUDA_APPLY_BC_KERNEL.contains("apply_bc"));
    }
}

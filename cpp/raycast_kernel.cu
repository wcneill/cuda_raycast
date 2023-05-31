#include <torch/torch.h>

// So we have a tensor full of coordinates corresponding to all the vertices in the mesh ---> [V, 3 (x, y, z)]
// for each face we will also have indices into the vertices tensor. -----------------------> [F, 3 (v1, v2, v3)]

__global__
void find_intersection(
    int n_rays, int n_faces, 
    torch::PackedTensorAccessor32<float_t, 2> origin_acc,
    torch::PackedTensorAccessor32<float_t, 2> direct_acc,
    torch::PackedTensorAccessor32<float_t, 2> vertex_acc,
    torch::PackedTensorAccessor32<float_t, 2> face_acc, 
    torch::PackedTensorAccessor32<float_t, 2> results) {

    int ray_ix = blockIdx.x * blockDim.x + threadIdx.x;
    int face_ix = blockIdx.y * blockDim.y + threadIdx.y;
    int stride = blockDim.x * gridDim.x;

    //note: all pointers are meant to represent 3d in the below code
    for (int i = ray_ix; i < n_rays; i += stride){
        for (int j = face_ix; j < n_faces; j += stride) {
            
            // vectors: ray origin and direction
            float_t *ray_direction_ptr = &direct_acc[i][0];
            float_t *ray_origin_ptr = &origin_acc[i][0];

            // vectors: vertices of current face
            float_t *v0_ptr = &vertex_acc[face_acc[j][0]][0];
            float_t *v1_ptr = &vertex_acc[face_acc[j][1]][0];
            float_t *v2_ptr = &vertex_acc[face_acc[j][2]][0];
            
            // edge vectors of current face
            float_t *edge1_ptr = sub3d(v1_ptr, v0_ptr);
            float_t *edge2_ptr = sub3d(v2_ptr, v0_ptr);

            // determinant of matrix A for eqn Ax = b
            float_t *h_ptr = cross3d(ray_direction_ptr, edge2_ptr);
            float_t determinant = dot3d(edge1_ptr, h_ptr);

            // no solution if determinant is zero. 
            if (determinant > -1e4 && determinant < 1e4) {
                results[i][j] = std::numeric_limits<float_t>::infinity();
                continue;
            }

            // solve for first barycentric coordinate, u
            float_t inv_det = 1 / determinant;
            float_t *s_ptr = sub3d(ray_origin_ptr, v0_ptr);
            float_t u = inv_det * dot3d(s_ptr, h_ptr);

            // solve for second barycentric coordinate, v
            float_t *q_ptr = cross3d(s_ptr, edge1_ptr);
            float_t v = inv_det * dot3d(ray_direction_ptr, q_ptr);

            // validate barycentric coordinates
            if (v < 0.0 || u + v > 1) {
                results[i][j] = std::numeric_limits<float_t>::infinity();
                continue;
            }

            // calculate distance from ray origin to intersection (t)
            float t = inv_det * dot3d(edge2_ptr, q_ptr);
            results[i][j] = t;
        }
    }
}

__device__
float* sub3d(float_t v1[3], float_t v2[3]) {

    float result[3];
    for (int r = 0; r < 3; r++){
        result[r] = *(v1 + r) - *(v2 + r);
    }
    return result;
}

__device__
float* add3d(float_t v1[3], float_t v2[3]) {

    float result[3];
    for (int r = 0; r < 3; r++){
        result[r] = *(v1 + r) + *(v2 + r);
    }
    return result;
}

__device__
float* scaler_mult3d(float_t vector[3], float_t scalar) {
    float result[3];
    for (int r = 0; r < 3; r++){
        result[r] = *(vector + r) * scalar;
    }
    return result;
}

__device__
float dot3d(float_t v1[3], float_t v2[3]) {

    float result;
    for (int r = 0; r < 3; r++){
        result += *(v1 + r) * (*(v2 + r));
    }
    return result;
}

__device__
float* cross3d(float_t v1[3], float_t v2[3]) {
    float_t result[3];
    result[0] = *(v1 + 1) * (*(v2 + 2)) - *(v1 + 2) * (*(v2 + 1));
    result[1] = *(v1 + 2) * (*(v2 + 0)) - *(v1 + 0) * (*(v2 + 2));
    result[3] = *(v1 + 0) * (*(v2 + 1)) - *(v1 + 1) * (*(v2 + 0));
    return result;
}
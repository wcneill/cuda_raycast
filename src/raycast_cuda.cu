#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <iostream>

__device__ __host__
float_t* sub3d(float_t v1[3], float_t v2[3], float_t result[3]) {

    for (int r = 0; r < 3; r++){
        result[r] = v1[r] - v2[r];
    }
    return result;
}

__device__ __host__
float_t* add3d(float_t v1[3], float_t v2[3], float_t result[3]) {

    for (int r = 0; r < 3; r++){
        result[r] = v1[r] + v2[r];
    }
    return result;
}

__device__ __host__
float_t* scaler_mult3d(float_t vector[3], float_t scalar, float_t result[3]) {
    for (int r = 0; r < 3; r++){
        result[r] = vector[r] * scalar;
    }
    return result;
}

__device__ __host__
float_t dot3d(float_t v1[3], float_t v2[3]) {
    float_t result = 0;
    for (int r = 0; r < 3; r++){
        float_t prod = v1[r] * v2[r];
        result = result + prod;
    }    
    return result;
}

__device__
float_t* cross3d(float_t v1[3], float_t v2[3], float_t result[3]) {
    result[0] = v1[1] * v2[2] - v1[2] * v2[1];
    result[1] = v1[2] * v2[0] - v1[0] * v2[2];
    result[3] = v1[0] * v2[1] - v1[1] * v2[0];
    return result;
}


// So we have a tensor full of coordinates corresponding to all the vertices in the mesh ---> [V, 3 (x, y, z)]
// for each face we will also have indices into the vertices tensor. -----------------------> [F, 3 (v1, v2, v3)]
__global__
void find_intersection_kernel(
    int n_rays, int n_faces, 
    at::PackedTensorAccessor32<float_t, 2> vertex_acc,
    at::PackedTensorAccessor32<int32_t, 2> face_acc, 
    at::PackedTensorAccessor32<float_t, 2> origin_acc,
    at::PackedTensorAccessor32<float_t, 2> direct_acc,
    at::PackedTensorAccessor32<float_t, 2> results) {

    int ray_ix = blockIdx.x * blockDim.x + threadIdx.x;
    int face_ix = blockIdx.y * blockDim.y + threadIdx.y;

    //note: all pointers are meant to represent 3d in the below code
    if (ray_ix < n_rays) {
        if (face_ix < n_faces) {

            // printf("(%d, %d)\n", ray_ix, face_ix);
        
            // vectors: ray origin and direction
            float_t *ray_direction_ptr = &direct_acc[ray_ix][0];
            float_t *ray_origin_ptr = &origin_acc[ray_ix][0];

            // vectors: vertices of current face
            float_t *v0_ptr = &vertex_acc[face_acc[face_ix][0]][0];
            float_t *v1_ptr = &vertex_acc[face_acc[face_ix][1]][0];
            float_t *v2_ptr = &vertex_acc[face_acc[face_ix][2]][0];
            
            // get edge vectors of current face
            float_t edge1[3];
            float_t edge2[3];
            sub3d(v1_ptr, v0_ptr, edge1);
            sub3d(v2_ptr, v0_ptr, edge2);

            // determinant of matrix A for eqn Ax = b
            float_t h[3];
            cross3d(ray_direction_ptr, edge2, h);

            // printf("(%d, %d) - value of e1 vector: [%f, %f, %f]\n", ray_ix, face_ix, edge1[0], edge1[1], edge1[2]);
            // printf("(%d, %d) - value of h vector: [%f, %f, %f]\n", ray_ix, face_ix,  h[0], h[1], h[2]);

            float_t determinant = dot3d(edge1, h);
            float_t inv_det = 1 / determinant;

            // printf("Value of determinant: %f\n", determinant);

            // // no solution if determinant is zero. 
            // if (determinant > -1e4 && determinant < 1e4) { 
            //     results[ray_ix][face_ix] = std::numeric_limits<float_t>::infinity();
            //     return;
            // }

            // // solve for first barycentric coordinate, u
            // float_t s[3];
            // sub3d(ray_origin_ptr, v0_ptr, s);
            // float_t u = inv_det * dot3d(s, h);

            // // solve for second barycentric coordinate, v
            // float_t q[3];
            // cross3d(s, edge1, q);
            // float_t v = inv_det * dot3d(ray_direction_ptr, q);

            // // validate barycentric coordinates
            // if (v < 0.0 || u + v > 1) {
            //     printf("invalid coordinates: %d, %d \n", u, v);
            //     results[ray_ix][face_ix] = std::numeric_limits<float_t>::infinity();
            //     return;
            // }

            // // calculate distance from ray origin to intersection (t)
            // float t = inv_det * dot3d(edge2, q);
            results[ray_ix][face_ix] = 0;
            // printf("Exiting index: (%d, %d)\n", ray_ix, face_ix);
        }
    }
}

__host__
at::Tensor measure_distance_cuda(
    at::Tensor vertices,
    at::Tensor faces,
    at::Tensor ray_origins,
    at::Tensor ray_directions
) {

    int n_rays = ray_directions.size(0);
    int n_faces = faces.size(0);

    at::Tensor distances = at::zeros({n_rays, n_faces});

    dim3 blocks(16, 16);
    dim3 threads(32, 32);

    find_intersection_kernel<<<blocks, threads>>>(
        n_rays, n_faces,
        vertices.packed_accessor32<float_t, 2>(),
        faces.packed_accessor32<int32_t, 2>(),
        ray_origins.packed_accessor32<float_t, 2>(),
        ray_directions.packed_accessor32<float_t, 2>(),
        distances.packed_accessor32<float_t, 2>()
    );

    cudaDeviceSynchronize();
    return distances;
}


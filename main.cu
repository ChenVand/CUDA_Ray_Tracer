// cspell: disable

#include <stdio.h>
#include <time.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

#include "rtweekend.h"

#include "hittable.h"
#include "material.h"
#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"

__global__ void generate_randoms(curandState_t* state, float* randoms) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curandState localState = state[tid];
    randoms[tid] = curand_uniform(&localState);
}

__global__ void create_world(hittable** world, material_list** mat_lst) {    //}, hittable** objects, int num_objects) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        // Materials
        const int num_materials = 4;
        material** materials = new material*[num_materials];

        materials[0] = new lambertian(color(0.8, 0.8, 0.0)); //ground
        materials[1] = new lambertian(color(0.1, 0.2, 0.5)); //center
        materials[2] = new metal(color(0.8, 0.8, 0.8), 0.3); //left
        materials[3] = new metal(color(0.8, 0.6, 0.2), 1.0); //right

        *mat_lst = new material_list(materials, num_materials); //"Owner" list


        // Objects
        const int num_objects = 4;
        hittable** objects = new hittable*[num_objects];

        objects[0] = new sphere(point3( 0.0, -100.5, -1.0), 100.0, materials[0]);
        objects[1] = new sphere(point3( 0.0,    0.0, -1.2),   0.5, materials[1]);
        objects[2] = new sphere(point3( -1.0,   0.0, -1.0),   0.5, materials[2]);
        objects[3] = new sphere(point3( 1.0,    0.0, -1.0),   0.5, materials[3]);

        *world = new hittable_list(objects, num_objects);
    }
}

__global__ void destroy_world(hittable** world, material_list** mat_lst) {   //}, hittable** objects, int num_objects) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete *world;
        delete *mat_lst;
    }
}


extern bool g_lambertian = true;

int main(int argc,char *argv[]) {
    /*exe_name image_width threads_per_block_x threads_per_block_y*/

    // External arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--lambertian") == 0 && i + 1 < argc) {
            g_lambertian = (strcmp(argv[i + 1], "true") == 0);
            i++; // Skip the next argument as it is the value
        }
    }

    // Camera preparation

    camera cam;

    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width  = (argc >1) ? atoi(argv[1]) : 400;
    cam.samples_per_pixel = (argc >2) ? atoi(argv[2]) : 32; //streches block x dim
    cam.initialize();

    // World

    hittable** world;
    cudaMalloc((void **)&world, sizeof(hittable*));
    material_list** mat_lst; //material packet for deallocation
    cudaMalloc((void **)&mat_lst, sizeof(material_list*));

    create_world<<<1,1>>>(world, mat_lst);
    cudaCheckErrors("create world kernel launch failed");
    cudaDeviceSynchronize();
    cudaCheckErrors("post-world-creation synchronization failed");

    // Render

    int pixels_per_block_x = (argc >3) ? atoi(argv[3]) : 2; //blockDim.x will be this times samples_per_pixel
    int pixels_per_block_y = (argc >4) ? atoi(argv[4]) : 8;
    float buffer_gen_time;

    std::cerr << "Rendering width " << cam.image_width << " image ";
    std::cerr << "with " << pixels_per_block_x*cam.samples_per_pixel << 
        "x" << pixels_per_block_y << " blocks.\n";

    cam.render(pixels_per_block_x, pixels_per_block_y, world, buffer_gen_time);
    
    std::cerr << "Buffer creation took " << buffer_gen_time << " seconds.\n";

    // Cleanup

    cudaDeviceSynchronize();
    cudaCheckErrors("final synchronization failed");
    destroy_world<<<1,1>>>(world, mat_lst);
    cudaCheckErrors("destroy world kernel launch failed");
    cudaFree(world);
    cudaFree(mat_lst);

}
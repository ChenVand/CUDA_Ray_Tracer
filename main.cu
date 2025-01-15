/*
cmake -B build
cmake --build build
build/inOneWeekend > image.ppm
*/ 

// cspell: disable

#include <stdio.h>
#include <time.h>
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>

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
#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"

__global__ void create_world(hittable** world, hittable** objects, int num_objects) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        objects[0] = new sphere(point3(0, 0, -1), 0.5);
        objects[1] = new sphere(point3(0, -100.5, -1), 100);
        *world = new hittable_list(objects, num_objects);
    }
}

__global__ void destroy_world(hittable** world, hittable** objects, int num_objects) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete *world;
        for (int i = 0; i < num_objects; i++) {
            delete objects[i];
        }
    }
}

int main(int argc,char *argv[]) {
    /*exe_name image_width threads_per_block_x threads_per_block_y*/

    // Camera preparation

    camera cam;

    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width  = (argc >1) ? atoi(argv[1]) : 400;
    cam.initialize();

    // World

    // device memory allocation for world and objects
    int num_objects = 2;
    hittable** world;
    cudaMalloc((void **)&world, sizeof(hittable*));
    hittable** objects;
    cudaMalloc((void **)&objects, sizeof(hittable*) * num_objects);
    create_world<<<1,1>>>(world, objects, num_objects);
    cudaDeviceSynchronize();
    cudaCheckErrors("post-world-creation synchronization failed");

    // Render

    int threads_per_block_x = (argc >2) ? atoi(argv[2]) : 8;
    int threads_per_block_y = (argc >3) ? atoi(argv[3]) : 8;
    float buffer_gen_time;

    std::cerr << "Rendering width " << cam.image_width << 
        "ratio" << cam.aspect_ratio << " image ";
    std::cerr << "with " << threads_per_block_x << 
        "x" << threads_per_block_y << " blocks.\n";

    cam.render(threads_per_block_x, threads_per_block_y, world, buffer_gen_time);
    
    std::cerr << "buffer creation took " << buffer_gen_time << " seconds.\n";

    // Cleanup

    cudaDeviceSynchronize();
    cudaCheckErrors("final synchronization failed");
    destroy_world<<<1,1>>>(world,
        objects,
        num_objects);
    cudaFree(world);
    cudaFree(objects);

}
/*
cmake -B build
cmake --build build
build/inOneWeekend > image.ppm
*/ 

// cspell: disable

#include <stdio.h>

#include "rtweekend.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"

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


__global__ void dummy_kernel(vec3 *fb, int size, hittable_list* world) {
    /*cam_deets: pixel00_loc, pixel_delta_u, pixel_delta_v, camera_center*/
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int pixel_index = y*10 + x;

    //debug
    // if (x%10==0 && y%10==0)
    // printf("reached here in render kernel for thread %d, %d. ", x, y);
    printf("x's of spheres are %d\n", world->size);

    if (pixel_index < size)
    fb[pixel_index] = vec3(0.0f,0.0f,1.0f);
}

// __managed__ vec3 cam_deets[4];
// __managed__ hittable_list world;

int main() {

    // World
    hittable_list* world;   
    cudaMallocManaged(&world, sizeof(hittable_list)); 
    cudaCheckErrors("world managed mem alloc failure");
    new (world) hittable_list(); // Placement new to call the constructor
    cudaCheckErrors("initialization error");

    int num_spheres = 2;
    sphere* spheres;
    cudaMallocManaged(&spheres, num_spheres*sizeof(hittable_list));
    cudaCheckErrors("spheres managed mem alloc failure");
    spheres[0] = sphere(point3(0,0,-1), 0.5);
    spheres[1] = sphere(point3(0,-100.5,-1), 100);
    cudaCheckErrors("initialization error");

    for (int i = 0; i < num_spheres; i++) {
        world->add(&spheres[i]);
    }
    cudaCheckErrors("initialization error");

    // //cam_deets: pixel00_loc, pixel_delta_u, pixel_delta_v, camera_center
    // vec3* cam_deets;
    // cudaMallocManaged(&cam_deets, 4*sizeof(vec3));
    // cudaCheckErrors("cam_deets managed mem alloc failure");
    // cam_deets[0] = pixel00_loc;
    // cam_deets[1] = pixel_delta_u;
    // cam_deets[2] = pixel_delta_v;
    // cam_deets[3] = camera_center;

    // allocate frame buffer
    size_t fb_size = 100*sizeof(vec3);
    vec3 *fb;
    // cudaMalloc(&fb, fb_size);
    // cudaMemcpy(d_cam_deets, &h_cam_deets, 4 * sizeof(vec3), cudaMemcpyHostToDevice);
    cudaMallocManaged(&fb, fb_size);
    cudaCheckErrors("frame buffer managed mem alloc failure");

    // block size
    int tx = 8;
    int ty = 8;

    // Render our buffer
    dim3 blocks(10/tx+1,10/ty+1);
    dim3 threads(tx,ty);

    //debug
    dummy_kernel<<<blocks, threads>>>(fb, fb_size, world);

    // cudaDeviceSynchronize();
    // cudaCheckErrors("device sync failure");
    // // cudaMemPrefetchAsync(fb, fb_size, cudaCpuDeviceId);
    // // cudaCheckErrors("device sync failure");

    

    // // Print

    // std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    // for (int j = 0; j < image_height; j++) {
    //     for (int i = 0; i < image_width; i++) {
    //         size_t pixel_index = j*image_width + i;
    //         auto pixel_color = fb[pixel_index];

    //         write_color(std::cout, pixel_color);
    //     }
    // }

    // Cleanup
    cudaFree(fb);
    
    return 0;
}
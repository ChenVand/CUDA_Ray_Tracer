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
        const int num_materials = 5;
        material** materials = new material*[num_materials];

        materials[0] = new lambertian(color(0.8, 0.2, 0.2)); //ground
        materials[1] = new lambertian(color(0.1, 0.2, 0.5)); //center
        materials[2] = new dielectric(1.50); //left
        materials[3] = new dielectric(1.00 / 1.50); //bubble
        materials[4] = new metal(color(0.7, 0.7, 0.7), 0.2); //right

        *mat_lst = new material_list(materials, num_materials); //"Owner" list


        // Objects
        const int num_objects = 5;
        hittable** objects = new hittable*[num_objects];

        objects[0] = new sphere(point3( 0.0, -100.5, -1.0), 100.0, materials[0]); //ground
        objects[1] = new sphere(point3( 0.0,    0.0, -1.2),   0.5, materials[1]); //center
        objects[2] = new sphere(point3( -1.0,   0.0, -1.0),   0.5, materials[2]); //left
        objects[3] = new sphere(point3( -1.0,   0.0, -1.0),   0.4, materials[3]); //bubble
        objects[4] = new sphere(point3( 1.0,    0.0, -1.0),   0.5, materials[4]); //right

        *world = new hittable_list(objects, num_objects);
    }
}

__global__ void create_world2(hittable** world, material_list** mat_lst) {    //}, hittable** objects, int num_objects) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        float R = cosf(pi/4);

        // Materials
        const int num_materials = 2;
        material** materials = new material*[num_materials];

        materials[0] = new lambertian(color(0,0,1));
        materials[1] = new lambertian(color(1,0,0));

        *mat_lst = new material_list(materials, num_materials); //"Owner" list


        // Objects
        const int num_objects = 2;
        hittable** objects = new hittable*[num_objects];

        objects[0] = new sphere(point3(-R, 0, -1), R, materials[0]); //ground
        objects[1] = new sphere(point3( R, 0, -1), R, materials[1]); //center
        
        *world = new hittable_list(objects, num_objects);
    }
}

__global__ void destroy_world(hittable** world, material_list** mat_lst) {   //}, hittable** objects, int num_objects) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete *world;
        delete *mat_lst;
    }
}

// Tunable variables

// extern bool g_lambertian = true; //Try again by making constant
size_t g_image_width = 400;
size_t g_samples_per_pixel = 32;
int g_threads_x = 2 * g_samples_per_pixel;
int g_threads_y = 8;

int main(int argc,char *argv[]) {
    /*exe_name image_width threads_per_block_x threads_per_block_y*/

    // External arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            g_image_width = atoi(argv[i + 1]);
            i++; // Skip the next argument as it is the value
        } else if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            g_samples_per_pixel = atoi(argv[i + 1]);
            i++; // Skip the next argument as it is the value
        } else if (strcmp(argv[i], "--threads") == 0 && i + 2 < argc) {
            g_threads_x = atoi(argv[i + 1]);
            g_threads_y = atoi(argv[i + 2]);
            i+=2; // Skip the next argument as it is the value
        } 
        // else if (strcmp(argv[i], "--lambertian") == 0 && i + 1 < argc) {
        //     g_lambertian = !(strcmp(argv[i + 1], "false") == 0);
        //     i++; // Skip the next argument as it is the value
        // } 
        else {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
            return 1;
        }
    }

    // Camera preparation

    camera cam;

    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width  = g_image_width;
    cam.samples_per_pixel = g_samples_per_pixel; //streches block x dim
    //cam.max_depth = 50; // Not used in this version

    cam.vfov = 20; // Zoom with range >0 (close up) to <180 (far away)
    cam.lookfrom = point3(-2,2,1);
    cam.lookat   = point3(0,0,-1);
    cam.vup      = vec3(0,1,0);

    cam.defocus_angle = 10.0;
    cam.focus_dist    = 3.4;

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

    int pixels_per_block_x = (g_threads_x + g_samples_per_pixel - 1)/g_samples_per_pixel; //blockDim.x will be this times samples_per_pixel
    int pixels_per_block_y = g_threads_y;

    std::cerr << "Rendering width " << cam.image_width << " image ";
    std::cerr << "with " << pixels_per_block_x*cam.samples_per_pixel << 
        "x" << pixels_per_block_y << " blocks.\n";

    float buffer_gen_time;
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
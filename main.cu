// // cspell: disable

// #include <stdio.h>
// #include <time.h>
// #include <device_launch_parameters.h>
// #include <curand.h>
// #include <curand_kernel.h>
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>

// // error checking macro
// #define cudaCheckErrors(msg) \
//     do { \
//         cudaError_t __err = cudaGetLastError(); \
//         if (__err != cudaSuccess) { \
//             fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
//                 msg, cudaGetErrorString(__err), \
//                 __FILE__, __LINE__); \
//             fprintf(stderr, "*** FAILED - ABORTING\n"); \
//             exit(1); \
//         } \
//     } while (0)


// #include "rtweekend.h"
// #include "hittable.h"
// #include "material.h"
// #include "sphere.h"
// #include "hittable_list.h"
// #include "bvh.h"
// #include "camera.h"
// #include "render_with_cuda.h"

// #include "helper.h"
// // #include "helper_experimental.h"


// // Tunable variables

// // extern bool g_lambertian = true; //Try again by making constant
// size_t g_image_width = 400;
// size_t g_samples_per_pixel = 128;
// int g_threads_x = g_samples_per_pixel;
// int g_threads_y = 8;

// int main(int argc,char *argv[]) {
//     /*exe_name image_width threads_per_block_x threads_per_block_y*/

//     // External arguments
//     for (int i = 1; i < argc; ++i) {
//         if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
//             g_image_width = atoi(argv[i + 1]);
//             i++; // Skip the next argument as it is the value
//         } else if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
//             g_samples_per_pixel = (atoi(argv[i + 1]) + 31)/32 * 32; //Round up to nearest multiple of 32
//             g_threads_x = g_samples_per_pixel;
//             g_threads_y = min(8, 1024/g_threads_x); //Max 1024 threads per block
//             i++; // Skip the next argument as it is the value
//         } 
//         // else if (strcmp(argv[i], "--threads") == 0 && i + 2 < argc) {
//         //     g_threads_x = atoi(argv[i + 1]);
//         //     g_threads_y = atoi(argv[i + 2]);
//         //     i+=2; // Skip the next argument as it is the value
//         // } 
//         // else if (strcmp(argv[i], "--lambertian") == 0 && i + 1 < argc) {
//         //     g_lambertian = !(strcmp(argv[i + 1], "false") == 0);
//         //     i++; // Skip the next argument as it is the value
//         // } 
//         else {
//             std::cerr << "Unknown argument: " << argv[i] << "\n";
//             return 1;
//         }
//     }

//     // Camera

//     camera* cam;
//     cudaMallocManaged((void **)&cam, sizeof(camera));
//     cudaDeviceSynchronize();
//     cudaCheckErrors("Camera managed mem alloc failure");

//     cam->aspect_ratio = 16.0 / 9.0;
//     cam->image_width  = g_image_width;
//     cam->samples_per_pixel = g_samples_per_pixel; //streches block x dim
//     //cam.max_depth = 50; // Not used in this version

//     cam->vfov = 20; // Zoom with range >0 (close up) to <180 (far away)
//     cam->lookfrom = point3(13,2,3);
//     cam->lookat   = point3(0,0,0);
//     cam->vup      = vec3(0,1,0);

//     cam->defocus_angle = 0.6;
//     cam->focus_dist    = 10.0;

//     cam->initialize();

//     // World

//     hittable** obj_lst;
//     cudaMallocManaged((void **)&obj_lst, sizeof(hittable*));
//     material_list** mat_lst; //material packet for deallocation
//     cudaMalloc((void **)&mat_lst, sizeof(material_list*));

//     create_world<<<1,1>>>(obj_lst, mat_lst);
//     cudaCheckErrors("create world kernel launch failed");
//     cudaDeviceSynchronize();
//     cudaCheckErrors("post-world-creation synchronization failed");

//     // Render

//     int pixels_per_block_x = (g_threads_x + g_samples_per_pixel - 1)/g_samples_per_pixel; //blockDim.x will be this times samples_per_pixel
//     int pixels_per_block_y = g_threads_y;

//     std::cerr << "Rendering width " << cam->image_width << " image ";
//     std::cerr << "with " << pixels_per_block_x*cam->samples_per_pixel << 
//         "x" << pixels_per_block_y << " blocks.\n";

//     float buffer_gen_time;
//     render(pixels_per_block_x, pixels_per_block_y, cam, obj_lst, buffer_gen_time);
    
//     std::cerr << "Buffer creation took " << buffer_gen_time << " seconds.\n";

//     // Cleanup

//     cudaDeviceSynchronize();
//     cudaCheckErrors("final synchronization failed");
//     destroy_world<<<1,1>>>(obj_lst, mat_lst);
//     cudaCheckErrors("destroy world kernel launch failed");
//     cudaFree(obj_lst);
//     cudaFree(mat_lst);
//     cudaFree(cam);

// }
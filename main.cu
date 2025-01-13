/*
cmake -B build
cmake --build build
build/inOneWeekend > image.ppm
*/ 

// cspell: disable

#include <stdio.h>
// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>

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


// #define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
// void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
//     if (result) {
//         std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
//         file << ":" << line << " '" << func << "' \n";
//         // Make sure we call CUDA Device Reset before exiting
//         cudaDeviceReset();
//         exit(99);
//     }
// }

__device__ color ray_color(const ray& r, const hittable* world) {
    hit_record rec;
    if (world->hit(r, 0, infinity, rec)) {
        return 0.5 * (rec.normal + color(1,1,1));
    }

    vec3 unit_direction = unit_vector(r.direction());
    float a = 0.5f*(unit_direction.y() + 1.0f);
    return (1.0f-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
}

__global__ void render(vec3 *fb, int max_x, int max_y, const vec3 *cam_deets, const hittable* world) {
        
    /*cam_deets: pixel00_loc, pixel_delta_u, pixel_delta_v, camera_center*/
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if((x >= max_x) || (y >= max_y)) return;
    int pixel_index = y*max_x + x;

    auto pixel_center = cam_deets[0] + (x * cam_deets[1]) + (y * cam_deets[2]);
    auto ray_direction = pixel_center - cam_deets[3];
    ray r(cam_deets[3], ray_direction);

    color pixel_color = ray_color(r, world);

    //debug
    // if (x%10==0 && y%10==0)
    printf("reached here for thread %d, %d\n pixel color %f,%f,%f\n", x, y, pixel_color[0], pixel_color[1], pixel_color[2]);

    fb[pixel_index] = pixel_color;

}

__global__ void dummy_kernel() {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //debug
    if (x%10==0 && y%10==0)
    printf("reached here in dummy kernel for thread %d, %d\n", x, y);
}

int main(int argc,char *argv[]) {

    // Image
    int image_width = (argc >1) ? atoi(argv[1]) : 16;
    auto aspect_ratio = 16.0 / 9.0;

    // Calculate the image height, and ensure that it's at least 1.
    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

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

    // Camera

    auto focal_length = 1.0;
    auto viewport_height = 2.0;
    auto viewport_width = viewport_height * (double(image_width)/image_height);
    auto camera_center = point3(0, 0, 0);

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    auto viewport_u = vec3(viewport_width, 0, 0);
    auto viewport_v = vec3(0, -viewport_height, 0);

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    auto pixel_delta_u = viewport_u / image_width;
    auto pixel_delta_v = viewport_v / image_height;

    // Calculate the location of the upper left pixel.
    auto viewport_upper_left = camera_center
                             - vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
    auto pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    // Render

    int num_pixels = image_width*image_height;

    //cam_deets: pixel00_loc, pixel_delta_u, pixel_delta_v, camera_center
    vec3* cam_deets;
    cudaMallocManaged(&cam_deets, 4*sizeof(vec3));
    cudaCheckErrors("cam_deets managed mem alloc failure");
    cam_deets[0] = pixel00_loc;
    cam_deets[1] = pixel_delta_u;
    cam_deets[2] = pixel_delta_v;
    cam_deets[3] = camera_center;

    // allocate frame buffer
    size_t fb_size = num_pixels*sizeof(vec3);
    vec3 *fb;
    // cudaMalloc(&fb, fb_size);
    // cudaMemcpy(d_cam_deets, &h_cam_deets, 4 * sizeof(vec3), cudaMemcpyHostToDevice);
    cudaMallocManaged(&fb, fb_size);
    cudaCheckErrors("frame buffer managed mem alloc failure");

    // block size
    int tx = 32;
    int ty = 8;

    // Render our buffer
    dim3 blocks(image_width/tx+1,image_height/ty+1);
    dim3 threads(tx,ty);
    // // cudaMemPrefetchAsync(fb, fb_size, 0);
    render<<<blocks, threads>>>(fb, image_width, image_height, cam_deets, world);
    // cudaCheckErrors("render kernel launch failure");

    // //debug
    // dummy_kernel<<<blocks, threads>>>();

    cudaDeviceSynchronize();
    // cudaCheckErrors("device sync failure");
    // // cudaMemPrefetchAsync(fb, fb_size, cudaCpuDeviceId);
    // cudaCheckErrors("device sync failure");

    

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
    world->clear();
    cudaFree(fb);
    cudaFree(spheres);
    cudaFree(cam_deets);
    
    return 0;
}
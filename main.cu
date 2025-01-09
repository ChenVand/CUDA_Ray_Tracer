/*
cmake -B build
cmake --build build
build/inOneWeekend > image.ppm
*/ 

#include <stdio.h>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "color.h"
#include "vec3.h"
#include "ray.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ color ray_color(const ray& r) {
    vec3 unit_direction = unit_vector(r.direction());
    float a = 0.5f*(unit_direction.y() + 1.0f);
    return (1.0f-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
}

__global__ void render(vec3 *fb, int max_x, int max_y, const vec3 *cam_deets) {
    /*cam_deets: pixel00_loc, pixel_delta_u, pixel_delta_v, camera_center*/
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;

    auto pixel_center = cam_deets[0] + (i * cam_deets[1]) + (j * cam_deets[2]);
    auto ray_direction = pixel_center - cam_deets[3];
    ray r(cam_deets[3], ray_direction);

    color pixel_color = ray_color(r);
    fb[pixel_index] = pixel_color;
}

int main() {

    // Image

    auto aspect_ratio = 16.0 / 9.0;
    int image_width = 400;

    // Calculate the image height, and ensure that it's at least 1.
    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

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
    vec3 h_cam_deets[4] = {pixel00_loc, pixel_delta_u, pixel_delta_v, camera_center};
    vec3* d_cam_deets;
    cudaMalloc(&d_cam_deets, 4 * sizeof(vec3));
    cudaMemcpy(d_cam_deets, &h_cam_deets, 4 * sizeof(vec3), cudaMemcpyHostToDevice);

    // allocate frame buffer
    size_t fb_size = num_pixels*sizeof(vec3);
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // block size
    int tx = 8;
    int ty = 8;

    // Render our buffer
    dim3 blocks(image_width/tx+1,image_height/ty+1);
    dim3 threads(tx,ty);
    render<<<blocks, threads>>>(fb, image_width, image_height, d_cam_deets);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Print

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            size_t pixel_index = j*image_width + i;
            auto pixel_color = fb[pixel_index];

            write_color(std::cout, pixel_color);
        }
    }
    checkCudaErrors(cudaFree(fb));
    checkCudaErrors(cudaFree(d_cam_deets));
}
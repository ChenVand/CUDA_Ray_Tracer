// cspell: disable
#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"
#include "material.h"
#include "render_with_cuda.h"

class camera {
  public:
    float  aspect_ratio      = 1.0;  // Ratio of image width over height
    int    image_width       = 100;  // Rendered image width in pixel count
    int    samples_per_pixel = 32;   // Count of random samples for each pixel
    // int    max_depth         = 10;   // Maximum number of ray bounces into scen
    
    float vfov = 90;                   // Vertical view angle (field of view)
    point3 lookfrom = point3(0,0,0);   // Point camera is looking from
    point3 lookat   = point3(0,0,-1);  // Point camera is looking at
    vec3   vup      = vec3(0,1,0);     // Camera-relative "up" direction

    void initialize();

    void render(int threads_x, int threads_y, hittable** world, float& timer_seconds) ;

//   private:
    int    image_height;         // Rendered image height
    float  pixel_samples_scale;  // Color scale factor for a sum of pixel samples
    point3 center;               // Camera center
    point3 pixel00_loc;          // Location of pixel 0, 0
    vec3   pixel_delta_u;        // Offset to pixel to the right
    vec3   pixel_delta_v;        // Offset to pixel below
    vec3   u, v, w;              // Camera frame basis vectors
    
    void display_frame(vec3* frame_buffer) {
    
        std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
        for (int j = 0; j < image_height; j++) {
            for (int i = 0; i < image_width; i++) {
                size_t pixel_index = j*image_width + i;
                auto pixel_color = frame_buffer[pixel_index];

                write_color(std::cout, pixel_color);
            }
        }
        cudaDeviceSynchronize();
        cudaCheckErrors("post-display device synchronization failed");
    }
};

void camera::initialize() {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        pixel_samples_scale = 1.0 / samples_per_pixel;

        center = lookfrom;

        // Determine viewport dimensions.
        auto focal_length = (lookfrom - lookat).length();
        auto theta = degrees_to_radians(vfov);
        auto h = std::tan(theta/2);
        auto viewport_height = 2 * h * focal_length;
        auto viewport_width = viewport_height * (double(image_width)/image_height);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        auto viewport_u = viewport_width * u;     // Vector across viewport horizontal edge
        auto viewport_v =  viewport_height * -v;  // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = center - (focal_length * w) - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    }

void camera::render(int pixels_per_block_x, 
                    int pixels_per_block_y, 
                    hittable** world, 
                    float& timer_seconds) {
    clock_t start, stop;
    
    dim3 blocks((image_width+pixels_per_block_x-1)/pixels_per_block_x,
                (image_height+pixels_per_block_y-1)/pixels_per_block_y);
    dim3 threads(pixels_per_block_x*samples_per_pixel, pixels_per_block_y);

    // Calculate the total number of threads
    int total_blocks = blocks.x * blocks.y * blocks.z;
    int threads_per_block = threads.x * threads.y * threads.z;
    int total_threads = total_blocks * threads_per_block;

    //set up random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, total_threads * sizeof(curandState));
    cudaCheckErrors("random states mem alloc failure");
    setup_random_states<<<blocks, threads>>>(d_rand_state, time(0));
    cudaDeviceSynchronize();
    cudaCheckErrors("setup_random_states kernel launch failed");

    //cam_deets: pixel00_loc, pixel_delta_u, pixel_delta_v, center
    vec3 h_cam_deets[4];
    vec3* d_cam_deets;
    cudaMalloc(&d_cam_deets, 4 * sizeof(vec3));
    cudaCheckErrors("d_cam_deets mem alloc failure");
    h_cam_deets[0] = pixel00_loc;
    h_cam_deets[1] = pixel_delta_u;
    h_cam_deets[2] = pixel_delta_v;
    h_cam_deets[3] = center;
    cudaMemcpy(d_cam_deets, h_cam_deets, 4 * sizeof(vec3), cudaMemcpyHostToDevice);

    // allocate frame buffer 
    size_t fb_size = image_width*image_height*sizeof(vec3);
    vec3 *frame_buffer;
    cudaMallocManaged((void **)&frame_buffer, fb_size);
    cudaCheckErrors("frame buffer managed mem alloc failure");

    start = clock();
    // launch render kernel
    cudaDeviceSynchronize();
    cudaCheckErrors("pre-kernel device synchronization failed");
    // cudaMemPrefetchAsync(frame_buffer, fb_size, 0);
    // cudaCheckErrors("frame buffer prefetch to GPU failed");
    render_kernel<<<blocks, threads>>>(
        frame_buffer,
        samples_per_pixel,
        pixel_samples_scale, 
        image_width,
        image_height,
        d_cam_deets, 
        world,
        d_rand_state);
    cudaCheckErrors("kernel launch error");
    cudaDeviceSynchronize();
    cudaCheckErrors("post-kernel device synchronization failed");
    // cudaMemPrefetchAsync(frame_buffer, fb_size, cudaCpuDeviceId);
    // cudaCheckErrors("frame buffer postfetch to CPU failed");
    stop = clock();

    // display frame
    display_frame(frame_buffer);

    cudaFree(d_rand_state);
    cudaFree(d_cam_deets);
    cudaFree(frame_buffer);

    timer_seconds = ((float)(stop - start)) / CLOCKS_PER_SEC;
}

#endif
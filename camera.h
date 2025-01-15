// cspell: disable
#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"

__device__ color ray_color(const ray& r, const hittable& world) {
        
        hit_record rec;
        // if ((*world)->hit(r, 0, infinity, rec)) {
        if (world.hit(r, interval(0, infinity), rec)) {
            return 0.5 * (rec.normal + color(1,1,1));
        }

        vec3 unit_direction = unit_vector(r.direction());
        float a = 0.5f*(unit_direction.y() + 1.0f);
        return (1.0f-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
}

__global__ void render_kernel(vec3 *fb, int max_x, int max_y, const vec3 *cam_deets, hittable** world) {

    /*cam_deets: pixel00_loc, pixel_delta_u, pixel_delta_v, camera_center*/
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if((x >= max_x) || (y >= max_y)) return;

    int pixel_index = y*max_x + x;

    auto pixel_center = cam_deets[0] + (x * cam_deets[1]) + (y * cam_deets[2]);
    auto ray_direction = pixel_center - cam_deets[3];
    ray r(cam_deets[3], ray_direction);

    color pixel_color = ray_color(r, **world);
    __syncthreads();
    fb[pixel_index] = pixel_color;

}

class camera {
  public:
    double aspect_ratio = 1.0;  // Ratio of image width over height
    int    image_width  = 100;  // Rendered image width in pixel count

    void initialize();

    void render(int threads_x, int threads_y, hittable** world, float& timer_seconds) ;

//   private:
    int    image_height;   // Rendered image height
    point3 center;         // Camera center
    point3 pixel00_loc;    // Location of pixel 0, 0
    vec3   pixel_delta_u;  // Offset to pixel to the right
    vec3   pixel_delta_v;  // Offset to pixel below
    
    void display_frame(vec3* fb) {
    
        std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
        for (int j = 0; j < image_height; j++) {
            for (int i = 0; i < image_width; i++) {
                size_t pixel_index = j*image_width + i;
                auto pixel_color = fb[pixel_index];

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

        center = point3(0, 0, 0);

        // Determine viewport dimensions.
        auto focal_length = 1.0;
        auto viewport_height = 2.0;
        auto viewport_width = viewport_height * (double(image_width)/image_height);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        auto viewport_u = vec3(viewport_width, 0, 0);
        auto viewport_v = vec3(0, -viewport_height, 0);

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left =
            center - vec3(0, 0, focal_length) - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    }

void camera::render(int threads_x, int threads_y, hittable** world, float& timer_seconds) {
    clock_t start, stop;
    
    dim3 blocks(image_width/threads_x+1,image_height/threads_y+1);
    dim3 threads(threads_x,threads_y);

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
    vec3 *fb;
    cudaMallocManaged((void **)&fb, fb_size);
    cudaCheckErrors("frame buffer managed mem alloc failure");

    start = clock();
    // launch render kernel
    cudaDeviceSynchronize();
    cudaCheckErrors("pre-kernel device synchronization failed");
    // cudaMemPrefetchAsync(fb, fb_size, 0);
    // cudaCheckErrors("frame buffer prefetch to GPU failed");
    render_kernel<<<blocks, threads>>>(fb, image_width, image_height, 
        d_cam_deets,
        world);
    cudaCheckErrors("kernel launch error");
    cudaDeviceSynchronize();
    cudaCheckErrors("post-kernel device synchronization failed");
    // cudaMemPrefetchAsync(fb, fb_size, cudaCpuDeviceId);
    // cudaCheckErrors("frame buffer postfetch to CPU failed");
    stop = clock();

    // display frame
    display_frame(fb);

    cudaFree(fb);

    timer_seconds = ((float)(stop - start)) / CLOCKS_PER_SEC;
}

#endif
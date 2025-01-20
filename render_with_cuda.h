#ifndef RENDERWCU_H
#define RENDERWCU_H

#include "camera.h"

__device__ color ray_color(curandState& rand_state, const ray& r, const hittable& world) {
    
    const int max_iter = 50;
    color attenuation_mult = vec3(1, 1, 1);
    ray current_ray = r;
    hit_record rec;
    ray scattered;
    color attenuation;
    // vec3 direction;
    for (int i=0; i<max_iter; i++) {
        if (world.hit(current_ray, interval(0.001, infinity), rec)) {
            if (rec.mat_ptr->scatter(rand_state, current_ray, rec, attenuation, scattered)) {
                attenuation_mult *= attenuation;
                current_ray = scattered;
            } else {
                return color(0, 0, 0);
            }
        } else {
            vec3 unit_direction = unit_vector(r.direction());
            float a = 0.5f*(unit_direction.y() + 1.0f);
            return attenuation_mult*((1.0f-a)*color(1.0, 1.0, 1.0) 
                    + a*color(0.5, 0.7, 1.0));
        }
    }
    return color(0, 0, 0);
}

__global__ void setup_random_states(curandState* state, unsigned long seed)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = y * gridDim.x * blockDim.x + x;
    curand_init(seed, tid, 100, &state[tid]);
}

__global__ void render_kernel(  
    vec3 *fb, // size: (image_width*samples_per_pixel) * image_height
    camera* cam,
    int image_width,
    int image_height,
    hittable** world,
    curandState* state) {

    /*Each warp belongs to a single pixel.
    cam_deets: [0] pixel00_loc, [1] pixel_delta_u, [2]pixel_delta_v, 
        [3] camera_center, [4] vec3(defocus_angle,0,0), [5] defocus_disk_u, [6] defocus_disk_v
    */

    // Preparation

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int global_tid = y * gridDim.x * blockDim.x + x;
    int local_tid = threadIdx.y * blockDim.x + threadIdx.x;
    int lane = local_tid % warpSize;
    // int warpID = local_tid / warpSize;

    int samples_per_pixel = cam->samples_per_pixel;
    float samples_scale = 1.0f / samples_per_pixel;
    int pixel_x = x / samples_per_pixel;
    int pixel_y = y;

    if((pixel_x >= image_width) || (pixel_y >= image_height)) return;

    int pixel_index = pixel_y*image_width + pixel_x;
    if (x % samples_per_pixel == 0) {
        //initialize frame buffer
        fb[pixel_index] = vec3(0, 0, 0);
    }
    __syncthreads();

    curandState loc_rand_state = state[global_tid];

    // Get random ray:
    // Construct a camera ray originating from the defocus disk and directed at a randomly
    // sampled point around the pixel location i, j.
    // cam_deets: [0] pixel00_loc, [1] pixel_delta_u, [2]pixel_delta_v, 
    // [3] camera_center, [4] vec3(defocus_angle,0,0), [5] defocus_disk_u, [6] defocus_disk_v

    // Get random offset for pixel
    float x_offset = curand_uniform(&loc_rand_state) - 0.5f;
    float y_offset = curand_uniform(&loc_rand_state) - 0.5f;

    auto pixel_sample = cam->pixel00_loc
                    + ((pixel_x + x_offset) * cam->pixel_delta_u) 
                    + ((pixel_y + y_offset) * cam->pixel_delta_v);

    // Get random point on defocus disk
    auto p = random_in_unit_disk(loc_rand_state);
    auto p_in_disk = cam->get_center() 
            + p.x()*cam->get_defocus_disk_u() 
            + p.y()*cam->get_defocus_disk_v();

    auto ray_origin = (cam->defocus_angle <= 0) ? cam->get_center() : p_in_disk;
    auto ray_direction = pixel_sample - ray_origin;

    ray r(ray_origin, ray_direction);


    // Get color

    color pixel_color = ray_color(loc_rand_state, r, **world);

    state[global_tid] = loc_rand_state; // "return local state" to source

    //warp-shuffle reduction
    float3 val = make_float3(pixel_color.r(), pixel_color.g(), pixel_color.b());
    __syncwarp();   
    for (int step = warpSize/2; step > 0; step >>= 1) {
        val.x += __shfl_down_sync(0xFFFFFFFF, val.x, step);
        val.y += __shfl_down_sync(0xFFFFFFFF, val.y, step);
        val.z += __shfl_down_sync(0xFFFFFFFF, val.z, step);
    }

    if (lane==0) {
        fb[pixel_index].atomicAddVec3(vec3(val.x, val.y, val.z) * samples_scale); 
    }


}

#endif
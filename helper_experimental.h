__device__ color ray_color_experimental(curandState& rand_state, const ray& r, const bvh_world* world) {
    // //debug
    // printf("got here 7\n");
    const int max_iter = 50;
    color attenuation_mult = vec3(1, 1, 1);
    ray current_ray = r;
    hit_record rec;
    ray scattered;
    color attenuation;
    // vec3 direction;
    for (int i=0; i<max_iter; i++) {
        if (world->hit(current_ray, interval(0.001, infinity), rec)) {
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

__global__ void render_kernel_experimental(  
    vec3 *fb, // size: (image_width*samples_per_pixel) * image_height
    camera* cam,
    int image_width,
    int image_height,
    bvh_world* world,
    curandState* state) {

    /*Each warp belongs to a single pixel.*/

    // Preparation

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int global_tid = y * gridDim.x * blockDim.x + x;
    int local_tid = threadIdx.y * blockDim.x + threadIdx.x;
    int lane = local_tid % warpSize;
    // int warpID = local_tid / warpSize;

    int samples_per_pixel = cam->samples_per_pixel;
    int pixel_x = x / samples_per_pixel;
    int pixel_y = y;

    if((pixel_x >= image_width) || (pixel_y >= image_height)) return;

    int pixel_index = pixel_y*image_width + pixel_x;
    if (x % samples_per_pixel == 0) {
        //initialize frame buffer
        fb[pixel_index] = vec3(0, 0, 0);
    }
    __syncthreads();

    // Get ray and then color
    curandState loc_rand_state = state[global_tid];
    ray r = get_ray(loc_rand_state, *cam, pixel_x, pixel_y);
    // //Debug 
    // printf("Got here 5\n");
    color pixel_color = ray_color_experimental(loc_rand_state, r, world);
    //Debug 
    printf("Got here 6\n");
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
        fb[pixel_index].atomicAddVec3(vec3(val.x, val.y, val.z) * cam->get_pixel_samples_scale()); 
    }


}

void render_experimental(int pixels_per_block_x, 
                    int pixels_per_block_y, 
                    camera* cam,
                    bvh_world* world, 
                    float& timer_seconds) {

    clock_t start, stop;

    int image_width = cam->image_width;
    int image_height = cam->get_image_height();
    
    dim3 blocks((image_width+pixels_per_block_x-1)/pixels_per_block_x,
                (image_height+pixels_per_block_y-1)/pixels_per_block_y);
    dim3 threads(pixels_per_block_x*cam->samples_per_pixel, pixels_per_block_y);

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

    // allocate frame buffer 
    size_t fb_size = image_width*image_height*sizeof(vec3);
    vec3 *frame_buffer;
    cudaMallocManaged((void **)&frame_buffer, fb_size);
    cudaCheckErrors("frame buffer managed mem alloc failure");

    start = clock();
    // launch render kernel
    cudaDeviceSynchronize();
    cudaCheckErrors("pre-kernel device synchronization failed");
    cudaMemPrefetchAsync(frame_buffer, fb_size, 0);
    cudaCheckErrors("frame buffer prefetch to GPU failed");
    render_kernel_experimental<<<blocks, threads>>>(
        frame_buffer,
        cam,
        image_width,
        image_height,
        world,
        d_rand_state);
    cudaCheckErrors("kernel launch error");
    cudaDeviceSynchronize();
    cudaCheckErrors("post-kernel device synchronization failed");
    // cudaMemPrefetchAsync(frame_buffer, fb_size, cudaCpuDeviceId);
    // cudaCheckErrors("frame buffer postfetch to CPU failed");
    stop = clock();

    // display frame
    cam->display_frame(frame_buffer);

    cudaFree(d_rand_state);
    cudaFree(frame_buffer);

    timer_seconds = ((float)(stop - start)) / CLOCKS_PER_SEC;
}
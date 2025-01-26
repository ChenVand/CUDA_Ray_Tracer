__global__ void create_world_exp(int num_objects, hittable** obj_lst, material_list** mat_lst) {    //}, hittable** objects, int num_objects) {
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
        // const int num_objects = 5;
        hittable** objects = new hittable*[num_objects];

        obj_lst[0] = new sphere(point3( 0.0, -100.5, -1.0), 100.0, materials[0]); //ground
        obj_lst[1] = new sphere(point3( 0.0,    0.0, -1.2),   0.5, materials[1]); //center
        obj_lst[2] = new sphere(point3( -1.0,   0.0, -1.0),   0.5, materials[2]); //left
        obj_lst[3] = new sphere(point3( -1.0,   0.0, -1.0),   0.4, materials[3]); //bubble
        obj_lst[4] = new sphere(point3( 1.0,    0.0, -1.0),   0.5, materials[4]); //right

        // *world = new hittable_list(objects, num_objects);
    }
}

__global__ void create_world_exp2(hittable_list** obj_lst, material_list** mat_lst) {    //}, hittable** objects, int num_objects) {
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

        *obj_lst = new hittable_list(objects, num_objects);
    }
}

// void create_world_experimental(hittable* obj_lst, material_list* mat_lst) { 


//     int capacity = 5;
//     cudaMallocManaged((void **)&obj_lst, capacity*sizeof(hittable_list));

//     thrust::device_vector<material*> materials(capacity);
//     thrust::device_vector<hittable*> objects(capacity);
//     // material** materials = new material*[capacity];
//     // hittable** objects = new hittable*[capacity];

//     materials[0] = new lambertian(color(0.8, 0.2, 0.2)); //ground
//     materials[1] = new lambertian(color(0.1, 0.2, 0.5)); //center
//     materials[2] = new dielectric(1.50); //left
//     materials[3] = new dielectric(1.00 / 1.50); //bubble
//     materials[4] = new metal(color(0.7, 0.7, 0.7), 0.2); //right

//     // Allocate materials and objects
//     mat_lst = new material_list(thrust::raw_pointer_cast(materials.data()), counter); //"Owner" list
//     obj_lst = new hittable_list(thrust::raw_pointer_cast(objects.data()), counter);
// }

__global__ void destroy_world_experimental(hittable_list* obj_lst, material_list* mat_lst) {   //}, hittable** objects, int num_objects) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete obj_lst;
        delete mat_lst;
    }
}

__global__ void render_kernel_experimental(  
    vec3 *fb, // size: (image_width*samples_per_pixel) * image_height
    camera* cam,
    int image_width,
    int image_height,
    hittable** world,
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
    //Debug 
    printf("Got here 1");
    color pixel_color = ray_color(loc_rand_state, r, **world);
    //Debug 
    printf("Got here 2");
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
                    hittable** world, 
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
    // cudaMemPrefetchAsync(frame_buffer, fb_size, 0);
    // cudaCheckErrors("frame buffer prefetch to GPU failed");
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
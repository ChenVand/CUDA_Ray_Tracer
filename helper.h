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

__global__ void create_world2(hittable** world, material_list** mat_lst) { 
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        curandState_t rand_state;
        curand_init(18, 0, 0, &rand_state);

        const int capacity = 4 + 22*22;
        material** materials = new material*[capacity];
        hittable** objects = new hittable*[capacity];


        // Ground

        materials[0] = new lambertian(color(0.5, 0.5, 0.5));
        objects[0] = new sphere(point3(0,-1000,0), 1000, materials[0]);

        // Big spheres

        materials[1] = new dielectric(1.5);
        objects[1] = new sphere(point3(0, 1, 0), 1.0, materials[1]);

        materials[2] = new lambertian(color(0.4, 0.2, 0.1));
        objects[2] = new sphere(point3(-4, 1, 0), 1.0, materials[2]);

        materials[3] = new metal(color(0.7, 0.6, 0.5), 0.0);
        objects[3] = new sphere(point3(4, 1, 0), 1.0, materials[3]);

        // Random spheres

        int counter = 4;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {

                auto choose_mat = random_float(rand_state);
                point3 center(a + 0.9*random_float(rand_state), 0.2, b + 0.9*random_float(rand_state));

                if ((center - point3(4, 0.2, 0)).length() > 0.9) {

                    if (choose_mat < 0.8) {
                        // diffuse
                        auto albedo = color::random(rand_state) * color::random(rand_state);
                        materials[counter] = new lambertian(albedo);
                        auto center2 = center + vec3(0, random_float(rand_state, 0, .5), 0);
                        objects[counter] = new sphere(center, center2, 0.2, materials[counter]);
                    } else if (choose_mat < 0.95) {
                        // metal
                        auto albedo = color::random(rand_state, 0.5, 1);
                        auto fuzz = random_float(rand_state, 0, 0.5);
                        materials[counter] = new metal(albedo, fuzz);
                        objects[counter] = new sphere(center, 0.2, materials[counter]);
                    } else {
                        // glass
                        materials[counter] = new dielectric(1.5);
                        objects[counter] = new sphere(center, 0.2, materials[counter]);
                    }

                    counter++;
                }
            }
        }

        // Allocate materials and objects
        *mat_lst = new material_list(materials, counter); //"Owner" list
        *world = new hittable_list(objects, counter);
    }
}

__global__ void destroy_world(hittable** world, material_list** mat_lst) {   //}, hittable** objects, int num_objects) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete *world;
        delete *mat_lst;
    }
}

void render(int pixels_per_block_x, 
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
    render_kernel<<<blocks, threads>>>(
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
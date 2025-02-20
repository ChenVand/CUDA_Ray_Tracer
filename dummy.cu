
#include <stdio.h>
#include <time.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/version.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/random.h>


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


class managed {
  public:
    // Override operator new
    void *operator new(size_t size) {
        void *ptr;
        cudaMallocManaged(&ptr, size);
        cudaDeviceSynchronize();
        cudaCheckErrors("cudaMallocManaged in new operator failed!");
        return ptr;
    }

    // Override operator delete
    void operator delete(void* ptr) {
        cudaFree(ptr);
    }
};

class int_class : public managed {
// class int_class {
public:
    int number;

    __host__ __device__
    int_class(int number) : number(number) {}

    __host__ __device__ 
    int value() const {
        return number;
    }
};

class my_comparator {
  public:

    __device__
    bool operator()(const int_class* a, const int_class* b) const {
        return (a->value() <= b->value()); // Use offset in comparison
    }

};

__host__ void sort_objects_recursive_ley(
            thrust::device_ptr<int_class*>& dev_ptr, 
            // thrust::device_vector<int_class*> d_vec,
            size_t start, 
            size_t end
    ) {

        size_t object_span = end - start;

        if (object_span >= 2)
        {
            // thrust::stable_sort(thrust::device, d_vec.begin(), d_vec.end(), my_comparator());
            // cudaDeviceSynchronize();
            // cudaCheckErrors("thrust stable_sort failure in bvh_world::sort_objects_recursive");
            thrust::stable_sort(thrust::device, dev_ptr + start,dev_ptr + end, my_comparator());

            auto mid = start + object_span/2;
            sort_objects_recursive_ley(dev_ptr, start, mid);
            sort_objects_recursive_ley(dev_ptr, mid, end);
        }
    }

// __global__ void assign(int_class** array_of_pointers) {
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
//         // int_class* int_pointer = {};
//         for (int i=0; i<64; i++) {
            
//             array_of_pointers[i] = &int_class(64 - i);
//         }
//     }
// }

int main () {
    
    int_class** array_of_pointers;
    cudaMallocManaged((void **)&array_of_pointers, 64*sizeof(int_class*));

    for (int i=0; i<64; i++) {
        array_of_pointers[i] = new int_class(64 - i);
    }
    // assign<<<1,1>>>(array_of_pointers);
    // cudaDeviceSynchronize();

    // printf("hot gere 1\n");

    thrust::device_ptr<int_class*> dev_ptr(array_of_pointers);

    // sort_objects_recursive_ley(dev_ptr, 0, 64);
    
    // thrust::device_vector<int_class*> d_vec(dev_ptr, dev_ptr + 64);

    printf("hot gere\n");

    sort_objects_recursive_ley(dev_ptr, 0, 64); 

    for (int i=0; i<64; i++) {
        std::printf("%d\n", array_of_pointers[i]->value());
    }
}


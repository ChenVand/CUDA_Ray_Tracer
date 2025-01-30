
#include <stdio.h>
#include <time.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/version.h>

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
public:
    int number;

    int_class(int number) : number(number) {}

    int value() const {
        return number;
    }
};

class my_comparator {
  public:
    int axis;

    // __host__ __device__
    // bbox_comparator(int axis) : axis(axis) {}

    // __host__ __device__
    // static bool box_compare(
    //     const hittable*& a, const hittable*& b, int axis_index
    // ) {
    //     auto a_axis_interval = a->bounding_box().axis_interval(axis_index);
    //     auto b_axis_interval = b->bounding_box().axis_interval(axis_index);
    //     return a_axis_interval.min <= b_axis_interval.min;
    // }

    __device__
    bool operator()(const int_class* a, const int_class* b) const {
        return (a->value() <= b->value()); // Use offset in comparison
    }

};

__host__ void sort_objects_recursive_ley(
            thrust::device_ptr<int_class*>& dev_ptr, 
            size_t start, 
            size_t end
    ) {

        size_t object_span = end - start;

        if (object_span >= 2)
        {
            thrust::stable_sort(dev_ptr + start, dev_ptr + end, my_comparator());
            // cudaDeviceSynchronize();
            cudaCheckErrors("thrust stable_sort failure in bvh_world::sort_objects_recursive");

            auto mid = start + object_span/2;
            sort_objects_recursive_ley(dev_ptr, start, mid);
            sort_objects_recursive_ley(dev_ptr, mid, end);
        }
    }

int main () {
    
    int_class** array_of_pointers;
    cudaMallocManaged((void **)&array_of_pointers, 64*sizeof(int_class*));

    for (int i=0; i<64; i++) {
        array_of_pointers[i] = new int_class(64 - i);
    }

    thrust::device_ptr<int_class*> dev_ptr(array_of_pointers);

    sort_objects_recursive_ley(dev_ptr, 0, 64);
    
    for (int i=0; i<64; i++) {
        std::printf("%d\n", array_of_pointers[i]->value());
    }
}


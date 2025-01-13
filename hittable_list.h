#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"

class hittable_list : public hittable {
  public:

    hittable** objects;
    int size;
    int capacity;

    // void *operator new(size_t size) {
    //     void *ptr;
    //     cudaMallocManaged(&ptr, size);
    //     cudaDeviceSynchronize();
    //     cudaCheckErrors("cudaMallocManaged for new hittable_list failed!");
    //     return ptr;
    // }
    __host__ hittable_list(int initial_capacity = 16) : objects(nullptr), size(0), capacity(initial_capacity) {
        cudaError_t err = cudaMallocManaged(&objects, sizeof(hittable*) * capacity);
        if (err != cudaSuccess) {
            printf("Failed to allocate managed memory for objects: %s\n", cudaGetErrorString(err));
        }
    }
    __host__ hittable_list(hittable* object, int initial_capacity = 16) : objects(nullptr), size(0), capacity(initial_capacity) { 
        cudaError_t err = cudaMallocManaged(&objects, sizeof(hittable*) * capacity);
        if (err != cudaSuccess) {
            printf("Failed to allocate managed memory for objects: %s\n", cudaGetErrorString(err));
        }
        add(object); 
    }

    __host__ ~hittable_list() {
        clear();
    }

    __host__ void clear() {
        for (int i = 0; i < size; ++i) {
            if (objects[i] != nullptr) {
                delete objects[i];
                objects[i] = nullptr;
            }
        }
        if (objects != nullptr) {
            cudaFree(objects);
            objects = nullptr;
        }
        size = 0;
        capacity = 0;
    }

     __host__ void add(hittable* object) {
        if (size == capacity) {
            // Increase capacity
            int new_capacity = capacity * 2;
            hittable** new_objects;
            cudaError_t err = cudaMallocManaged(&new_objects, sizeof(hittable*) * new_capacity);
            if (err != cudaSuccess) {
                printf("Failed to allocate managed memory for new_objects: %s\n", cudaGetErrorString(err));
                return;
            }
            // Copy existing data to new memory
            for (int i = 0; i < size; ++i) {
                new_objects[i] = objects[i];
            }
            clear(); // delete old memory
            capacity = new_capacity;
        }
        objects[size++] = object;
    }

    __device__ bool hit(const ray& r, float ray_tmin, float ray_tmax, hit_record* rec) const override {
        //debug
        printf("reached hit function for hittable_list\n");
        hit_record* temp_rec = new hit_record;
        bool hit_anything = false;
        auto closest_so_far = ray_tmax;

        for (int i = 0; i < size; ++i) {
            if (objects[i]->hit(r, ray_tmin, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec->t;
                *rec = *temp_rec;
            }
        }

        return hit_anything;
    }
};

#endif // HITTABLE_LIST_H
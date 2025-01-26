#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

// #include "aabb.h"
#include "hittable.h"

class hittable_list : public hittable {
  public:

    hittable** objects;
    int size;

    __host__ __device__ hittable_list(hittable** object_list, int length) : objects(object_list), size(length) 
    {
        //This will probably have to change, at least to a reduce algo
        aabb box;
        if (length == 0) {
            box = aabb();
        } else {
            box = object_list[0]->bounding_box();
            for (int i = 1; i < length; ++i) {
                box = aabb(box, object_list[i]->bounding_box());
            }
        }
        bbox = box;
    }

    __host__ ~hittable_list() override{
        for (int i = 0; i < size; ++i) {
            delete objects[i]; // Delete each hittable object
        }
        delete[] objects; // Delete the array of pointers
    }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override;

    __host__ __device__ aabb bounding_box() const override { return bbox; }

  private:
    aabb bbox;
};

__device__ bool hittable_list::hit(const ray& r, interval ray_t, hit_record& rec) const {
        // hit_record* temp_rec = new hit_record;
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;

        for (int i = 0; i < size; ++i) {
            if (objects[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

#endif // HITTABLE_LIST_H


// -----------------------------------------------------------------------------
// __host__ int available_capacity() {
    //     return capacity - size;
    // }

    // __host__ void add_capacity(int delta_capacity) {
    //     int num_objects = size;
    //     int new_capacity = capacity + delta_capacity;
    //     hittable** new_object_list;
    //     cudaError_t err = cudaMalloc(&new_object_list, sizeof(hittable*) * new_capacity);
    //     if (err != cudaSuccess) {
    //         printf("Failed to allocate managed memory for new_object_list: %s\n", cudaGetErrorString(err));
    //         return;
    //     }
    //     // Copy existing data to new memory

    //     cudaMemcpy(new_object_list, objects, sizeof(hittable*) * num_objects, cudaMemcpyDefault);
    //     cudaCheckErrors("cuda memcpy failed in add_capacity");
    //     clear(); // delete old memory
    //     objects = new_object_list;
    //     capacity = new_capacity;
    //     size = num_objects;
    // }

    // __device__ void add_objects(hittable* new_objects, size_t num_objects) {
    //     if (num_objects + size > capacity)  {
    //         printf("Error: num_objects > capacity\n");
    //         return;
    //     }
    //     for (int i = 0; i < num_objects; ++i) {
    //         objects[size++] = &new_objects[i];
    //     }
    // }

    //ADD INCREASE CAPACITY FUNCTION

    // __host__ hittable_list(int initial_capacity = 16) : objects(nullptr), size(0), capacity(initial_capacity) {
    //     cudaError_t err = cudaMalloc(&objects, sizeof(hittable*) * capacity);
    //     if (err != cudaSuccess) {
    //         printf("Failed to allocate managed memory for objects: %s\n", cudaGetErrorString(err));
    //     }
    // }
    // __host__ hittable_list(hittable* object, int initial_capacity = 16) : objects(nullptr), size(0), capacity(initial_capacity) { 
    //     cudaError_t err = cudaMalloc(&objects, sizeof(hittable*) * capacity);
    //     if (err != cudaSuccess) {
    //         printf("Failed to allocate managed memory for objects: %s\n", cudaGetErrorString(err));
    //     }
    //     add(object); 
    // }

    // __host__ void clear() {
    //     // for (int i = 0; i < size; ++i) {
    //     //     if (objects[i] != nullptr) {
    //     //         cudaFree(objects[i]);
    //     //         objects[i] = nullptr;
    //     //     }
    //     // }
    //     if (objects != nullptr) {
    //         cudaFree(objects);
    //         objects = nullptr;
    //     }
    //     // size = 0;
    //     // capacity = 0;
    // }

    //  __host__ __device__ void add(hittable* object) {
    //     if (size == capacity) {
    //         // Increase capacity
    //         int new_capacity = capacity * 2;
    //         hittable** new_objects;
    //         cudaError_t err = cudaMalloc(&new_objects, sizeof(hittable*) * new_capacity);
    //         if (err != cudaSuccess) {
    //             printf("Failed to allocate managed memory for new_objects: %s\n", cudaGetErrorString(err));
    //             return;
    //         }
    //         // Copy existing data to new memory

    //         cudaMemcpy(new_objects, objects, sizeof(hittable*) * size, cudaMemcpyDefault);
    //         clear(); // delete old memory
    //         objects = new_objects;
    //         capacity = new_capacity;
    //     }
    //     objects[size++] = object;
    // }
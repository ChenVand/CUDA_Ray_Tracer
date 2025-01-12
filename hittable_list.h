#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"

class hittable_list : public hittable {
  public:

    hittable** objects;
    int size;
    int capacity;

    __device__ hittable_list() : objects(nullptr), size(0), capacity(0) {}
    __device__ hittable_list(hittable object) { add(&object); }

     __device__ void clear() {
        for (int i = 0; i < size; ++i) {
            delete objects[i];
        }
        delete[] objects;
        objects = nullptr;
        size = 0;
        capacity = 0;
    }

    __device__ void add(hittable* object) {
        if (size == capacity) {
            capacity = (capacity == 0) ? 1 : capacity * 2;
            hittable** new_objects = new hittable*[capacity];
            for (int i = 0; i < size; ++i) {
                new_objects[i] = objects[i];
            }
            delete[] objects;
            objects = new_objects;
        }
        objects[size++] = object;
    }

    __device__ bool hit(const ray& r, float ray_tmin, float ray_tmax, hit_record& rec) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_tmax;

        for (int i = 0; i < size; ++i) {
            if (objects[i]->hit(r, ray_tmin, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }
};

#endif
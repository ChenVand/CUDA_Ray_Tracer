#ifndef HITTABLE_H
#define HITTABLE_H

#include "aabb.h"

class material;

class hit_record {
  public:
    point3 p;
    vec3 normal;
    material* mat_ptr;
    float t;
    bool front_face;

    __device__ void set_face_normal(const ray& r, const vec3& outward_normal) {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.
  
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

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

class hittable: public managed {
  public:

    __host__ __device__ virtual ~hittable() {}

    __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;

    __host__ __device__ virtual aabb bounding_box() const = 0;
};

#endif
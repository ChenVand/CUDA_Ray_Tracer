#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"

class sphere : public hittable {
  public:

    // Stationary Sphere
    __host__ __device__ sphere(const point3& static_center, float radius, material* new_mat) 
        : center(static_center, vec3(0,0,0)), radius(fmaxf(0,radius)), mat_ptr(new_mat) {
            auto rvec = vec3(radius, radius, radius);
            bbox = aabb(static_center - rvec, static_center + rvec);
        }

    // Moving Sphere
    __host__ __device__ sphere(const point3& center1, const point3& center2, float radius, 
            material* new_mat) 
        : center(center1, center2 - center1), radius(fmaxf(0,radius)), mat_ptr(new_mat)
    {
        auto rvec = vec3(radius, radius, radius);
        aabb box1(center.at(0) - rvec, center.at(0) + rvec);
        aabb box2(center.at(1) - rvec, center.at(1) + rvec);
        bbox = aabb(box1, box2);
    }


    __host__ __device__ sphere(const sphere& other) 
        : center(other.center), radius(other.radius), mat_ptr(other.mat_ptr) {}
    
    __host__ ~sphere() override { mat_ptr = nullptr; }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override;

    __host__ __device__ aabb bounding_box() const override { return bbox; }

  // private:
    ray center;
    float radius;
    material* mat_ptr;
    aabb bbox;
};

__device__ bool sphere::hit(const ray& r, interval ray_t, hit_record& rec) const {

        point3 current_center = center.at(r.time());
        vec3 oc = current_center - r.origin();
        auto a = r.direction().length_squared();
        auto h = dot(r.direction(), oc);
        auto c = oc.length_squared() - radius*radius;

        auto discriminant = h*h - a*c;
        if (discriminant < 0)
            return false;

        auto sqrtd = sqrtf(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root))
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - current_center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat_ptr = mat_ptr;

        return true;
    }

#endif
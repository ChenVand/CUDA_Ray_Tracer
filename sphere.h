#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"

class sphere : public hittable {
  public:

    __host__ __device__ sphere() : center(vec3()), radius(0) {}
    __host__ __device__ sphere(const point3& center, float radius) : center(center), radius(fmaxf(0,radius)) {}
    __host__ __device__ sphere(const sphere& other) : center(other.center), radius(other.radius) {}
    __device__ bool hit(const ray& r, float ray_tmin, float ray_tmax, hit_record& rec) const override;

  // private:
    point3 center;
    float radius;
};

__device__ bool sphere::hit(const ray& r, float ray_tmin, float ray_tmax, hit_record& rec) const {

        vec3 oc = center - r.origin();
        auto a = r.direction().length_squared();
        auto h = dot(r.direction(), oc);
        auto c = oc.length_squared() - radius*radius;

        auto discriminant = h*h - a*c;
        if (discriminant < 0)
            return false;

        auto sqrtd = sqrtf(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (h - sqrtd) / a;
        if (root <= ray_tmin || ray_tmax <= root) {
            root = (h + sqrtd) / a;
            if (root <= ray_tmin || ray_tmax <= root)
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);

        return true;
    }

#endif
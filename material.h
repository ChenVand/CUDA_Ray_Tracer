#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.h"

// class material : public managed{
class material {
  public:
    virtual ~material() = default;

    __device__ virtual bool scatter(
        curandState& rand_state, 
        const ray& r_in, 
        const hit_record& rec, 
        color& attenuation, 
        ray& scattered
    ) const {
        return false;
    }
};

class lambertian : public material {
  public:
    __host__ __device__ lambertian(const color& albedo) : albedo(albedo) {}

    __device__ bool scatter(curandState& rand_state, const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
    const override {
        auto scatter_direction = rec.normal + random_unit_vector(rand_state); //Lambertian diffuse
        // auto scatter_direction = random_on_hemisphere(rand_state, rec.normal); //Basic diffuse
        
        // Catch degenerate scatter direction
        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction, r_in.time());
        attenuation = albedo;
        return true;
    }

  private:
    color albedo;
};

class metal : public material {
  public:
    __host__ __device__ metal(const color& albedo, float fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}

    __device__ bool scatter(curandState& rand_state, const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
    const override {
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        reflected = unit_vector(reflected) + (fuzz * random_unit_vector(rand_state));
        scattered = ray(rec.p, reflected, r_in.time());
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }

  private:
    color albedo;
    float fuzz;
};

class dielectric : public material {
  public:
    __host__ __device__ dielectric(float refraction_index) : refraction_index(refraction_index) {}

    __device__ bool scatter(curandState& rand_state, const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
    const override {
        attenuation = color(1.0, 1.0, 1.0);
        float ri = rec.front_face ? (1.0/refraction_index) : refraction_index;

        vec3 unit_direction = unit_vector(r_in.direction());
        float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0);
        float sin_theta = sqrtf(1.0 - cos_theta*cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, ri) > random_float(rand_state))
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, ri);

        scattered = ray(rec.p, direction, r_in.time());
        return true;
    }

  private:
    // Refractive index in vacuum or air, or the ratio of the material's refractive index over
    // the refractive index of the enclosing media
    float refraction_index;

    __host__ __device__ static float reflectance(float cosine, float refraction_index) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0*r0;
        return r0 + (1-r0)*powf((1 - cosine),5);
    }
};

class material_list {
  public:

    material** materials;
    int size;

    __host__ __device__ material_list(material** material_list_list, int length) : materials(material_list_list), size(length) {}

    __host__ __device__ ~material_list() {
        for (int i = 0; i < size; ++i) {
            delete materials[i]; // Delete each hittable object
        }
        delete[] materials; // Delete the array of pointers
    }

};

#endif
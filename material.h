#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.h"

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

        scattered = ray(rec.p, scatter_direction);
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
        scattered = ray(rec.p, reflected);
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }

  private:
    color albedo;
    float fuzz;
};

class dielectric : public material {
  public:
    __host__ __device__ dielectric(double refraction_index) : refraction_index(refraction_index) {}

    __device__ bool scatter(curandState& rand_state, const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
    const override {
        attenuation = color(1.0, 1.0, 1.0);
        double ri = rec.front_face ? (1.0/refraction_index) : refraction_index;

        vec3 unit_direction = unit_vector(r_in.direction());
        vec3 refracted = refract(unit_direction, rec.normal, ri);

        scattered = ray(rec.p, refracted);
        return true;
    }

  private:
    // Refractive index in vacuum or air, or the ratio of the material's refractive index over
    // the refractive index of the enclosing media
    double refraction_index;
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
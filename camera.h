// cspell: disable
#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"
#include "material.h"
// #include "render_with_cuda.h"

class camera {
  public:
    float  aspect_ratio      = 1.0;  // Ratio of image width over height
    int    image_width       = 100;  // Rendered image width in pixel count
    int    samples_per_pixel = 32;   // Count of random samples for each pixel
    // int    max_depth         = 10;   // Maximum number of ray bounces into scen
    
    float vfov = 90;                   // Vertical view angle (field of view)
    point3 lookfrom = point3(0,0,0);   // Point camera is looking from
    point3 lookat   = point3(0,0,-1);  // Point camera is looking at
    vec3   vup      = vec3(0,1,0);     // Camera-relative "up" direction

    float defocus_angle = 0;  // Variation angle of rays through each pixel
    float focus_dist = 10;    // Distance from camera lookfrom point to plane of perfect focus

    void initialize();

    __host__ __device__ int get_image_height() const { return image_height; }

    __host__ __device__ float get_pixel_samples_scale() const { return pixel_samples_scale; }

    __host__ __device__ point3 get_center() const { return center; }

    __host__ __device__ point3 get_pixel00_loc() const { return pixel00_loc; }

    __host__ __device__ point3 get_pixel_delta_u() const { return pixel_delta_u; }

    __host__ __device__ point3 get_pixel_delta_v() const { return pixel_delta_v; }

    __host__ __device__ point3 get_defocus_disk_u() const { return defocus_disk_u; }

    __host__ __device__ point3 get_defocus_disk_v() const { return defocus_disk_v; }

    // void render(int threads_x, int threads_y, hittable** world, float& timer_seconds) ;

    void display_frame(vec3* frame_buffer) {
    
        std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
        for (int j = 0; j < image_height; j++) {
            for (int i = 0; i < image_width; i++) {
                size_t pixel_index = j*image_width + i;
                auto pixel_color = frame_buffer[pixel_index];

                write_color(std::cout, pixel_color);
            }
        }
        cudaDeviceSynchronize();
        cudaCheckErrors("post-display device synchronization failed");
    }

  private:
    int    image_height;         // Rendered image height
    float  pixel_samples_scale;  // Color scale factor for a sum of pixel samples
    point3 center;               // Camera center
    point3 pixel00_loc;          // Location of pixel 0, 0
    vec3   pixel_delta_u;        // Offset to pixel to the right
    vec3   pixel_delta_v;        // Offset to pixel below
    vec3   u, v, w;              // Camera frame basis vectors
    vec3   defocus_disk_u;       // Defocus disk horizontal radius
    vec3   defocus_disk_v;       // Defocus disk vertical radius

};

void camera::initialize() {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        pixel_samples_scale = 1.0 / samples_per_pixel;

        center = lookfrom;

        // Determine viewport dimensions.
        auto theta = degrees_to_radians(vfov);
        auto h = std::tan(theta/2);
        auto viewport_height = 2 * h * focus_dist; // (focal length is set equal to focus_dist)
        auto viewport_width = viewport_height * (double(image_width)/image_height);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        auto viewport_u = viewport_width * u;     // Vector across viewport horizontal edge
        auto viewport_v =  viewport_height * -v;  // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = center - (focus_dist * w) - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    
        // Calculate the camera defocus disk basis vectors.
        auto defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }

#endif
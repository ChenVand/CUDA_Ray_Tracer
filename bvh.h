#ifndef BVH_H
#define BVH_H

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/random.h>

#include "aabb.h"
#include "hittable.h"
#include "hittable_list.h"

/*BVH creation is done in a reduce fashion with a cuda kernel, and hit is done iteratively.
Done with help from:
https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/
*/

// A handle for hittables, to have a lighter data structure to sort with thrust::stable_sort
class bvh_primitive {
public:
    size_t object_index;
    aabb bbox;

    __host__ __device__ bvh_primitive() {}

    __host__ __device__ bvh_primitive(size_t object_index, aabb bbox) : object_index(object_index), bbox(bbox) {}
};

class primitive_comparator {
  public:
    int axis;

    __host__ __device__
    primitive_comparator(int axis) : axis(axis) {}

    __device__
    bool operator()(
        bvh_primitive a, bvh_primitive b
    ) {
        auto a_axis_interval = a.bbox.axis_interval(axis);
        auto b_axis_interval = b.bbox.axis_interval(axis);
        return a_axis_interval.min <= b_axis_interval.min;
    }

};

class bvh_node {
  public:
    aabb bbox;
    int loc_for_leaf = -1;
    int left_child_loc;
    int right_child_loc;

    __host__ __device__ bvh_node() {}

    __host__ __device__ bvh_node(aabb bbox) : bbox(bbox) {}

    __host__ __device__ bvh_node(aabb bbox, int loc_for_leaf) : bbox(bbox), loc_for_leaf(loc_for_leaf) {}

    __host__ __device__ bvh_node(aabb bbox, int left_child_loc, int right_child_loc) : 
            bbox(bbox), left_child_loc(left_child_loc), right_child_loc(right_child_loc) {}
};

__global__ 
void create_bvh(
        int num_objects,
        bvh_primitive* primitives,
        int tree_depth,
        bvh_node* d_nodes) {

    /*BVH tree is created from bottom to top, filling d_nodes so that the root is the first element*/

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int num_leaves = pow(2, tree_depth);
    if (idx >= num_leaves) {return;}

    int curr_offset = num_leaves - 1;

    if (idx < num_objects) {
        d_nodes[curr_offset+idx] = bvh_node(primitives[idx].bbox, primitives[idx].object_index);
    } else {
        d_nodes[curr_offset+idx] = bvh_node(empty_aabb());
    }
    __syncthreads();
    int prev_offset = curr_offset;
    int left_child, right_child;
    for (int i=num_leaves/2; i>0; i/=2) {
        if (idx>=i) {return;}

        curr_offset -= i;
        left_child = prev_offset+2*idx;
        right_child = prev_offset+2*idx+1;

        d_nodes[curr_offset+idx] = bvh_node(
            aabb(d_nodes[left_child].bbox, d_nodes[right_child].bbox), left_child, right_child);
        
        prev_offset = curr_offset;
        __syncthreads();
    }
}  

// class bvh_world: public hittable, public managed {
class bvh_world: public hittable {
  public:
    int num_objects;
    hittable** m_objects;
    int tree_depth;
    bvh_node* d_nodes;

    __host__ bvh_world(hittable** object_list, int size) {

        num_objects = size;
        m_objects = object_list;
        tree_depth = ceil(log2(num_objects));

        // cudaMemPrefetchAsync(d_objects, num_objects * sizeof(hittable*), 0, 0); //This is new ########################

        int num_nodes = pow(2,tree_depth + 1) - 1;
        cudaMalloc((void **)&d_nodes, num_nodes * sizeof(bvh_node));

        thrust::default_random_engine random_engine(17);
        rng = random_engine;
        thrust::uniform_int_distribution<int> distribution(0, 2);
        dist = distribution;

        thrust::device_vector<bvh_primitive> primitives(num_objects);
        for (size_t i = 0; i < num_objects; ++i) {
            primitives[i] = bvh_primitive(i, m_objects[i]->bounding_box());
        }

        //Debug
        printf("got here 3\n");

        sort_primitives_recursive(primitives, 0, num_objects);

        //Debug
        printf("got here 4\n");

        // create BVH
        int blocks = (pow(2, tree_depth)+31)/32;
        int threads = 32;
        create_bvh<<<blocks, threads>>>(num_objects, raw_pointer_cast(primitives.data()), 
                tree_depth, d_nodes);
        cudaCheckErrors("create_bvh kernel launch failed");
        cudaDeviceSynchronize();
        cudaCheckErrors("post create_bvh kernel sync failed");

        // Move objects to device memory
        cudaMemPrefetchAsync(m_objects, num_objects, 0);
        cudaCheckErrors("m_objects prefetch to GPU failed");
        cudaDeviceSynchronize();
        cudaCheckErrors("post m_objects prefetch sync failed");
    }

    __host__ ~bvh_world() { cudaFree(d_nodes); }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        /*Done iteratively instead of recursively, using a stack*/

        hit_record temp_rec;
        bool hit_anything = false;
        auto current_interval = ray_t;

        bvh_node* stack[32];
        bvh_node** stackPtr = stack;
        *stackPtr++ = NULL; // push

        // Traverse nodes starting from the root.
        bvh_node* node = &d_nodes[0]; // Initialize at root, which is surely hit
        bool left_hit, right_hit;
        bvh_node* childL = nullptr;
        bvh_node* childR = nullptr;
        do {
            if (node->loc_for_leaf > -1) {
                // node is leaf, hit actual object
                if (m_objects[node->loc_for_leaf]->hit(r, current_interval, temp_rec)) {
                    hit_anything = true;
                    current_interval.max = temp_rec.t;
                    rec = temp_rec;
                }
            } else {
                // node is internal
                childL = &d_nodes[node->left_child_loc];
                childR = &d_nodes[node->right_child_loc];
                left_hit = childL->bbox.hit(r, current_interval);
                right_hit = childR->bbox.hit(r, current_interval);
                if (left_hit) {
                    node = childL;
                    if (right_hit) {
                        *stackPtr++ = childR; //stack
                    } 
                } else {
                    if (right_hit) {
                        node = childR;
                    } else {
                        node = *--stackPtr; //pop
                    }
                }
            }
        } while (node != NULL);

        return hit_anything;
    } 

    __host__ __device__ aabb bounding_box() const override {return universe_aabb();}


  private:
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist;

    __host__ void sort_primitives_recursive(
            thrust::device_vector<bvh_primitive>& primitives, 
            int start, 
            int end
    ) {

        int axis = dist(rng);

        int object_span = end - start;

        if (object_span >= 2)
        {
            // //Debug
            // printf("got to start: %d, end: %d\n", int(start), int(end));
            // thrust::stable_sort(thrust::device, dev_ptr + start, dev_ptr + end, bbox_comparator(axis));
            thrust::stable_sort(thrust::device, primitives.begin() + start, primitives.begin() + end, primitive_comparator(axis));
            // cudaDeviceSynchronize();
            // cudaCheckErrors("thrust stable_sort failure in bvh_world::sort_objects_recursive");

            auto mid = start + object_span/2;
            sort_primitives_recursive(primitives, start, mid);
            sort_primitives_recursive(primitives, mid, end);
        }
    }

}; 

#endif
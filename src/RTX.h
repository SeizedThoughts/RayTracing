#pragma once

#include <fstream>

// #include "Vertex.h"

#include <cuda_runtime.h>

namespace RTX{
    struct Vertex{
        float x;
        float y;
        float z;

        __host__ __device__ int operator == (const Vertex &vert) const;

        __host__ __device__ int operator != (const Vertex &vert) const;

        __host__ __device__ Vertex& operator = (const Vertex &vert);

        __host__ __device__ Vertex& operator = (const float array[3]);

        __host__ __device__ Vertex operator + (const Vertex &vert) const;

        __host__ __device__ Vertex& operator += (const Vertex &vert);

        __host__ __device__ Vertex operator - (const Vertex &vert) const;

        __host__ __device__ Vertex& operator -= (const Vertex &vert);

        __host__ __device__ float operator * (const Vertex &vert) const;

        __host__ __device__ Vertex operator * (const float scalar) const;

        __host__ __device__ Vertex& operator *= (const float scalar);

        __host__ __device__ Vertex& normalize();

        __host__ __device__ Vertex cross(const Vertex &vert) const;
    };

    struct Quaternion{
        float w;
        float x;
        float y;
        float z;

        __host__ __device__ int operator == (const Quaternion &quat) const;

        __host__ __device__ int operator != (const Quaternion &quat) const;

        __host__ __device__ Quaternion& operator = (const Quaternion &quat);

        __host__ __device__ Quaternion& operator = (const float array[3]);

        __host__ __device__ Quaternion operator * (const Quaternion &quat) const;

        __host__ __device__ Quaternion& operator *= (const Quaternion &quat);

        __host__ __device__ Quaternion operator * (const Vertex &vert) const;

        __host__ __device__ Quaternion& operator *= (const Vertex &vert);

        __host__ __device__ Quaternion& normalize();

        __host__ __device__ Vertex rotate(const Vertex &vert)const;

        __host__ __device__ Vertex& rotateInPlace(Vertex &vert)const;
    };

    struct Face{
        int v1;
        int v2;
        int v3;

        int vt1;
        int vt2;
        int vt3;

        int vn;

        __host__ __device__ Face& operator = (const int array[3]);
    };

    struct Texture{
        unsigned int width;
        unsigned int height;

        uchar4 *pixel_buffer;

        Texture(const unsigned int i_width, const unsigned int i_height, const unsigned char *img, const unsigned int channels);

        Texture(const unsigned int i_width, const unsigned int i_height, const unsigned char *img);

        __host__ __device__ Texture& operator = (const Texture &texture);
    };

    struct Transform{
        Vertex position;
        Quaternion rotation;
        Vertex scale;

        __host__ __device__ Vertex transform(const Vertex &vert) const;

        __host__ __device__ Vertex& transformInPlace(Vertex &vert) const;

        __host__ __device__ Transform& operator = (const Transform &transform);
    };
    
    struct Camera{
        Vertex position;
        Quaternion rotation;
        float x_fov;
        float y_fov;

        __host__ __device__ Vertex getPoint();

        __host__ __device__ Vertex getPoint(const int width, const int height, const int x, const int y);
    };

    struct Model{
        unsigned int vertex_start;
        unsigned int normal_start;
        unsigned int texture_vertex_start;
        unsigned int face_start;

        unsigned int face_count;
    };

    struct Renderer{
        unsigned int model;
        unsigned int texture;
        unsigned int transform;
    };

    struct Buffers{
        unsigned int vertex_count;
        unsigned int normal_count;
        unsigned int texture_vertex_count;
        unsigned int face_count;
        unsigned int transform_count;
        unsigned int model_count;
        unsigned int texture_count;
        unsigned int renderer_count;
        unsigned int camera_count;

        unsigned int max_vertex_count;
        unsigned int max_normal_count;
        unsigned int max_texture_vertex_count;
        unsigned int max_face_count;
        unsigned int max_transform_count;
        unsigned int max_model_count;
        unsigned int max_texture_count;
        unsigned int max_renderer_count;
        unsigned int max_camera_count;
        
        Vertex *vertex_buffer;
        Vertex *normal_buffer;
        Vertex *texture_vertex_buffer;
        Face *face_buffer;
        Transform *transform_buffer;
        Model *model_buffer;
        Texture *texture_buffer;
        Renderer *renderer_buffer;
        Camera *camera_buffer;
    };

    struct PrecomputedFace{
        unsigned int v1;
        unsigned int v2;
        unsigned int v3;

        Vertex vu;
        Vertex vv;
        //vw = vn
        
        Vertex vvxvw;
        Vertex vwxvu;
        Vertex vuxvv;

        Vertex vtu;
        Vertex vtv;
        //vtw = (0, 0, 1)

        Vertex vn;
    };

    struct PrecomputedBuffers{
        Vertex *transformed_vertex_buffer;
        Vertex *transformed_normal_buffer;
        PrecomputedFace *precomputed_face_buffer;

        int *face_in_region;

        int *renderer_update_flags;
        int *model_update_flags;

        Buffers data;
    };

    extern Buffers buffers;
    extern Buffers *d_buffers;
    
    void makeBuffers(
        unsigned int max_vertex_count,
        unsigned int max_normal_count,
        unsigned int max_texture_vertex_count,
        unsigned int max_face_count,
        unsigned int max_transform_count, //TODO: is this used for anything other than renderers?
        unsigned int max_model_count,
        unsigned int max_texture_count,
        unsigned int max_renderer_count,
        unsigned int max_camera_count
    );

    //model, texture map, material map, maybe normal map
    int load(const char *file_path);

    int createCamera();

    void updateCamera(unsigned int camera);

    int push(Transform transform);

    int createRenderer(unsigned int model, unsigned int texture, unsigned int transform);
};

std::ostream& operator << (std::ostream &os, const RTX::Vertex &vert);

std::ostream& operator << (std::ostream &os, const RTX::Quaternion &quat);

std::ostream& operator << (std::ostream &os, const RTX::Face &face);

std::ostream& operator << (std::ostream &os, const RTX::Texture &texture);
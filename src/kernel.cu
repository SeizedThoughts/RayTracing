#include "kernel.h"
#include <math.h>

#include "RTX.h"

using namespace RTX;

__device__ inline void make_transformation(const Vertex &vt1, const Vertex &vt2, const Vertex &vt3, const Vertex &vt4, const Vertex &v1, const Vertex &v2, const Vertex &v3, const Vertex &v4, float *out){
    /*
        https://stackoverflow.com/questions/3780493/map-points-between-two-triangles-in-3d-space

        A = [
            v1.x v1.y v1.z 1
            v2.x v2.y v2.z 1
            v3.x v3.y v3.z 1
            v4.x v4.y v4.z 1
        ]

        https://semath.info/src/inverse-cofactor-ex4.html

        A^-1 = A~ / |A|

        A~_ij = (-1) ^ (i + j) * |M_ij|

        M_ij is the matrix multiplied by A_ij when determinining the determinant

        A = [
            vt1.x vt1.y vt1.z 1
            vt2.x vt2.y vt2.z 1
            vt3.x vt3.y vt3.z 1
            vt4.x vt4.y vt4.z 1
        ]

        T = A'A^-1
    */
}

__device__ inline void texture_map(const Vertex &vt1, const Vertex &vt2, const Vertex &vt3, const Vertex &v1, const Vertex &v2, const Vertex &v3, const Vertex &hit){


}

__device__ inline void printBuffers(Buffers buffers){
    printf("texture_size: %d\n", buffers.texture_size);

    printf("vertex_count: %d\n", buffers.vertex_count);
    printf("normal_count: %d\n", buffers.normal_count);
    printf("texture_vertex_count: %d\n", buffers.texture_vertex_count);
    printf("face_count: %d\n", buffers.face_count);
    printf("camera_count: %d\n", buffers.camera_count);
    printf("model_count: %d\n", buffers.model_count);
    printf("texture_count: %d\n", buffers.texture_count);
    printf("renderer_count: %d\n", buffers.renderer_count);

    printf("max_vertex_count: %d\n", buffers.max_vertex_count);
    printf("max_normal_count: %d\n", buffers.max_normal_count);
    printf("max_texture_vertex_count: %d\n", buffers.max_texture_vertex_count);
    printf("max_face_count: %d\n", buffers.max_face_count);
    printf("max_camera_count: %d\n", buffers.max_camera_count);
    printf("max_model_count: %d\n", buffers.max_model_count);
    printf("max_texture_count: %d\n", buffers.max_texture_count);
    printf("max_renderer_count: %d\n", buffers.max_renderer_count);

    unsigned int i;
    printf("vertex_buffer:\n");
    for(i = 0; i < buffers.vertex_count; i++){
        printf("    %d: (%f, %f, %f)\n", i, buffers.vertex_buffer[i].x, buffers.vertex_buffer[i].y, buffers.vertex_buffer[i].z);
    }

    printf("normal_buffer:\n");
    for(i = 0; i < buffers.normal_count; i++){
        printf("    %d: (%f, %f, %f)\n", i, buffers.normal_buffer[i].x, buffers.normal_buffer[i].y, buffers.normal_buffer[i].z);
    }

    printf("texture_vertex_buffer:\n");
    for(i = 0; i < buffers.texture_vertex_count; i++){
        printf("    %d: (%f, %f, %f)\n", i, buffers.texture_vertex_buffer[i].x, buffers.texture_vertex_buffer[i].y, buffers.texture_vertex_buffer[i].z);
    }

    printf("camera_buffer:\n");
    for(i = 0; i < buffers.camera_count; i++){
        printf("    %d: (position: (%f, %f, %f), rotation: (%f, %f, %f, %f), fov: (%f, %f))\n", i, buffers.camera_buffer[i].position.x, buffers.camera_buffer[i].position.y, buffers.camera_buffer[i].position.z, buffers.camera_buffer[i].rotation.w, buffers.camera_buffer[i].rotation.x, buffers.camera_buffer[i].rotation.y, buffers.camera_buffer[i].rotation.z, buffers.camera_buffer[i].x_fov, buffers.camera_buffer[i].y_fov);
    }

    printf("face_buffer:\n");
    for(i = 0; i < buffers.face_count; i++){
        printf("    %d: (v: (%d, %d, %d), vt: (%d, %d, %d), vn: %d)\n", i, buffers.face_buffer[i].v1, buffers.face_buffer[i].v2, buffers.face_buffer[i].v3, buffers.face_buffer[i].vt1, buffers.face_buffer[i].vt2, buffers.face_buffer[i].vt3, buffers.face_buffer[i].vn);
    }

    printf("model_buffer:\n");
    for(i = 0; i < buffers.model_count; i++){
        printf("    %d:\n        vertex_start: %d\n        normal_start: %d\n        texture_vertex_start: %d\n        face_start: %d\n        face_count: %d\n", i, buffers.model_buffer[i].vertex_start, buffers.model_buffer[i].normal_start, buffers.model_buffer[i].texture_vertex_start, buffers.model_buffer[i].face_start, buffers.model_buffer[i].face_count);
    }

    printf("renderer_buffer:\n");
    for(i = 0; i < buffers.renderer_count; i++){
        printf("    %d:\n        model: %d\n        texture: %d\n", i, buffers.renderer_buffer[i].model, buffers.renderer_buffer[i].texture);
    }
}

__device__ inline void draw(uchar4 *frame, const unsigned int idx, const unsigned int x, const unsigned int y, const unsigned int width, const unsigned int height, void *user_pointer){
    Buffers buffers = *((Buffers *)user_pointer);

    Camera camera = buffers.camera_buffer[0];
    Vertex point = camera.getPoint(width, height, x, y);
    
    unsigned int texture;
    unsigned int vertices[6];
    float closest = -1;
    for(unsigned int i = 0; i < buffers.renderer_count; i++){
        Renderer renderer = buffers.renderer_buffer[i];
        Model model = buffers.model_buffer[renderer.model];
        Vertex *model_vertices = &buffers.vertex_buffer[model.vertex_start];

        for(unsigned int j = 0; j < model.face_count; j++){
            Face face = buffers.face_buffer[model.face_start + j];
            Vertex vn = buffers.normal_buffer[model.normal_start + face.vn];

            /*
                (dist * point - (v1 - position)) * vn = 0
                dist * point * vn - (v1 - position) * vn = 0
                dist = (v1 - position) * vn / (point * vn)
            */
            float dist_den = dist_den = point * vn;

            //facing towards us
            if(dist_den < 0){
                Vertex v1 = model_vertices[face.v1];

                float dist = (v1 - camera.position) * vn / dist_den;
                if(dist > 0 && (dist < closest || closest == -1)){
                    Vertex hit = point * dist + camera.position;

                    Vertex v2 = model_vertices[face.v2];
                    Vertex v3 = model_vertices[face.v3];

                    Vertex bn12 = vn.cross(v1 - v2);

                    Vertex bn23 = vn.cross(v2 - v3);

                    Vertex bn31 = vn.cross(v3 - v1);


                    if((bn12 * (hit - v2)) > 0) continue;
                    if((bn23 * (hit - v3)) > 0) continue;
                    if((bn31 * (hit - v1)) > 0) continue;

                    closest = dist;

                    vertices[0] = model.vertex_start + face.v1;
                    vertices[1] = model.vertex_start + face.v2;
                    vertices[2] = model.vertex_start + face.v3;
                    vertices[3] = model.vertex_start + face.vt1;
                    vertices[4] = model.vertex_start + face.vt2;
                    vertices[5] = model.vertex_start + face.vt3;

                    texture = renderer.texture;
                    break;
                }
            }
        }
    }

    if(closest != -1){
        /*
            https://mathworld.wolfram.com/MatrixInverse.html
            B = [
                v1.x v2.x v3.x
                v1.y v2.y v3.y
                v1.z v2.z v3.z
            ]

            |B| = (
                  (v1.x * ((v2.y * v3.z) - (v3.y * v2.z))) - (v2.x * ((v1.y * v3.z) - (v3.y * v1.z))) + (v3.x * ((v1.y * v2.z) - (v2.y * v1.z)))
                - (v1.y * ((v2.x * v3.z) - (v3.x * v2.z))) + (v2.y * ((v1.x * v3.z) - (v3.x * v1.z))) - (v3.y * ((v1.x * v2.z) - (v2.x * v1.z)))
                + (v1.z * ((v2.x * v3.y) - (v3.x * v2.y))) - (v2.z * ((v1.x * v3.y) - (v3.x * v1.y))) + (v3.z * ((v1.x * v2.y) - (v2.x * v1.y)))
            )

            B^-1 = [
                ((v2.y * v3.z) - (v3.y * v2.z)) ((v3.x * v2.z) - (v2.x * v3.z)) ((v2.x * v3.y) - (v3.x * v2.y))
                ((v3.y * v1.z) - (v1.y * v3.z)) ((v1.x * v3.z) - (v3.x * v1.z)) ((v3.x * v1.y) - (v1.x * v3.y))
                ((v1.y * v2.z) - (v2.y * v1.z)) ((v2.x * v1.z) - (v1.x * v2.z)) ((v1.x * v2.y) - (v2.x * v1.y))
            ] / |B|

            B' = [
                vt1.x vt2.x vt3.x
                vt1.y vt2.y vt3.y
                vt1.z vt2.z vt3.z
            ]

            B'B^-1h = the point in space
        */

        Vertex v1 = buffers.vertex_buffer[vertices[0]];
        Vertex v2 = buffers.vertex_buffer[vertices[1]];
        Vertex v3 = buffers.vertex_buffer[vertices[2]];
        Vertex vt1 = buffers.texture_vertex_buffer[vertices[3]];
        Vertex vt2 = buffers.texture_vertex_buffer[vertices[4]];
        Vertex vt3 = buffers.texture_vertex_buffer[vertices[5]];

        Vertex hit = point * closest;

        float detB = (
              (v1.x * ((v2.y * v3.z) - (v3.y * v2.z))) - (v2.x * ((v1.y * v3.z) - (v3.y * v1.z))) + (v3.x * ((v1.y * v2.z) - (v2.y * v1.z)))
            - (v1.y * ((v2.x * v3.z) - (v3.x * v2.z))) + (v2.y * ((v1.x * v3.z) - (v3.x * v1.z))) - (v3.y * ((v1.x * v2.z) - (v2.x * v1.z)))
            + (v1.z * ((v2.x * v3.y) - (v3.x * v2.y))) - (v2.z * ((v1.x * v3.y) - (v3.x * v1.y))) + (v3.z * ((v1.x * v2.y) - (v2.x * v1.y)))
        );

        float invB11, invB12, invB13,
              invB21, invB22, invB23,
              invB31, invB32, invB33;

        invB11 = ((v2.y * v3.z) - (v3.y * v2.z)); invB12 = ((v3.x * v2.z) - (v2.x * v3.z)); invB13 = ((v2.x * v3.y) - (v3.x * v2.y));
        invB21 = ((v3.y * v1.z) - (v1.y * v3.z)); invB22 = ((v1.x * v3.z) - (v3.x * v1.z)); invB23 = ((v3.x * v1.y) - (v1.x * v3.y));
        invB31 = ((v1.y * v2.z) - (v2.y * v1.z)); invB32 = ((v2.x * v1.z) - (v1.x * v2.z)); invB33 = ((v1.x * v2.y) - (v2.x * v1.y));

        invB11 /= detB; invB12 /= detB; invB13 /= detB;
        invB21 /= detB; invB22 /= detB; invB23 /= detB;
        invB31 /= detB; invB32 /= detB; invB33 /= detB;

        float T11, T12, T13,
              T21, T22, T23;
        //    T31, T32, T33;
        
        T11 = (vt1.x * invB11) + (vt2.x * invB21) + (vt3.x * invB31); T12 = (vt1.x * invB12) + (vt2.x * invB22) + (vt3.x * invB32); T13 = (vt1.x * invB13) + (vt2.x * invB23) + (vt3.x * invB33);
        T21 = (vt1.y * invB11) + (vt2.y * invB21) + (vt3.y * invB31); T22 = (vt1.y * invB12) + (vt2.y * invB22) + (vt3.y * invB32); T23 = (vt1.y * invB13) + (vt2.y * invB23) + (vt3.y * invB33);
        // T31 = (vt1.z * invB11) + (vt2.z * invB21) (vt3.z * invB31); T32 = (vt1.z * invB12) + (vt2.z * invB22) (vt3.z * invB32); T33 = (vt1.z * invB13) + (vt2.z * invB23) (vt3.z * invB33);

        float mapped_x = (hit.x * T11) + (hit.y * T12) + (hit.z * T13);
        float mapped_y = (hit.x * T21) + (hit.y * T22) + (hit.z * T23);

        frame[idx] = buffers.texture_buffer[(unsigned int)((texture * buffers.texture_size * buffers.texture_size) + (buffers.texture_size * buffers.texture_size * mapped_y) + (buffers.texture_size * mapped_x))];
    }else{
        frame[idx].x = point.x * 255;
        frame[idx].y = point.y * 255;
        frame[idx].z = point.z * 255;
        frame[idx].w = 255;
    }
}

__global__ void testKernel(uchar4 *d_frames, const unsigned int buffer_count, const unsigned int buffer, unsigned int blocks, const unsigned int width, const unsigned int height, void *user_pointer){
    uchar4 *frame = &d_frames[buffer * width * height];
    unsigned int idx = (blockDim.x * blockIdx.x) + threadIdx.x;

    while(idx < width * height){
        unsigned int x = idx % width;
        unsigned int y = idx / width;

        draw(frame, idx, x, y, width, height, user_pointer);

        idx += blockDim.x * blocks;
    }
}

#include <iostream>
void test_kernel(const unsigned int blocks, const unsigned int threads_per_block, const unsigned int shared_memory_per_block, cudaStream_t stream, uchar4 *d_frames, const unsigned int buffer_count, const unsigned int buffer, const unsigned int width, const unsigned int height, void *user_pointer){
    testKernel<<<blocks, threads_per_block, shared_memory_per_block, stream>>>(d_frames, buffer_count, buffer, blocks, width, height, user_pointer);
}
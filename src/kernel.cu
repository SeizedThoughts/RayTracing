#include "kernel.h"
#include <math.h>

#include "RTX.h"

using namespace RTX;

__device__ inline void texture_map(const Vertex &vt1, const Vertex &vt2, const Vertex &vt3, const Vertex &v1, const Vertex &v2, const Vertex &v3, const Vertex &hit){


}

__device__ inline void printVertex(Vertex vertex){
    printf("(%f, %f, %f)", vertex.x, vertex.y, vertex.z);
}

__device__ inline void printBuffers(Buffers buffers){
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
    
    unsigned int texture_id;
    unsigned int vertices[7];
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
                    //TODO: maybe just check where the hit is on the texture (u, v, w < 1 & > 0 ?)
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

                    vertices[6] = model.normal_start + face.vn;

                    texture_id = renderer.texture;
                }
            }
        }
    }

    if(closest != -1){
        /*
            working out for triangle mapping done here:
            https://www.desmos.com/calculator/xkip2fenoy
        */

        Vertex v1 = buffers.vertex_buffer[vertices[0]];
        Vertex v2 = buffers.vertex_buffer[vertices[1]];
        Vertex v3 = buffers.vertex_buffer[vertices[2]];
        Vertex vt1 = buffers.texture_vertex_buffer[vertices[3]];
        Vertex vt2 = buffers.texture_vertex_buffer[vertices[4]];
        Vertex vt3 = buffers.texture_vertex_buffer[vertices[5]];

        Vertex hit = camera.position + (point * closest);

        Vertex vu = v2 - v1;
        Vertex vv = v3 - v1;
        Vertex vw = buffers.normal_buffer[vertices[6]];

        Vertex vuxvv = vu.cross(vv);
        Vertex vvxvw = vv.cross(vw);
        Vertex vwxvu = vw.cross(vu);

        float u = ((hit - v1) * vvxvw) / (vu * vvxvw);
        float v = ((hit - v1) * vwxvu) / (vv * vwxvu);
        float w = ((hit - v1) * vuxvv) / (vw * vuxvv);

        Vertex vtu = vt2 - vt1;
        Vertex vtv = vt3 - vt1;
        Vertex vtw = Vertex({0, 0, 1}) - vt1;

        float mapped_x = (vtu.x * u) + (vtv.x * v) + (vtw.x * w) + vt1.x;
        float mapped_y = (vtu.y * u) + (vtv.y * v) + (vtw.y * w) + vt1.y;

        mapped_x -= (int)mapped_x;
        mapped_y -= (int)mapped_y;

        if(mapped_x < 1 && mapped_x > 0 && mapped_y < 1 && mapped_y > 0){
            Texture texture = buffers.texture_buffer[texture_id];

            unsigned int tm = (texture.width * (unsigned int)(texture.height * mapped_y)) + (unsigned int)(texture.width * mapped_x);

            frame[idx] = texture.pixel_buffer[tm];
        }
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
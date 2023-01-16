#pragma once

#include "RTX.h"

#include <fstream>

#include <iostream>

#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>


//TODO: remove
inline cudaError_t cuda_assert(const cudaError_t code, const char* const file, const unsigned int line){
    if(code != cudaSuccess){
        std::cout << "CUDA error \"" << cudaGetErrorString(code) << "\" (" << code << ") on line " << line << " in " << file << std::endl;
        exit(code);
    }

    return code;
}

#define cuda(...) cuda_assert(cuda##__VA_ARGS__, __FILE__, __LINE__);


int str_equal(const char *a, const char *b){
    int i = 0;
    while(a[i] != '\0' && b[i] != '\0' && a[i] == b[i]) i++;
    return a[i] == b[i];
}

//https://en.wikipedia.org/wiki/Fast_inverse_square_root
__host__ __device__ inline float inv_sqrt(float number){
	long i;
	float x2, y;

	x2 = number * 0.5;
	y = number;
	i = *(long*)&y;
	i = 0x5f3759df - (i >> 1);
	y = *(float*)&i;
	y = y * (1.5 - (x2 * y * y));   // 1st iteration
	// y = y * (1.5 - (x2 * y * y));   // 2nd iteration, this can be removed

	return y;
}

__host__ __device__ int RTX::Vertex::operator == (const RTX::Vertex &vert) const{
    return (x == vert.x) * (y == vert.y) * (z == vert.z);
}

__host__ __device__ int RTX::Vertex::operator != (const RTX::Vertex &vert) const{
    return 1 - ((x == vert.x) * (y == vert.y) * (z == vert.z));
}

__host__ __device__ RTX::Vertex& RTX::Vertex::operator = (const RTX::Vertex &vert){
    x = vert.x;
    y = vert.y;
    z = vert.z;

    return *this;
}

__host__ __device__ RTX::Vertex& RTX::Vertex::operator = (const float array[3]){
    x = array[0];
    y = array[1];
    z = array[2];

    return *this;
}

__host__ __device__ RTX::Vertex RTX::Vertex::operator + (const RTX::Vertex &vert) const{
    return Vertex({
        x + vert.x,
        y + vert.y,
        z + vert.z
    });
}

__host__ __device__ RTX::Vertex& RTX::Vertex::operator += (const RTX::Vertex &vert){
    x += vert.x;
    y += vert.y;
    z += vert.z;

    return *this;
}

__host__ __device__ RTX::Vertex RTX::Vertex::operator - (const RTX::Vertex &vert) const{
    return Vertex({
        x - vert.x,
        y - vert.y,
        z - vert.z
    });
}

__host__ __device__ RTX::Vertex& RTX::Vertex::operator -= (const RTX::Vertex &vert){
    x -= vert.x;
    y -= vert.y;
    z -= vert.z;

    return *this;
}

__host__ __device__ float RTX::Vertex::operator * (const RTX::Vertex &vert) const{
    return (x * vert.x) + (y * vert.y) + (z * vert.z);
}

__host__ __device__ RTX::Vertex RTX::Vertex::operator * (const float scalar) const{
    return Vertex({
        x * scalar,
        y * scalar,
        z * scalar
    });
}

__host__ __device__ RTX::Vertex& RTX::Vertex::operator *= (const float scalar){
    x *= scalar;
    y *= scalar;
    z *= scalar;

    return *this;
}

__host__ __device__ RTX::Vertex& RTX::Vertex::normalize(){
    return *this *= inv_sqrt((x * x) + (y * y) + (z * z));
}

__host__ __device__ RTX::Vertex RTX::Vertex::cross(const RTX::Vertex &vert) const{
    /*
        a * b
        =
        det(
            i   j   k
            a.x a.y a.z
            b.x b.y b.z
        )
        =
        i((a.y * b.z) - (a.z * b.y)) + j((a.z * b.x) - (a.x * b.z)) + k((a.x * b.y) - (a.y * b.x))
        =
        x: (a.y * b.z) - (a.z * b.y)
        y: (a.z * b.x) - (a.x * b.z)
        z: (a.x * b.y) - (a.y * b.x)
    */

    return Vertex({
        (y * vert.z) - (z * vert.y),
        (z * vert.x) - (x * vert.z),
        (x * vert.y) - (y * vert.x)
    });
}

__host__ __device__ int RTX::Quaternion::operator == (const RTX::Quaternion &quat) const{
    return (w == quat.w) * (x == quat.x) * (y == quat.y) * (z == quat.z);
}

__host__ __device__ int RTX::Quaternion::operator != (const RTX::Quaternion &quat) const{
    return 1 - ((w == quat.w) * (x == quat.x) * (y == quat.y) * (z == quat.z));
}

__host__ __device__ RTX::Quaternion& RTX::Quaternion::operator = (const RTX::Quaternion &quat){
    w = quat.w;
    x = quat.x;
    y = quat.y;
    z = quat.z;

    return *this;
}

__host__ __device__ RTX::Quaternion& RTX::Quaternion::operator = (const float array[4]){
    w = array[0];
    x = array[1];
    y = array[2];
    z = array[3];

    return *this;
}

/*
    Based on the table here:
    https://en.wikipedia.org/wiki/Quaternion
    a * b
    =
    w(a.w * b.w) + x(a.x * b.w) + y(a.y * b.w) + z(a.z * b.w) +
    x(a.w * b.x) - w(a.x * b.x) + z(a.y * b.x) - y(a.z * b.x) +
    y(a.w * b.y) - z(a.x * b.y) - w(a.y * b.y) + x(a.z * b.y) +
    z(a.w * b.z) + y(a.x * b.z) - x(a.y * b.z) - w(a.z * b.z)
    =
    w((a.w * b.w) - (a.x * b.x) - (a.y * b.y) - (a.z * b.z)) +
    x((a.x * b.w) + (a.w * b.x) + (a.z * b.y) - (a.y * b.z)) +
    y((a.y * b.w) - (a.z * b.x) + (a.w * b.y) + (a.x * b.z)) +
    z((a.z * b.w) + (a.y * b.x) - (a.x * b.y) + (a.w * b.z))
    =
    w: (a.w * b.w) - (a.x * b.x) - (a.y * b.y) - (a.z * b.z)
    x: (a.x * b.w) + (a.w * b.x) + (a.z * b.y) - (a.y * b.z)
    y: (a.y * b.w) - (a.z * b.x) + (a.w * b.y) + (a.x * b.z)
    z: (a.z * b.w) + (a.y * b.x) - (a.x * b.y) + (a.w * b.z)
*/
__host__ __device__ RTX::Quaternion RTX::Quaternion::operator * (const RTX::Quaternion &quat) const{
    return Quaternion({
        (w * quat.w) - (x * quat.x) - (y * quat.y) - (z * quat.z),
        (x * quat.w) + (w * quat.x) + (z * quat.y) - (y * quat.z),
        (y * quat.w) - (z * quat.x) + (w * quat.y) + (x * quat.z),
        (z * quat.w) + (y * quat.x) - (x * quat.y) + (w * quat.z)
    });
}

__host__ __device__ RTX::Quaternion& RTX::Quaternion::operator *= (const RTX::Quaternion &quat){
    const float tw = (w * quat.w) - (x * quat.x) - (y * quat.y) - (z * quat.z);
    const float tx = (x * quat.w) + (w * quat.x) + (z * quat.y) - (y * quat.z);
    const float ty = (y * quat.w) - (z * quat.x) + (w * quat.y) + (x * quat.z);

    z = (z * quat.w) + (y * quat.x) - (x * quat.y) + (w * quat.z);

    w = tw;
    x = tx;
    y = ty;

    return *this;
}

__host__ __device__ RTX::Quaternion RTX::Quaternion::operator * (const RTX::Vertex &vert) const{
    return Quaternion({
        (x * vert.x) - (y * vert.y) - (z * vert.z),
        (w * vert.x) + (z * vert.y) - (y * vert.z),
        (z * vert.x) + (w * vert.y) + (x * vert.z),
        (y * vert.x) - (x * vert.y) + (w * vert.z)
    });
}

__host__ __device__ RTX::Quaternion& RTX::Quaternion::operator *= (const RTX::Vertex &vert){
    const float tw = (x * vert.x) - (y * vert.y) - (z * vert.z);
    const float tx = (w * vert.x) + (z * vert.y) - (y * vert.z);
    const float ty = (z * vert.x) + (w * vert.y) + (x * vert.z);

    z = (y * vert.x) - (x * vert.y) + (w * vert.z);

    w = tw;
    x = tx;
    y = ty;

    return *this;
}

__host__ __device__ RTX::Quaternion& RTX::Quaternion::normalize(){
    const float scalar = inv_sqrt((w * w) + (x * x) + (y * y) + (z * z));

    w *= scalar;
    x *= scalar;
    y *= scalar;
    z *= scalar;

    return *this;
}

__host__ __device__ RTX::Vertex RTX::Quaternion::rotate(const RTX::Vertex &vert)const{
    const float tw = (x * vert.x) - (y * vert.y) - (z * vert.z);
    const float tx = (w * vert.x) + (z * vert.y) - (y * vert.z);
    const float ty = (z * vert.x) + (w * vert.y) + (x * vert.z);
    const float tz = (y * vert.x) - (x * vert.y) + (w * vert.z);

    return Vertex({
        (tx * w) - (tw * x) - (tz * y) + (ty * z),
        (ty * w) + (tz * x) - (tw * y) - (tx * z),
        (tz * w) - (ty * x) + (tx * y) - (tw * z)
    });
}

__host__ __device__ RTX::Vertex& RTX::Quaternion::rotateInPlace(RTX::Vertex &vert)const{
    const float tw = (x * vert.x) - (y * vert.y) - (z * vert.z);
    const float tx = (w * vert.x) + (z * vert.y) - (y * vert.z);
    const float ty = (z * vert.x) + (w * vert.y) + (x * vert.z);
    const float tz = (y * vert.x) - (x * vert.y) + (w * vert.z);

    vert.x = (tx * w) - (tw * x) - (tz * y) + (ty * z);
    vert.y = (ty * w) + (tz * x) - (tw * y) - (tx * z);
    vert.z = (tz * w) - (ty * x) + (tx * y) - (tw * z);

    return vert;
}

__host__ __device__ RTX::Face& RTX::Face::operator = (const int array[3]){
    v1 = array[0];
    v2 = array[1];
    v3 = array[2];

    vt1 = -1;
    vt2 = -1;
    vt3 = -1;
    
    vn = -1;

    return *this;
}

__host__ __device__ RTX::Vertex RTX::Transform::transform(const RTX::Vertex &vert) const{
    Vertex point = Vertex({
        vert.x * scale.x,
        vert.y * scale.y,
        vert.z * scale.z
    });

    rotation.rotateInPlace(point);

    return point += position;
}

__host__ __device__ RTX::Vertex& RTX::Transform::transformInPlace(RTX::Vertex &vert) const{
    rotation.rotateInPlace(vert);
    vert.x *= scale.x;
    vert.y *= scale.y;
    vert.z *= scale.z;
    vert += position;
    return vert;
}
        
__host__ __device__ RTX::Vertex RTX::Camera::getPoint(){
    return rotation.rotateInPlace((Vertex&)Vertex({0, 0, 1}));
}

//https://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToQuaternion/steps/index.htm
__host__ __device__ RTX::Vertex RTX::Camera::getPoint(const int width, const int height, const int x, const int y){
    //only perspective right now
    Vertex t = rotation.rotateInPlace(Quaternion({cosf(y_fov * 0.25), sinf(y_fov * 0.25), 0, 0}).rotateInPlace((Vertex&)Vertex({0, 0, 1}))).normalize();
    Vertex l = rotation.rotateInPlace(Quaternion({cosf(x_fov * 0.25), 0, sinf(x_fov * 0.25), 0}).rotateInPlace((Vertex&)Vertex({0, 0, 1}))).normalize();
    Vertex r = rotation.rotateInPlace(Quaternion({cosf(x_fov * 0.25), 0, -sinf(x_fov * 0.25), 0}).rotateInPlace((Vertex&)Vertex({0, 0, 1}))).normalize();

    Vertex dr = r - l;
    Vertex tl = t - (dr * 0.5);

    return (tl + ((t - tl) * ((float)x / (float) width * 2)) + ((l - tl) * ((float)y / (float) height * 2))).normalize();
}

std::ostream& operator << (std::ostream &os, const RTX::Vertex &vert){
    os << "(" << vert.x << ", " << vert.y << ", " << vert.z << ")";

    return os;
}

std::ostream& operator << (std::ostream &os, const RTX::Quaternion &quat){
    os << "(" << quat.w << ", " << quat.x << ", " << quat.y << ", " << quat.z << ")";

    return os;
}

std::ostream& operator << (std::ostream &os, const RTX::Face &face){
    os << "(v: (" << face.v1 << ", " << face.v2 << ", " << face.v3 << "), vt: (" << face.vt1 << ", " << face.vt2 << ", " << face.vt3 << "), vn: " << face.vn << ")";

    return os;
}

RTX::Buffers RTX::buffers;
RTX::Buffers *RTX::d_buffers;
RTX::Vertex *d_vertex_buffer;
RTX::Vertex *d_normal_buffer;
RTX::Vertex *d_texture_vertex_buffer;
RTX::Face *d_face_buffer;
RTX::Camera *d_camera_buffer;
RTX::Model *d_model_buffer;
RTX::Renderer *d_renderer_buffer;
uchar4 *d_texture_buffer;

void RTX::makeBuffers(
    unsigned int max_vertex_count,
    unsigned int max_normal_count,
    unsigned int max_texture_vertex_count,
    unsigned int max_face_count,
    unsigned int max_camera_count,
    unsigned int max_model_count,
    unsigned int max_texture_count,
    unsigned int max_renderer_count
){
    buffers.texture_size = 512;

    buffers.vertex_count = 0;
    buffers.normal_count = 0;
    buffers.texture_vertex_count = 0;
    buffers.face_count = 0;
    buffers.camera_count = 0;
    buffers.model_count = 0;
    buffers.texture_count = 0;
    buffers.renderer_count = 0;

    buffers.max_vertex_count = max_vertex_count;
    buffers.max_normal_count = max_normal_count;
    buffers.max_texture_vertex_count = max_texture_vertex_count;
    buffers.max_face_count = max_face_count;
    buffers.max_camera_count = max_camera_count;
    buffers.max_model_count = max_model_count;
    buffers.max_texture_count = max_texture_count;
    buffers.max_renderer_count = max_renderer_count;

    buffers.vertex_buffer = (Vertex*)malloc(buffers.max_vertex_count * sizeof(Vertex));
    buffers.normal_buffer = (Vertex*)malloc(buffers.max_normal_count * sizeof(Vertex));
    buffers.texture_vertex_buffer = (Vertex*)malloc(buffers.max_texture_vertex_count * sizeof(Vertex));
    buffers.face_buffer = (Face*)malloc(buffers.max_face_count * sizeof(Face));
    buffers.camera_buffer = (Camera*)malloc(buffers.max_camera_count * sizeof(Camera));
    buffers.model_buffer = (Model*)malloc(buffers.max_model_count * sizeof(Model));
    buffers.texture_buffer = (uchar4*)malloc(buffers.max_texture_count * buffers.texture_size * buffers.texture_size * sizeof(uchar4));
    buffers.renderer_buffer = (Renderer*)malloc(buffers.max_renderer_count * sizeof(Renderer));

    cuda(Malloc((void**)&d_buffers, 1 * sizeof(Buffers)));

    cuda(Memcpy((void*)&d_buffers->texture_size, (void*)&buffers.texture_size, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));

    cuda(Memcpy((void*)&d_buffers->vertex_count, (void*)&buffers.vertex_count, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    cuda(Memcpy((void*)&d_buffers->normal_count, (void*)&buffers.normal_count, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    cuda(Memcpy((void*)&d_buffers->texture_vertex_count, (void*)&buffers.texture_vertex_count, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    cuda(Memcpy((void*)&d_buffers->face_count, (void*)&buffers.face_count, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    cuda(Memcpy((void*)&d_buffers->camera_count, (void*)&buffers.camera_count, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    cuda(Memcpy((void*)&d_buffers->model_count, (void*)&buffers.model_count, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    cuda(Memcpy((void*)&d_buffers->texture_count, (void*)&buffers.texture_count, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    cuda(Memcpy((void*)&d_buffers->renderer_count, (void*)&buffers.renderer_count, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));

    cuda(Memcpy((void*)&d_buffers->max_vertex_count, (void*)&buffers.max_vertex_count, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    cuda(Memcpy((void*)&d_buffers->max_normal_count, (void*)&buffers.max_normal_count, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    cuda(Memcpy((void*)&d_buffers->max_texture_vertex_count, (void*)&buffers.max_texture_vertex_count, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    cuda(Memcpy((void*)&d_buffers->max_face_count, (void*)&buffers.max_face_count, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    cuda(Memcpy((void*)&d_buffers->max_camera_count, (void*)&buffers.max_camera_count, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    cuda(Memcpy((void*)&d_buffers->max_model_count, (void*)&buffers.max_model_count, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    cuda(Memcpy((void*)&d_buffers->max_texture_count, (void*)&buffers.max_texture_count, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    cuda(Memcpy((void*)&d_buffers->max_renderer_count, (void*)&buffers.max_renderer_count, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));

    cuda(Malloc((void**)&d_vertex_buffer, buffers.max_vertex_count * sizeof(Vertex)));
    cuda(Malloc((void**)&d_normal_buffer, buffers.max_normal_count * sizeof(Vertex)));
    cuda(Malloc((void**)&d_texture_vertex_buffer, buffers.max_texture_vertex_count * sizeof(Vertex)));
    cuda(Malloc((void**)&d_face_buffer, buffers.max_face_count * sizeof(Face)));
    cuda(Malloc((void**)&d_camera_buffer, buffers.max_camera_count * sizeof(Camera)));
    cuda(Malloc((void**)&d_model_buffer, buffers.max_model_count * sizeof(Model)));
    cuda(Malloc((void**)&d_texture_buffer, buffers.max_texture_count * buffers.texture_size * buffers.texture_size * sizeof(uchar4)));
    cuda(Malloc((void**)&d_renderer_buffer, buffers.max_renderer_count * sizeof(Renderer)));

    cuda(Memcpy((void*)&d_buffers->vertex_buffer, (void*)&d_vertex_buffer, 1 * sizeof(Vertex*), cudaMemcpyHostToDevice));
    cuda(Memcpy((void*)&d_buffers->normal_buffer, (void*)&d_normal_buffer, 1 * sizeof(Vertex*), cudaMemcpyHostToDevice));
    cuda(Memcpy((void*)&d_buffers->texture_vertex_buffer, (void*)&d_texture_vertex_buffer, 1 * sizeof(Vertex*), cudaMemcpyHostToDevice));
    cuda(Memcpy((void*)&d_buffers->face_buffer, (void*)&d_face_buffer, 1 * sizeof(Face*), cudaMemcpyHostToDevice));
    cuda(Memcpy((void*)&d_buffers->camera_buffer, (void*)&d_camera_buffer, 1 * sizeof(Camera*), cudaMemcpyHostToDevice));
    cuda(Memcpy((void*)&d_buffers->model_buffer, (void*)&d_model_buffer, 1 * sizeof(Model*), cudaMemcpyHostToDevice));
    cuda(Memcpy((void*)&d_buffers->texture_buffer, (void*)&d_texture_buffer, 1 * sizeof(uchar4*), cudaMemcpyHostToDevice));
    cuda(Memcpy((void*)&d_buffers->renderer_buffer, (void*)&d_renderer_buffer, 1 * sizeof(Renderer*), cudaMemcpyHostToDevice));
}

//model, texture map, material map, maybe normal map
int RTX::load(const char *file_path){
    int loc = 0;
    char *file_extension;
    while(file_path[loc] != '\0'){
        if(file_path[loc] == '.'){
            file_extension = (char *)&file_path[loc + 1];
        }

        loc++;
    }

    if(str_equal(file_extension, "obj")){
        //obj control path
        std::ifstream obj(file_path);
        obj >> std::noskipws;

        unsigned int vertex_start = buffers.vertex_count;
        unsigned int normal_start = buffers.normal_count;
        unsigned int texture_vertex_start = buffers.texture_vertex_count;
        unsigned int face_start = buffers.face_count;
        char type;
        
        while(obj.good() && !obj.eof()){
            obj >> type;

            switch(type){
                case 'v':
                    obj >> type;
                    float x, y, z;
                    switch(type){
                        case ' ':
                            if(buffers.vertex_count >= buffers.max_vertex_count){
                                //TODO: throw some error
                                std::cout << "Too many vertices: " << buffers.vertex_count << std::endl;
                                break;
                            }
                            obj >> x >> type >> y >> type >> z;
                            buffers.vertex_buffer[buffers.vertex_count++] = {x, y, z};
                            break;
                        case 'n':
                            if(buffers.normal_count >= buffers.max_normal_count){
                                //TODO: throw some error
                                std::cout << "Too many normals: " << buffers.normal_count << std::endl;
                                break;
                            }
                            obj >> type >> x >> type >> y >> type >> z;
                            buffers.normal_buffer[buffers.normal_count++] = {x, y, z};
                            break;
                        case 't':
                            if(buffers.texture_vertex_count >= buffers.max_texture_vertex_count){
                                //TODO: throw some error
                                std::cout << "Too many texture vertices: " << buffers.texture_vertex_count << std::endl;
                                break;
                            }
                            obj >> type >> x >> type >> y;
                            if(obj.peek() == '\n'){
                                z = 0;
                            }else{
                                obj >> type >> z;
                            }
                            buffers.texture_vertex_buffer[buffers.texture_vertex_count++] = {x, y, z};
                            break;
                    }
                    break;
                case 'f':
                    if(buffers.face_count >= buffers.max_face_count){
                        //TODO: throw some error
                        std::cout << "Too many faces: " << buffers.face_count << std::endl;
                        break;
                    }
                    unsigned int v[3];
                    obj >> type >> v[0] >> type;
                    switch(type){
                        case ' ':
                            obj >> v[1] >> type >> v[2];
                            v[0]--;
                            v[1]--;
                            v[2]--;

                            /*
                                https://stackoverflow.com/questions/3780493/map-points-between-two-triangles-in-3d-space
                                v4 = v1 + (v2 - v1) x (v3 - v1)
                                vt4 = vt1 + (vt2 - vt1) x (vt3 - vt1)
                            */

                            buffers.normal_buffer[buffers.normal_count] = (buffers.vertex_buffer[v[1]] - buffers.vertex_buffer[v[0]]).cross(buffers.vertex_buffer[v[2]] - buffers.vertex_buffer[v[0]]);

                            buffers.vertex_buffer[buffers.vertex_count] = buffers.vertex_buffer[v[0]] + buffers.normal_buffer[buffers.normal_count];
                            
                            buffers.normal_buffer[buffers.normal_count].normalize();

                            buffers.face_buffer[buffers.face_count++] = {(int)v[0], (int)v[1], (int)v[2], (int)(buffers.vertex_count++), -1, -1, -1, -1, (int)(buffers.normal_count++)};
                            break;
                        case '/':
                            if(obj.peek() == '/'){
                                //v//vn
                                unsigned int vn[3];

                                obj >> type >> vn[0] >> type >> v[1] >> type >> type >> vn[1] >> type >> v[2] >> type >> type >> vn[2];
                                
                                v[0]--;
                                v[1]--;
                                v[2]--;

                                vn[0]--;
                                vn[1]--;
                                vn[2]--;

                                if(!(vn[0] == vn[1] && vn[0] == vn[2])){
                                    buffers.normal_buffer[buffers.normal_count] = (buffers.normal_buffer[vn[0]] + buffers.normal_buffer[vn[1]] + buffers.normal_buffer[vn[2]]).normalize();
                                
                                    vn[0] = buffers.normal_count++;
                                }

                                /*
                                    https://stackoverflow.com/questions/3780493/map-points-between-two-triangles-in-3d-space
                                    v4 = v1 + (v2 - v1) x (v3 - v1)
                                    vt4 = vt1 + (vt2 - vt1) x (vt3 - vt1)
                                */

                                buffers.vertex_buffer[buffers.vertex_count] = buffers.vertex_buffer[v[0]] + (buffers.vertex_buffer[v[1]] - buffers.vertex_buffer[v[0]]).cross(buffers.vertex_buffer[v[2]] - buffers.vertex_buffer[v[0]]);

                                buffers.face_buffer[buffers.face_count++] = {(int)v[0], (int)v[1], (int)v[2], (int)(buffers.vertex_count++), -1, -1, -1, -1, (int)vn[0]};
                            }else{
                                unsigned int vt[3];
                                obj >> vt[0];
                                if(obj.peek() == '/'){
                                    //v/vt/vn
                                    unsigned int vn[3];

                                    obj >> type >> vn[0] >> type >> v[1] >> type >> vt[1] >> type >> vn[1] >> type >> v[2] >> type >> vt[2] >> type >> vn[2];

                                    v[0]--;
                                    v[1]--;
                                    v[2]--;

                                    vt[0]--;
                                    vt[1]--;
                                    vt[2]--;

                                    vn[0]--;
                                    vn[1]--;
                                    vn[2]--;

                                    if(!(vn[0] == vn[1] && vn[0] == vn[2])){
                                        buffers.normal_buffer[buffers.normal_count] = (buffers.normal_buffer[vn[0]] + buffers.normal_buffer[vn[1]] + buffers.normal_buffer[vn[2]]).normalize();
                                    
                                        vn[0] = buffers.normal_count++;
                                    }

                                    /*
                                        https://stackoverflow.com/questions/3780493/map-points-between-two-triangles-in-3d-space
                                        v4 = v1 + (v2 - v1) x (v3 - v1)
                                        vt4 = vt1 + (vt2 - vt1) x (vt3 - vt1)
                                    */

                                    buffers.vertex_buffer[buffers.vertex_count++] = buffers.vertex_buffer[v[0]] + (buffers.vertex_buffer[v[1]] - buffers.vertex_buffer[v[0]]).cross(buffers.vertex_buffer[v[2]] - buffers.vertex_buffer[v[0]]);
                                    buffers.vertex_buffer[buffers.vertex_count] = buffers.texture_vertex_buffer[vt[0]] + (buffers.texture_vertex_buffer[vt[1]] - buffers.texture_vertex_buffer[vt[0]]).cross(buffers.texture_vertex_buffer[vt[2]] - buffers.texture_vertex_buffer[vt[0]]);

                                    buffers.face_buffer[buffers.face_count++] = {(int)v[0], (int)v[1], (int)v[2], (int)(buffers.vertex_count - 1), (int)vt[0], (int)vt[1], (int)vt[2], (int)(buffers.vertex_count++), (int)vn[0]};
                                }else{
                                    //v/vt
                                    obj >> type >> v[1] >> type >> vt[1] >> type >> v[1] >> type >> vt[2];

                                    buffers.normal_buffer[buffers.normal_count] = (buffers.vertex_buffer[v[0]] + buffers.vertex_buffer[v[1]] + buffers.vertex_buffer[v[2]]);

                                    /*
                                        https://stackoverflow.com/questions/3780493/map-points-between-two-triangles-in-3d-space
                                        v4 = v1 + (v2 - v1) x (v3 - v1)
                                        vt4 = vt1 + (vt2 - vt1) x (vt3 - vt1)
                                    */

                                    buffers.vertex_buffer[buffers.vertex_count++] = buffers.vertex_buffer[v[0]] + buffers.normal_buffer[buffers.normal_count];
                                    buffers.vertex_buffer[buffers.vertex_count] = buffers.texture_vertex_buffer[vt[0]] + (buffers.texture_vertex_buffer[vt[1]] - buffers.texture_vertex_buffer[vt[0]]).cross(buffers.texture_vertex_buffer[vt[2]] - buffers.texture_vertex_buffer[vt[0]]);

                                    buffers.normal_buffer[buffers.normal_count].normalize();

                                    buffers.face_buffer[buffers.face_count++] = {(int)v[0], (int)v[1], (int)v[2], (int)(buffers.vertex_count - 1), (int)vt[0], (int)vt[1], (int)vt[2], (int)(buffers.vertex_count++), (int)(buffers.normal_count++)};
                                }
                            }
                            break;
                    }
                    break;
                default:
                    while(type != '\n' && !obj.eof()){
                        obj >> type;
                    }
                    break;
            }
        }

        if(!obj.good() && !obj.eof()){
            std::cout << "Something went wrong reading file: " << file_path << std::endl;
        }

        obj.close();

        if(buffers.vertex_count > vertex_start){
            cuda(Memcpy((void*)&d_buffers->vertex_count, (void*)&buffers.vertex_count, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
            
            cuda(Memcpy((void*)&d_vertex_buffer[vertex_start], (void*)&buffers.vertex_buffer[vertex_start], (buffers.vertex_count - vertex_start) * sizeof(Vertex), cudaMemcpyHostToDevice));
        }
        
        if(buffers.normal_count > normal_start){
            cuda(Memcpy((void*)&d_buffers->normal_count, (void*)&buffers.normal_count, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
            
            cuda(Memcpy((void*)&d_normal_buffer[normal_start], (void*)&buffers.normal_buffer[normal_start], (buffers.normal_count - normal_start) * sizeof(Vertex), cudaMemcpyHostToDevice));
        }
        
        if(buffers.texture_vertex_count > texture_vertex_start){
            cuda(Memcpy((void*)&d_buffers->texture_vertex_count, (void*)&buffers.texture_vertex_count, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
            
            cuda(Memcpy((void*)&d_texture_vertex_buffer[texture_vertex_start], (void*)&buffers.texture_vertex_buffer[texture_vertex_start], (buffers.texture_vertex_count - texture_vertex_start) * sizeof(Vertex), cudaMemcpyHostToDevice));
        }
        
        if(buffers.face_count > face_start){
            cuda(Memcpy((void*)&d_buffers->face_count, (void*)&buffers.face_count, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
            
            cuda(Memcpy((void*)&d_face_buffer[face_start], (void*)&buffers.face_buffer[face_start], (buffers.face_count - face_start) * sizeof(Face), cudaMemcpyHostToDevice));
        }

        buffers.model_buffer[buffers.model_count] = {vertex_start, normal_start, texture_vertex_start, face_start, buffers.face_count - face_start};

        cuda(Memcpy((void*)&d_model_buffer[buffers.model_count], (void*)&buffers.model_buffer[buffers.model_count], 1 * sizeof(Model), cudaMemcpyHostToDevice));
    
        buffers.model_count++;

        cuda(Memcpy((void*)&d_buffers->model_count, (void*)&buffers.model_count, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));

        return buffers.model_count - 1;    
    }else{
        int width, height, channels;
        unsigned char *img = stbi_load(file_path, &width, &height, &channels, 0);
        
        if(img == NULL) {
            printf("Error in loading the image\n");
            return -1;
        }
        
        uchar4 *loc = &buffers.texture_buffer[buffers.texture_count * buffers.texture_size * buffers.texture_size];
        for(unsigned int i = 0; i < buffers.texture_size * buffers.texture_size; i++){
            loc[i].x = img[i * channels];
            loc[i].y = img[i * channels + 1];
            loc[i].z = img[i * channels + 2];
            loc[i].w = 255;
        }

        cuda(Memcpy((void*)&d_texture_buffer[buffers.texture_count * buffers.texture_size * buffers.texture_size], (void*)loc, buffers.texture_size * buffers.texture_size * sizeof(uchar4), cudaMemcpyHostToDevice));

        buffers.texture_count++;

        cuda(Memcpy((void*)&d_buffers->texture_count, (void*)&buffers.texture_count, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));

        return buffers.texture_count - 1;
    }
}

#define PI 3.14159265
int RTX::createCamera(){
    buffers.camera_buffer[buffers.camera_count] = {
        {0, 0, -10},
        {1, 0, 0, 0},
        PI * 0.5,
        PI * 0.5
    };

    cuda(Memcpy((void*)&d_camera_buffer[buffers.camera_count], (void*)&buffers.camera_buffer[buffers.camera_count], 1 * sizeof(Camera), cudaMemcpyHostToDevice));

    buffers.camera_count++;

    cuda(Memcpy((void*)&d_buffers->camera_count, (void*)&buffers.camera_count, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));

    return buffers.camera_count - 1;
}

void RTX::updateCamera(unsigned int camera){
    cuda(Memcpy((void*)&d_camera_buffer[camera], (void*)&buffers.camera_buffer[camera], 1 * sizeof(Camera), cudaMemcpyHostToDevice));
}

int RTX::createRenderer(unsigned int model, unsigned int texture){
    buffers.renderer_buffer[buffers.renderer_count] = {
        model,
        texture,
        Transform({
            {0, 0, 0},
            {1, 0, 0, 0},
            {1, 1, 1}
        })
    };

    cuda(Memcpy((void*)&d_renderer_buffer[buffers.renderer_count], (void*)&buffers.renderer_buffer[buffers.renderer_count], 1 * sizeof(Renderer), cudaMemcpyHostToDevice));

    buffers.renderer_count++;

    cuda(Memcpy((void*)&d_buffers->renderer_count, (void*)&buffers.renderer_count, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));

    return buffers.renderer_count - 1;
}
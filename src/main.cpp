#include <pee.h>

#include "kernel.h"

#include <cuda_runtime.h>

#include "RTX.h"

#include <iostream>

#define WINDOW_WIDTH 512
#define WINDOW_HEIGHT 512
#define WINDOW_NAME "Window"

#define PI 3.14159265

#define PEE_DEBUG

int focused = 0;

int main(int argc, char** argv){
    pee::initOpenGL(argc, argv);
    pee::setDevice();
    pee::createWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_NAME);
    pee::createBuffers(2);
    pee::setKernel(test_kernel);

    RTX::makeBuffers(64, 64, 64, 64, 1, 1, 1, 1);

    RTX::createRenderer(RTX::load("obj/cube.obj"), RTX::load("textures/crate.jpg"));

    RTX::createCamera();

    pee::setUserPointer((void *)RTX::d_buffers);

    pee::setKeyboardFunction([](unsigned char key, int x, int y){
        if(focused){
            float speed = 0.25;
            switch(key){
                case 'w':
                    RTX::buffers.camera_buffer[0].position += RTX::buffers.camera_buffer[0].rotation.rotateInPlace(RTX::Vertex({0, 0, 1})) * speed;
                    break;
                case 's':
                    RTX::buffers.camera_buffer[0].position -= RTX::buffers.camera_buffer[0].rotation.rotateInPlace(RTX::Vertex({0, 0, 1})) * speed;
                    break;
                case 'a':
                    RTX::buffers.camera_buffer[0].position -= RTX::Vertex({0, 1, 0}).cross(RTX::buffers.camera_buffer[0].rotation.rotateInPlace(RTX::Vertex({0, 0, 1}))).normalize() * speed;
                    break;
                case 'd':
                    RTX::buffers.camera_buffer[0].position += RTX::Vertex({0, 1, 0}).cross(RTX::buffers.camera_buffer[0].rotation.rotateInPlace(RTX::Vertex({0, 0, 1}))).normalize() * speed;
                    break;
                case 32:
                    RTX::buffers.camera_buffer[0].position.y += speed;
                    break;
                case 'h':
                    RTX::buffers.camera_buffer[0].position.y -= speed;
                    break;
                case 27:
                    focused = 0;
                    break;
                default:
                    std::cout << (int)key << std::endl;
                    break;
            }

            RTX::updateCamera(0);

            pee::requestRedisplay();
        }
    });

    pee::setMouseFunction([](int button, int state, int x, int y){
        if(button == 0){
            focused = 1;
        }
    });

    // pee::setReshapeFunction([](int x, int y){
    //     std::cout << "Resize: " << x << ", " << y << std::endl;
    // });

    pee::setPassiveMotionFunction([](int x, int y){
        if(focused){
            static int just_warped = 0;
            glutSetCursor(GLUT_CURSOR_NONE);

            if(just_warped){
                just_warped = 0;
                return;
            }

            float a = (pee::width * 0.5 - x) * PI / 1000;

            float yaw_w = cosf(a * 0.5);
            float yaw_y = sinf(a * 0.5);

            a = (pee::height * 0.5 - y) * PI / 1000;
            float sin_phi = sinf(a * 0.5);

            RTX::Vertex left = RTX::Vertex({0, 1, 0}).cross(RTX::buffers.camera_buffer[0].rotation.rotateInPlace(RTX::Vertex({0, 0, 1}))).normalize();

            left.x *= sin_phi;
            left.y *= sin_phi;
            left.z *= sin_phi;

            RTX::buffers.camera_buffer[0].rotation = (RTX::Quaternion({yaw_w, 0, yaw_y, 0}).normalize() * RTX::Quaternion({cosf(a * 0.5), left.x, left.y, left.z}).normalize() * RTX::buffers.camera_buffer[0].rotation).normalize();

            just_warped = 1;

            RTX::updateCamera(0);

            glutWarpPointer(pee::width * 0.5, pee::height * 0.5);
            
            pee::requestRedisplay();
        }else{
            glutSetCursor(GLUT_CURSOR_INHERIT);
        }
    });

    pee::start();

    return 0;
}
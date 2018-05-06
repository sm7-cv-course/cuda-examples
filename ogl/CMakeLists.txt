find_package(CUDA REQUIRED)
find_package(OpenGL 2.1 REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(PkgConfig)
pkg_check_modules(GLFW REQUIRED glfw3)

find_package(GLUT)
if(GLUT_FOUND)
    include_directories(${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR})
endif(GLUT_FOUND)

pkg_search_module(SDL2 sdl2)
if(SDL2_FOUND)
    include_directories(${SDL2_INCLUDE_DIRS})
endif(SDL2_FOUND)


set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(${OPENGL_INCLUDE_DIR} ${GLUT_INCLUDE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)

cuda_add_executable(cuda-ogl basic_interop.cu)
cuda_add_executable(heat heat.cu)


target_link_libraries(cuda-ogl ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} m)
target_link_libraries(heat ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} m)

cuda_add_cufft_to_target(cuda-ogl)
cuda_add_cufft_to_target(heat)

#cuda_add_executable(cuda-test cuda-test.cu)

#cuda_add_executable(fft fft.cu)



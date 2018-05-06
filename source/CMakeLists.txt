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



cuda_add_executable(cuda-cufft simpleCUFFT.cu)
target_link_libraries(cuda-cufft ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} m)
cuda_add_cufft_to_target(cuda-cufft)

#cuda_add_executable(simpleCUFFT_GPUanim simpleCUFFT_GPUanim.cu)
#target_link_libraries(simpleCUFFT_GPUanim ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} m)
#cuda_add_cufft_to_target(simpleCUFFT_GPUanim)

#cuda_add_executable(fft fft.cu)

cuda_add_executable(simpleCUFFT_class simpleCUFFT_class.hpp simpleCUFFT_main.cu)
target_link_libraries(simpleCUFFT_class ${GLUT_LIBRARIES} ${OPENGL_LIBRARIES} m)
cuda_add_cufft_to_target(simpleCUFFT_class)

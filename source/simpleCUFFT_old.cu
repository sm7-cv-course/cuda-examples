#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex>
#include <math.h>
#include <iostream>
#include <valarray>
#include <time.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuComplex.h>
#include "device_launch_parameters.h"

#define BLOCKSIZE 2
#define Dim 8

const float PI = 3.141592653589793238460;

typedef std::complex<double> Complex;


__device__ double arg(double real, double imag)
{
    return atan(imag / real);
}

__device__ double abs_d(double real, double imag)
{
    return sqrt(real * real + imag * imag);
}

__device__ double real_d(double abs, double phase)
{
    return (abs*cos(phase));
}

__device__ double imag_d(double abs, double phase)
{
    return (abs*sin(phase)); 
}

__global__ void MyKernel(cufftDoubleComplex *d_target, cufftDoubleComplex *data, size_t d_pitch, int nx, bool flag)
{
    int ind_y = threadIdx.y + blockDim.y * blockIdx.y;
    int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    data[ind_y*nx + ind_x].x /= (Dim*Dim);
    data[ind_y*nx + ind_x].y /= (Dim*Dim);
    
    if (flag)
    {
        data[ind_y*nx + ind_x].x = abs_d(data[ind_y*nx + ind_x].x, data[ind_y*nx + ind_x].y);
        data[ind_y*nx + ind_x].y = arg(data[ind_y*nx + ind_x].x, data[ind_y*nx + ind_x].y);
        data[ind_y*nx + ind_x].x = d_target[ind_y*nx + ind_x].x;
        // a + ib
        data[ind_y*nx + ind_x].x = real_d(data[ind_y*nx + ind_x].x, data[ind_y*nx + ind_x].y);
        data[ind_y*nx + ind_x].y = imag_d(data[ind_y*nx + ind_x].x, data[ind_y*nx + ind_x].y);
    }
    else 
    {
        data[ind_y*nx + ind_x].x = abs_d(data[ind_y*nx + ind_x].x, data[ind_y*nx + ind_x].y);
        data[ind_y*nx + ind_x].y = arg(data[ind_y*nx + ind_x].x, data[ind_y*nx + ind_x].y);
        data[ind_y*nx + ind_x].x = d_target[ind_y*nx + ind_x].x;
        // a + ib
        data[ind_y*nx + ind_x].x = real_d(data[ind_y*nx + ind_x].x, data[ind_y*nx + ind_x].y);
        data[ind_y*nx + ind_x].y = imag_d(data[ind_y*nx + ind_x].x, data[ind_y*nx + ind_x].y);
    }
}

int main()
{
    const Complex test[Dim][Dim] = {
        Complex(1.0, PI / 13), Complex(1.0, PI / 11), Complex(1.0, PI / 4), Complex(1.0, PI / 6), Complex(1.0, PI), Complex(1.0, 5 * PI / 4), Complex(1.0, 7 * PI / 4), Complex(1.0, 6 * PI / 4),
        Complex(1.0, 20 * PI / 17), Complex(1.0, 11 * PI / 5), Complex(1.0, PI / 4), Complex(1.0, PI / 6), Complex(1.0, PI), Complex(1.0, 5 * PI / 4), Complex(1.0, 7 * PI / 4), Complex(1.0, 6 * PI / 4),
        Complex(1.0, PI / 2), Complex(1.0, PI / 3), Complex(1.0, PI / 4), Complex(1.0, PI / 6), Complex(1.0, PI), Complex(1.0, 5 * PI / 4), Complex(1.0, 7 * PI / 4), Complex(1.0, 6 * PI / 4),
        Complex(1.0, PI / 2), Complex(1.0, PI / 3), Complex(1.0, PI / 4), Complex(1.0, PI / 6), Complex(1.0, PI), Complex(1.0, 5 * PI / 4), Complex(1.0, 7 * PI / 4), Complex(1.0, 6 * PI / 4),
        Complex(1.0, PI / 2), Complex(1.0, PI / 3), Complex(1.0, PI / 4), Complex(1.0, PI / 6), Complex(1.0, PI), Complex(1.0, 5 * PI / 4), Complex(1.0, 7 * PI / 4), Complex(1.0, 6 * PI / 4),
        Complex(1.0, PI / 2), Complex(1.0, PI / 3), Complex(1.0, PI / 4), Complex(1.0, PI / 6), Complex(1.0, PI), Complex(1.0, 5 * PI / 4), Complex(1.0, 7 * PI / 4), Complex(1.0, 6 * PI / 4),
        Complex(1.0, PI / 2), Complex(1.0, PI / 3), Complex(1.0, PI / 4), Complex(1.0, PI / 6), Complex(1.0, PI), Complex(1.0, 5 * PI / 4), Complex(1.0, 7 * PI / 4), Complex(1.0, 6 * PI / 4),
        Complex(1.0, PI / 2), Complex(1.0, PI / 3), Complex(1.0, PI / 4), Complex(1.0, PI / 6), Complex(1.0, PI), Complex(1.0, 5 * PI / 4), Complex(1.0, 7 * PI / 4), Complex(1.0, 6 * PI / 4)
    };

    Complex target[Dim][Dim] = {
        Complex(0,0), Complex(0,0), Complex(0,0), Complex(0,0), Complex(0,0), Complex(0,0), Complex(0,0), Complex(0,0),
        Complex(0,0), Complex(0,0), Complex(0,0), Complex(0,0), Complex(0,0), Complex(0,0), Complex(0,0), Complex(0,0),
        Complex(0,0), Complex(0,0), Complex(0.4,0), Complex(0.4,0), Complex(0.4,0), Complex(0,0), Complex(0,0), Complex(0,0),
        Complex(0,0), Complex(0,0), Complex(0.4,0), Complex(1,0), Complex(1,0), Complex(0.4,0), Complex(0,0), Complex(0,0),
        Complex(0,0), Complex(0,0), Complex(0.4,0), Complex(1,0), Complex(1,0), Complex(0.4,0), Complex(0,0), Complex(0,0),
        Complex(0,0), Complex(0,0), Complex(0,0), Complex(0.4,0), Complex(0.4,0), Complex(0.4,0), Complex(0,0), Complex(0,0),
        Complex(0,0), Complex(0,0), Complex(0,0), Complex(0,0), Complex(0,0), Complex(0,0), Complex(0,0), Complex(0,0),
        Complex(0,0), Complex(0,0), Complex(0,0), Complex(0,0), Complex(0,0), Complex(0,0), Complex(0,0), Complex(0,0)
    };
     
    Complex ISO[Dim][Dim];

    memcpy(ISO, test, Dim * Dim * sizeof(Complex));
    
    for(size_t i = 0; i < Dim; i++)
    {
        for(size_t j = 0; j < Dim; j++)
            // a + ib
            ISO[i][j] = std::polar(real((ISO[i][j])), imag(ISO[i][j])); 
    }
    for (size_t i = 0; i < Dim; i++)
    {
        for(size_t j = 0; j < Dim; j++)
            // a + ib
            target[i][j] = std::polar(real((ISO[i][j])), imag(ISO[i][j])); 
    }

    std::cout << "initial matrix" << std::endl;

    for(size_t i = 0; i < Dim; i++)
    {
        for (size_t j = 0; j < Dim; j++)
            std::cout << ISO[i][j]; 
        std::cout << std::endl;
    }
    
    // size_t size = Dim * Dim * sizeof(cufftDoubleComplex);
    unsigned nx_mem_size = sizeof(Complex) * Dim;
    size_t d_pitch, h_pitch = nx_mem_size ;

    cufftDoubleComplex *dataInOutCuda;
    cufftDoubleComplex *data;
    cufftDoubleComplex *d_target;
    cudaMallocPitch((void**) &dataInOutCuda, &d_pitch, nx_mem_size, Dim);
    cudaMallocPitch((void**) &data, &d_pitch, nx_mem_size, Dim);
    cudaMallocPitch((void**) &d_target, &d_pitch, nx_mem_size, Dim);
    cudaMemcpy2D(d_target, d_pitch, target, h_pitch, nx_mem_size, Dim, cudaMemcpyHostToDevice);
    cudaMemcpy2D(dataInOutCuda, d_pitch, ISO, h_pitch, nx_mem_size, Dim, cudaMemcpyHostToDevice);
                          
    ///----------Device code
    
    dim3 Blk(BLOCKSIZE, BLOCKSIZE);
    dim3 Grd(BLOCKSIZE, BLOCKSIZE);
    cufftHandle planFFT, planIFFT;  
    cufftPlan2d(&planFFT, Dim, Dim * sizeof(cufftDoubleComplex), CUFFT_Z2Z);
    cufftPlan2d(&planIFFT, Dim, Dim * sizeof(cufftDoubleComplex), CUFFT_Z2Z);
    for(size_t i = 0; i < 150; i++)
    {
    	// dataInOutCuda -> data
        cufftExecZ2Z(planFFT, dataInOutCuda, data, CUFFT_FORWARD);
        cudaThreadSynchronize();
        MyKernel <<<Grd, Blk >>> (d_target, data, d_pitch, Dim, true);
        cudaThreadSynchronize();
        // data -> dataInOutCuda
        cufftExecZ2Z(planIFFT, data, dataInOutCuda, CUFFT_INVERSE);
        cudaThreadSynchronize();
        MyKernel <<<Grd, Blk >>> (d_target, dataInOutCuda, d_pitch, Dim, false);
        cudaThreadSynchronize();
    }
    Complex Tester[Dim][Dim] = { Complex(0,0), Complex(0,0), Complex(0,0), Complex(0,0) };
    cudaMemcpy2D(Tester, h_pitch, dataInOutCuda, d_pitch, nx_mem_size, Dim, cudaMemcpyDeviceToHost);

    ///   --- end device code

    std::cout << std::endl << "Here's your successes " << std::endl << std::endl;
    for(size_t i = 0; i < Dim; i++)
    {
        for (size_t j = 0; j < Dim; j++)
            std::cout << Tester[i][j]; 
        std::cout << std::endl;
    }

    cufftDestroy(planFFT);
    cufftDestroy(planIFFT);

    cudaFree(dataInOutCuda);
    cudaFree(data);

    //delete[] dataOut, dataIn;
    
    getchar();

    return 0;
}





//#include <iostream>
//#include <fstream>
//#include <vector>
//#include <stdio.h>
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <conio.h>
//#include <cufft.h>
//#include <complex>
//#include <cuComplex.h>
//
//#define BLOCK 3
////#define N (6*6)
//#define NRANK 2
//#define BATCH 1
//#define Dim 8 
//typedef std::complex<float> Complex;
//
//__global__ void Kernel (cufftComplex *dA, cufftComplex *target, size_t d_pitch, int nx)
//{
//    int ind_y = threadIdx.y + blockDim.y*blockIdx.y;
//    int ind_x = threadIdx.x + blockDim.x*blockIdx.x;
//    
//    
//}
//
//int main() {
//    using std::cout;
//    int N;
//
//    N = 6;
//    cufftComplex *dA;
//    cufftComplex *dAcom;
//    cufftComplex * d_target;
//    size_t d_pitch, h_pitch;
//
//    int size = 6 * 6;
//    unsigned int nxy_mem_size = sizeof(float)*size * 2;
//    unsigned int nx_mem_size = sizeof(float) * 6 * 2;
//
//    h_pitch = nx_mem_size;
//    
//    const float PI = 3.141592653589793238460;
//
//    Complex *hA;
//    Complex *hAout;
//    hA = (Complex*)malloc(nxy_mem_size);
//    hAout = (Complex*)malloc(nxy_mem_size);
//    /*for (int i = 0; i < 6; i++)
//    {
//        for (int j = 0; j < 6; j++)
//        {
//            hA[i * 6 + j] = 1;
//            hAout[i * 6 + j] = 1;
//        }
//    }
//
//    for (size_t i = 0; i < 6; i++)
//    {
//        for (size_t j = 0; j < 6; j++)
//            std::cout << hA[i * 6 + j];
//        std::cout << std::endl;
//    }
//    std::cout << "\n";
//    */
//    const Complex test[Dim*Dim] =
//
//    {   Complex(1.0, PI / 13), Complex(1.0, PI / 11), Complex(1.0, PI / 4), Complex(1.0, PI / 6), Complex(1.0, PI), Complex(1.0, 5 * PI / 4), Complex(1.0, 7 * PI / 4), Complex(1.0, 6 * PI / 4),
//        Complex(1.0,20 * PI / 17), Complex(1.0,11 * PI / 5), Complex(1.0, PI / 4), Complex(1.0, PI / 6), Complex(1.0, PI), Complex(1.0, 5 * PI / 4), Complex(1.0, 7 * PI / 4), Complex(1.0, 6 * PI / 4),
//        Complex(1.0, PI / 2), Complex(1.0, PI / 3), Complex(1.0, PI / 4), Complex(1.0, PI / 6), Complex(1.0, PI), Complex(1.0, 5 * PI / 4), Complex(1.0, 7 * PI / 4), Complex(1.0, 6 * PI / 4),
//        Complex(1.0, PI / 2), Complex(1.0, PI / 3), Complex(1.0, PI / 4), Complex(1.0, PI / 6), Complex(1.0, PI), Complex(1.0, 5 * PI / 4), Complex(1.0, 7 * PI / 4), Complex(1.0, 6 * PI / 4),
//        Complex(1.0, PI / 2), Complex(1.0, PI / 3), Complex(1.0, PI / 4), Complex(1.0, PI / 6), Complex(1.0, PI), Complex(1.0, 5 * PI / 4), Complex(1.0, 7 * PI / 4), Complex(1.0, 6 * PI / 4),
//        Complex(1.0, PI / 2), Complex(1.0, PI / 3), Complex(1.0, PI / 4), Complex(1.0, PI / 6), Complex(1.0, PI), Complex(1.0, 5 * PI / 4), Complex(1.0, 7 * PI / 4), Complex(1.0, 6 * PI / 4),
//        Complex(1.0, PI / 2), Complex(1.0, PI / 3), Complex(1.0, PI / 4), Complex(1.0, PI / 6), Complex(1.0, PI), Complex(1.0, 5 * PI / 4), Complex(1.0, 7 * PI / 4), Complex(1.0, 6 * PI / 4),
//        Complex(1.0, PI / 2), Complex(1.0, PI / 3), Complex(1.0, PI / 4), Complex(1.0, PI / 6), Complex(1.0, PI), Complex(1.0, 5 * PI / 4), Complex(1.0, 7 * PI / 4), Complex(1.0, 6 * PI / 4) };
//
    
//
//    //Complex ISO[2][2];              ///���������� ������  //����� �������� ������� ����� �������������� �����
//
//    memcpy(hA, test, 6 * 6 * sizeof(Complex));
//    //ISO[0][0] = std::polar(real((ISO[0][0])), imag(ISO[0][0]));
//
//    /*std::cout<<ISO[0][0]<<std::endl;
//    std::cout << abs(ISO[0][0]);*/
//    for (size_t i = 0; i < 6; i++)
//    {
//        for (size_t j = 0; j < 6; j++)
//            hA[i*6+j] = std::polar(real((test[i*6+j])), imag(test[i*6+j])); // ������ �� ������� ����� � �������� ������ ���� a + ib
//    }
//
//    std::cout << "initial matrix" << std::endl;
//
//    for (size_t i = 0; i < 6; i++)
//    {
//        for (size_t j = 0; j < 6; j++)
//            std::cout << hA[i * 6 + j];                                 // ������� ���������� 
//        std::cout << std::endl;
//    }
//    std::cout <<"\n";
//    //cudaMallocPitch ((void**) &dA, &d_pitch, nx_mem_size, 6 );  
//    //cudaMallocPitch ((void**) &dAcom, &d_pitch, nx_mem_size, 6 );
//    //cudaMallocPitch ((void**) &C, &d_pitch, nx_mem_size, 6 );
//
//    cudaMalloc((void**)&dA, sizeof(cufftComplex)*N*N);
//    cudaMalloc((void**)&dAcom, sizeof(cufftComplex)*N*N);
//    cudaMalloc((void**)&d_target, sizeof(cufftComplex)*N*N);   // ������ ��� target-�����������
//    
//                                                               
//    //cudaMemcpy2D ( dA, d_pitch, hA, h_pitch, nx_mem_size, 6, cudaMemcpyHostToDevice );
//    cudaMemcpy(dA, hA, sizeof(cufftComplex)*N*N, cudaMemcpyHostToDevice);
//
//    cudaMemcpy(d_target, target, sizeof(cufftComplex)*N*N, cudaMemcpyHostToDevice); // �������� target-����������� �� ������
//    cufftHandle plan;
//    cufftPlan2d(&plan, N, N, CUFFT_C2C);
//
//    cufftExecC2C(plan, dA, dAcom, CUFFT_FORWARD); // �� ������� ����� �������������� ���� ������������ ����������-������� ���, ���� ���������� CUDA �������
//
//
//
//    dim3 Blk (BLOCK, BLOCK);
//    dim3 Grd (6/BLOCK, 6/BLOCK);
//
//    Kernel <<<Grd, Blk>>> (dA, d_target, d_pitch, 6);
//
//    //cudaMemcpy2D( hAout, h_pitch, dAcom, d_pitch, nx_mem_size, 6, cudaMemcpyDeviceToHost );
//    cudaMemcpy(hAout, dAcom, sizeof(cufftComplex)*N*N, cudaMemcpyDeviceToHost);
//    
//    for (size_t i = 0; i < 6; i++)
//    {
//        for (size_t j = 0; j < 6; j++)
//            std::cout << hAout[i * 6 + j];
//        std::cout << std::endl;
//    }
//    std::cout << "\n";
//    cufftComplex *dAout;
//    cufftHandle plan1;
//
//    //cudaMallocPitch((void**)&dAout, &d_pitch, nx_mem_size, 6 );
//    cudaMalloc((void**)&dAout, sizeof(cufftComplex)*N*N);
//
//    cufftPlan2d(&plan1, N, N, CUFFT_C2C);
//
//    cufftExecC2C(plan1, dAcom, dAout, CUFFT_INVERSE);
//
//    //cudaMemcpy2D( hAout, h_pitch, dAout, d_pitch, nx_mem_size, 6, cudaMemcpyDeviceToHost );
//    cudaMemcpy(hAout, dAout, sizeof(cufftComplex)*N*N, cudaMemcpyDeviceToHost);
//    for (size_t i = 0; i < 6; i++)
//        for (size_t j = 0; j < 6; j++)
//        {
//            hAout[6*i+j] /= 36;
//        }
//    for (size_t i = 0; i < 6; i++)
//    {
//        for (size_t j = 0; j < 6; j++)
//            std::cout << hAout[i * 6 + j];
//        std::cout << std::endl;
//    }
//
//    //cufftDestroy(plan);
//    //cufftDestroy(plan1);
//    cudaFree(dA);
//    cudaFree(dAcom);
//    system("PAUSE");
//    return 0;
//}

#ifndef __SIMPLECUFFT_CLASS_H__
#define __SIMPLECUFFT_CLASS_H__

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <valarray>
#include <time.h>

#include <complex>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuComplex.h>
#include "device_launch_parameters.h"
#include "../common/book.h"
// #include "../common/cpu_bitmap.h"
#include "../common/gpu_anim.h"

#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED 0.25f
#define DIM 1024
#define BLOCKSIZE 8

const float PI = 3.141592653589793238460;

typedef std::complex<double> stdComplex;


/*****************/
/* CUDA MEMCHECK */
/*****************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %dn", cudaGetErrorString(code), file, line);
        if (abort) { getchar(); exit(code); }
    }
}
/*******************/
/* Host functions  */
/*******************/

// Round odd to even.
int iDivUp(int hostPtr, int b) { return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b); }

// Globals needed by the update routine
struct DataBlock {
    float          *dev_inSrc;
    float          *dev_outSrc;

    cudaEvent_t     start, stop;
    float           totalTime;
    float           frames;
};

/*************************************/
/* Device functions for complex data */
/*************************************/
__device__ double arg(double real, double imag)
{
    return atan(imag / real);
}
__device__ double abs_d(double real, double imag)
{
    return sqrt(real*real+imag*imag);
}
__device__ double real_d(double abs, double phase)
{
    return (abs*cos(phase));
}
__device__ double imag_d(double abs, double phase)
{
    return (abs*sin(phase)); //      !
}
__device__ double sqrt_d(double arg)
{
    return sqrt(arg);
}

/**********************************************/
/* Device functions for operation with bitmap */
/**********************************************/



/****************************************/
/* Kernel 2D functions for complex data */
/****************************************/
__global__ void MyKernel (cufftDoubleComplex *d_target, cufftDoubleComplex *data, size_t d_pitch, int nx, bool flag)
{
    int ind_y = threadIdx.y + blockDim.y * blockIdx.y;
    int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    int idx = ind_y + ind_x * nx;
    double Norm = (DIM);

    {
        data[idx].x = abs_d(data[idx].x, data[idx].y);
        data[idx].y = arg(data[idx].x, data[idx].y);      // Преобразовали к амплитудно-фазовому
        data[idx].y /= (Norm);                            // Нормируем на размерность матрицы
        data[idx].x = d_target[idx].x;                    // Заменили амплитуду
        data[idx].x = real_d(data[idx].x, data[idx].y);   // Преобразуем к a + ib
        data[idx].y = imag_d(data[idx].x, data[idx].y);
    }
}

__global__ void complex_to_double (double *optr, cufftDoubleComplex *d_in_complex)
{
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    optr[offset] = abs_d(d_in_complex[offset].x, d_in_complex[offset].y);
}

/**************************************************/
/* Kernel 2D functions for operations with bitmap */
/**************************************************/


class CuFFTFotonTrap {
public:

	CuFFTFotonTrap() {
		initCPUdata();
		initGPUdata();
		initBitMap ();
		bindTextures();
	}

	~CuFFTFotonTrap() {
		freeCPUData();
		freeGPUData();
	}

	void copyH2D() {

	}

	void copyD2H() {

	}

	// void copy

	void anim_gpu (uchar4* outputBitmap, DataBlock *d, int ticks) {
	    HANDLE_ERROR(cudaEventRecord(d->start, 0));
	    dim3 blocks(DIM/16, DIM/16);
	    dim3 threads(16, 16);

	    // since tex is global and bound, we have to use a flag to
	    // select which is in/out per iteration
	    volatile bool dstOut = true;
	    for (int i=0; i < 90; i++) {
	        float *in, *out;
	        if (dstOut) {
	            in  = d->dev_inSrc;
	            out = d->dev_outSrc;
	        } else {
	            out = d->dev_inSrc;
	            in  = d->dev_outSrc;
	        }
	        // copy_const_kernel<<<blocks,threads>>>( in );
	        // blend_kernel<<<blocks,threads>>>( out, dstOut );
	        dstOut = !dstOut;
	    }
	    // float_to_color<<<blocks,threads>>>( outputBitmap,
	    //                                    d->dev_inSrc );

	    HANDLE_ERROR( cudaEventRecord( d->stop, 0 ) );
	    HANDLE_ERROR( cudaEventSynchronize( d->stop ) );
	    float   elapsedTime;
	    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, d->start, d->stop ) );
	    d->totalTime += elapsedTime;
	    ++d->frames;
	    printf( "Average Time per frame:  %3.1f ms\n", d->totalTime/d->frames );
	}

	void anim_exit () {

	}

private:
	void initCPUdata () {
		test = new stdComplex[DIM * DIM];
		target = new stdComplex[DIM * DIM];

	    for (size_t i=0; i < DIM; ++i) {
	    	for (size_t j=0; j < DIM; ++j) {
	    		test[i * DIM + j] = stdComplex(PI * i / (i + j + 1), PI * j / (i + j + 1));
	    		target[i * DIM + j] = stdComplex(0, 0);
	    	}
	    }

	    memcpy( ISO, test, DIM * DIM * sizeof( stdComplex ) ); // Скопируем исходный массив чтобы сделать преобразование

	    // Создаём из массива масок и амплитуд начального распределения массив вида a + ib.
	    for (size_t i=0; i < DIM; i++)
	    {
	        for (size_t j = 0; j < DIM; j++)
	            ISO[i * DIM + j] = std::polar( real( ISO[i * DIM + j] ), imag( ISO[i * DIM + j] ) );
	    }
	}

	void initBitMap () {
		bitmap = GPUAnimBitmap( DIM, DIM, &bitmap_data );
		bitmap_data.totalTime = 0;
		bitmap_data.frames = 0;
	    HANDLE_ERROR( cudaEventCreate( &bitmap_data.start ) );
	    HANDLE_ERROR( cudaEventCreate( &bitmap_data.stop ) );

	    imageSize = bitmap.image_size();

	    // assume float == 4 chars in size (ie rgba)
	    HANDLE_ERROR( cudaMalloc( (void**)&bitmap_data.dev_inSrc,
	                              imageSize ) );
	    HANDLE_ERROR( cudaMalloc( (void**)&bitmap_data.dev_outSrc,
	                              imageSize ) );
	}

	void initGPUdata () {
	    unsigned nx_mem_size = sizeof(stdComplex) * DIM;         // Если делать 2D-копирование с разделением
	    size_t d_pitch, h_pitch = nx_mem_size ;                 // Если делать 2D-копирование с разделением
	    size_t size = DIM * DIM * sizeof(cufftDoubleComplex); // Копирование без разделения
	    cudaMalloc( (void**)&dataInOutCuda, size );
	    cudaMalloc( (void**)&data, size );
	    cudaMalloc( (void**)&d_target, size );
	    cudaMalloc( (void**)&d_basic, size );
	    cudaMemcpy( dataInOutCuda, ISO, size, cudaMemcpyHostToDevice );   // Копируем подбираемое распределение на карту
	    cudaMemcpy( d_target, target, size, cudaMemcpyHostToDevice );     // Копируем целевое распределение  на карту
	    cudaMemcpy( d_basic, test, size, cudaMemcpyHostToDevice );        // Копируем  входное распределение на карту
	    ///----------Device code-------------
	    Grd = dim3( iDivUp( DIM, BLOCKSIZE ), iDivUp( DIM, BLOCKSIZE ) );
	    Blk = dim3( BLOCKSIZE, BLOCKSIZE );

	    cufftPlan2d( &planFFT, DIM, DIM, CUFFT_Z2Z );
	    cufftPlan2d( &planIFFT, DIM, DIM, CUFFT_Z2Z );
	}

	void bindTextures() {
	    HANDLE_ERROR( cudaBindTexture( NULL, texIn,
	    							   bitmap_data.dev_inSrc,
	                                   imageSize ) );

	    HANDLE_ERROR( cudaBindTexture( NULL, texOut,
	    		                       bitmap_data.dev_outSrc,
	                                   imageSize ) );
	}

	void freeGPUData() {
	    cufftDestroy(planFFT);
	    cufftDestroy(planIFFT);

	    cudaFree(dataInOutCuda);
	    cudaFree(data);
	    cudaFree(d_target);
	    cudaFree(d_basic);

	    HANDLE_ERROR( cudaUnbindTexture( texIn ) );
	    HANDLE_ERROR( cudaUnbindTexture( texOut ) );
	    HANDLE_ERROR( cudaFree( bitmap_data.dev_inSrc ) );
	    HANDLE_ERROR( cudaFree( bitmap_data.dev_outSrc ) );

	    HANDLE_ERROR( cudaEventDestroy( bitmap_data.start ) );
	    HANDLE_ERROR( cudaEventDestroy( bitmap_data.stop ) );
	}

	void freeCPUData() {
		delete[] test;
		delete[] target;
	}

private:
	dim3 Grd, Blk;

	//---These exist on the GPU side.---
	// Textures.
	texture<float> texIn;
	texture<float> texOut;
	// Data buffers.
	cufftDoubleComplex *dataInOutCuda;
	cufftDoubleComplex *data;
	cufftDoubleComplex *d_target;
	cufftDoubleComplex *d_basic;
	// CuFFT plans.
	cufftHandle planFFT, planIFFT;
	//---CPU data.---
	// Temporary buffers
	stdComplex *target, *test;
	// Отладочный массив  // Буфер хранения образов после преобразования Фурье
	stdComplex *ISO;
	// BitMap's data
    DataBlock bitmap_data;
    GPUAnimBitmap bitmap;
    unsigned int imageSize;
};

#endif


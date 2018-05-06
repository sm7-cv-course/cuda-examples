#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex>
#include <math.h>
#include <iostream>
#include <valarray>
#include <time.h>
#include <stdlib.h>
// includes, project
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
#define SPEED   0.25f

// these exist on the GPU side
texture<float>  texConstSrc;
texture<float>  texIn;
texture<float>  texOut;

cufftDoubleComplex *dataInOutCuda;
cufftDoubleComplex *data;
cufftDoubleComplex *d_target;
cufftDoubleComplex *d_basic;
cufftHandle planFFT, planIFFT;


#define BLOCKSIZE 8
#define Dim 8
const float PI = 3.141592653589793238460;
typedef std::complex<double> stdComplex;

/*****************/
/* CUDA MEMCHECK */
/*****************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// globals needed by the update routine
struct DataBlock {
    float           *dev_inSrc;
    float           *dev_outSrc;
    float           *dev_constSrc;

    cudaEvent_t     start, stop;
    float           totalTime;
    float           frames;
};

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
int iDivUp(int hostPtr, int b) { return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b); }

/*void
complex_to_bitmap(unsigned char *bm, stdComplex data_in[Dim], unsigned int w, unsigned int h) {
	unsigned int count = 0;

	for(unsigned int i=0; i < h; ++i) {
	    for(unsigned int j=0; j < w; ++j) {
	    	bm[count] = sqrt(std::real(data_in[i][j]) * std::real(data_in[i][j]) + std::imag(data_in) * std::imag(data_in));
	    	++count;
	    }
	}
}*/
/*******************/
/* Device functions */
/*******************/
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
/*******************/
/* Kernel 2D FUNCTION */
/*******************/
__global__ void MyKernel(cufftDoubleComplex *d_target, cufftDoubleComplex *data, size_t d_pitch, int nx, bool flag)
{
    int ind_y = threadIdx.y + blockDim.y * blockIdx.y;
    int ind_x = threadIdx.x + blockDim.x * blockIdx.x;
    int idx = ind_y + ind_x * nx;
    double Norm = (Dim);

    {
        data[idx].x = abs_d(data[idx].x, data[idx].y);
        data[idx].y = arg(data[idx].x, data[idx].y);      // Преобразовали к амплитудно-фазовому
        data[idx].y /= (Norm);                            // Нормируем на размерность матрицы
        data[idx].x = d_target[idx].x;                    // Заменили амплитуду
        data[idx].x = real_d(data[idx].x, data[idx].y);   // Преобразуем к a + ib
        data[idx].y = imag_d(data[idx].x, data[idx].y);
    }
}

__global__ void complex_to_double( double *optr, cufftDoubleComplex *d_in_complex ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    optr[offset] = abs_d(d_in_complex.x, d_in_complex.y);
}

// NOTE - texOffsetConstSrc could either be passed as a
// parameter to this function, or passed in __constant__ memory
// if we declared it as a global above, it would be
// a parameter here:
// __global__ void copy_const_kernel( float *iptr,
//                                    size_t texOffset )
__global__ void copy_const_kernel( float *iptr ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float c = tex1Dfetch(texConstSrc,offset);
    if (c != 0)
        iptr[offset] = c;
}

void
bitmap_init(CPUBitmap * bitmap) {
	stdComplex target[Dim][Dim];
    stdComplex test[Dim][Dim];

    for(unsigned int i = 0; i < Dim; ++i) {
    	for(unsigned int j = 0; j < Dim; ++j) {
    		test[i][j] = stdComplex(PI * i / (i + j + 1), PI * j / (i + j + 1));
    		target[i][j] = stdComplex(0, 0);
    	}
    }

    stdComplex ISO[Dim][Dim];  // Отладочный массив  // Буфер хранения образов после преобразования Фурье
    memcpy(ISO, test, Dim * Dim * sizeof(stdComplex)); // Скопируем исходный массив, чтобы сделать преобразование

    // Создаём из массива масок и амплитуд начального распределения массив вида a + ib.
    for (size_t i = 0; i < Dim; i++)
    {
        for (size_t j = 0; j < Dim; j++)
            ISO[i][j] = std::polar(real((ISO[i][j])), imag(ISO[i][j]));
    }

    // Выводим полученное.
    std::cout << " Initial matrix" << std::endl;
    for (size_t i = 0; i < Dim; i++)
    {
        for (size_t j = 0; j < Dim; j++)
            std::cout << ISO[i][j];
        std::cout << std::endl;
    }

    unsigned nx_mem_size = sizeof(stdComplex) * Dim;         // Если делать 2D-копирование с разделением
    size_t d_pitch, h_pitch=nx_mem_size ;                 // Если делать 2D-копирование с разделением
    size_t size = Dim * Dim * sizeof(cufftDoubleComplex); // Копирование без разделения
    /*cufftDoubleComplex *dataInOutCuda;
    cufftDoubleComplex *data;
    cufftDoubleComplex *d_target;
    cufftDoubleComplex *d_basic;*/
    //gpuErrchk(cudaMallocPitch((void**)&dataInOutCuda, &d_pitch, nx_mem_size, Dim));
    //gpuErrchk(cudaMallocPitch((void**) &data, &d_pitch, nx_mem_size, Dim));
    //gpuErrchk(cudaMallocPitch((void**)&d_target, &d_pitch, nx_mem_size, Dim));
    //gpuErrchk(cudaMallocPitch((void**)&d_basic, &d_pitch, nx_mem_size, Dim));
    //gpuErrchk(cudaMemcpy2D(d_target, d_pitch, target, h_pitch, nx_mem_size, Dim, cudaMemcpyHostToDevice));
    //gpuErrchk(cudaMemcpy2D(dataInOutCuda,d_pitch,  ISO, h_pitch, nx_mem_size, Dim, cudaMemcpyHostToDevice));
    //gpuErrchk(cudaMemcpy2D(d_basic, d_pitch, test, h_pitch, nx_mem_size, Dim, cudaMemcpyHostToDevice));
    cudaMalloc((void**)&dataInOutCuda, size);
    cudaMalloc((void**)&data, size);
    cudaMalloc((void**)&d_target, size);
    cudaMalloc((void**)&d_basic, size);
    cudaMemcpy(dataInOutCuda, ISO, size, cudaMemcpyHostToDevice);   // Копируем подбираемое распределение на карту
    cudaMemcpy(d_target, target, size, cudaMemcpyHostToDevice);     // Копируем целевое распределение  на карту
    cudaMemcpy(d_basic, test, size, cudaMemcpyHostToDevice);        // Копируем  входное распределение на карту
    // std::cout<<std::endl << "host : " << h_pitch << " device : " << d_pitch << std::endl; /// Отладка
    ///----------Device code

    Dim3 Grd(iDivUp(Dim, BLOCKSIZE), iDivUp(Dim, BLOCKSIZE));
    Dim3 Blk(BLOCKSIZE, BLOCKSIZE);
    // Выделяем поле прямых и обратных преобразований.
    // cufftHandle planFFT, planIFFT;
    cufftPlan2d(&planFFT, Dim, Dim, CUFFT_Z2Z);
    cufftPlan2d(&planIFFT, Dim, Dim, CUFFT_Z2Z);

    for(size_t i = 0; i < 100; i++)
    {
        cufftExecZ2Z(planFFT, dataInOutCuda, data, CUFFT_FORWARD); //  Прямое преобразование Фурье для входящего распределения
        cudaThreadSynchronize();
        MyKernel << <Grd, Blk >> > (d_target, data, d_pitch, Dim, true); // Ядро, заменяющее полученную после прямого преобразования матрицу матрицей искомого
        cudaThreadSynchronize();
        cufftExecZ2Z(planIFFT, data, dataInOutCuda, CUFFT_INVERSE); // Обратное преобразование Фурье с заменёнными амплитудами
        cudaThreadSynchronize();
        MyKernel << <Grd, Blk >> > (d_basic, dataInOutCuda, d_pitch, Dim, false); // Ядро, заменяющее полученную после обратного преобразования матрицу
                                                                                    // входным
        cudaThreadSynchronize();
    }
    stdComplex Tester[Dim][Dim];
    cudaMemcpy(Tester, dataInOutCuda, size, cudaMemcpyDeviceToHost); // Читаем с GPU
    ///   --- end device code

    // Выводим полученное
    std::cout << std::endl << "Here's Your successes " << std::endl << std::endl;
    for(size_t i = 0; i < Dim; i++)
    {
        for(size_t j = 0; j < Dim; j++)
            std::cout << Tester[i][j];
        std::cout << std::endl;
    }

    unsigned int count = 0;
	for(unsigned int i=0; i < Dim; ++i) {
	    for(unsigned int j=0; j < Dim; ++j) {
	    	bitmap->get_ptr()[count] = sqrt(std::real(Tester[i][j]) * std::real(Tester[i][j]) + std::imag(Tester[i][j]) * std::imag(Tester[i][j])) * 255;
	    	++count;
	    }
	}

    cufftDestroy(planFFT);
    cufftDestroy(planIFFT);

    cudaFree(dataInOutCuda);
    cudaFree(data);
}

void
anim_gpu( uchar4* outputBitmap, DataBlock *d, int ticks ) {
    HANDLE_ERROR( cudaEventRecord( d->start, 0 ) );
    dim3    blocks(DIM/16,DIM/16);
    dim3    threads(16,16);

    // since tex is global and bound, we have to use a flag to
    // select which is in/out per iteration
    volatile bool dstOut = true;
    for (int i=0; i<90; i++) {
        float   *in, *out;
        if (dstOut) {
            in  = d->dev_inSrc;
            out = d->dev_outSrc;
        } else {
            out = d->dev_inSrc;
            in  = d->dev_outSrc;
        }
        copy_const_kernel<<<blocks,threads>>>( in );
        blend_kernel<<<blocks,threads>>>( out, dstOut );
        dstOut = !dstOut;
    }
    float_to_color<<<blocks,threads>>>( outputBitmap,
                                        d->dev_inSrc );

    HANDLE_ERROR( cudaEventRecord( d->stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( d->stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                        d->start, d->stop ) );
    d->totalTime += elapsedTime;
    ++d->frames;
    printf( "Average Time per frame:  %3.1f ms\n",
            d->totalTime/d->frames  );
}

// clean up memory allocated on the GPU
void
anim_exit( DataBlock *d ) {
    HANDLE_ERROR( cudaUnbindTexture( texIn ) );
    HANDLE_ERROR( cudaUnbindTexture( texOut ) );

    HANDLE_ERROR( cudaFree( d->dev_inSrc ) );
    HANDLE_ERROR( cudaFree( d->dev_outSrc ) );

    HANDLE_ERROR( cudaEventDestroy( d->start ) );
    HANDLE_ERROR( cudaEventDestroy( d->stop ) );
}

int
main() {
	DataBlock   data;
	GPUAnimBitmap bitmap( Dim, Dim, &data );
	data.totalTime = 0;
	data.frames = 0;
	HANDLE_ERROR( cudaEventCreate( &data.start ) );
	HANDLE_ERROR( cudaEventCreate( &data.stop ) );

	int imageSize = bitmap.image_size();

    // assume float == 4 chars in size (ie rgba)
    HANDLE_ERROR( cudaMalloc( (void**)&data.dev_inSrc,
                              imageSize ) );
    HANDLE_ERROR( cudaMalloc( (void**)&data.dev_outSrc,
                              imageSize ) );

    HANDLE_ERROR( cudaBindTexture( NULL, texIn,
                                   data.dev_inSrc,
                                   imageSize ) );

    HANDLE_ERROR( cudaBindTexture( NULL, texOut,
                                   data.dev_outSrc,
                                   imageSize ) );

    float *temp = (float*)malloc( imageSize );


    HANDLE_ERROR( cudaMemcpy( data.dev_inSrc, temp,
                              imageSize,
                              cudaMemcpyHostToDevice ) );
    free( temp );

    bitmap.anim_and_exit( (void (*)(uchar4*,void*,int))anim_gpu,
                           (void (*)(void*))anim_exit );
}

int main() {
	// Device bitmap.
	unsigned char *dev_bitmap;

	// CPU bitmap.
	CPUBitmap bitmap(Dim, Dim, dev_bitmap);

	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap,
	                          bitmap.image_size()));

	// Form bitmap.
	bitmap_init(&bitmap);


	// Show bitmap.
	bitmap.display_and_exit();

    getchar();

    return 0;
}


/*int main()
{
    const stdComplex test[Dim][Dim] =
    {
        stdComplex(1.0, PI / 13), stdComplex(1.0,PI / 11), stdComplex(1.0,PI / 4), stdComplex(1.0,PI / 6), stdComplex(1.0, PI), stdComplex(1.0,5 * PI / 4), stdComplex(1.0,7 * PI / 4), stdComplex(1.0,6 * PI / 4),
        stdComplex(1.0,20 * PI / 17), stdComplex(1.0,11 * PI / 5), stdComplex(1.0,PI / 4), stdComplex(1.0,PI / 6), stdComplex(1.0, PI), stdComplex(1.0,5 * PI / 4), stdComplex(1.0,7 * PI / 4), stdComplex(1.0,6 * PI / 4),
        stdComplex(1.0,PI / 2), stdComplex(1.0,PI / 3), stdComplex(1.0,PI / 4), stdComplex(1.0,PI / 6), stdComplex(1.0, PI), stdComplex(1.0,5 * PI / 4), stdComplex(1.0,7 * PI / 4), stdComplex(1.0,6 * PI / 4),
        stdComplex(1.0,PI / 2), stdComplex(1.0,PI / 3), stdComplex(1.0,PI / 4), stdComplex(1.0,PI / 6), stdComplex(1.0, PI), stdComplex(1.0,5 * PI / 4), stdComplex(1.0,7 * PI / 4), stdComplex(1.0,6 * PI / 4),
        stdComplex(1.0,PI / 2), stdComplex(1.0,PI / 3), stdComplex(1.0,PI / 4), stdComplex(1.0,PI / 6), stdComplex(1.0, PI), stdComplex(1.0,5 * PI / 4), stdComplex(1.0,7 * PI / 4), stdComplex(1.0,6 * PI / 4),
        stdComplex(1.0,PI / 2), stdComplex(1.0,PI / 3), stdComplex(1.0,PI / 4), stdComplex(1.0,PI / 6), stdComplex(1.0, PI), stdComplex(1.0,5 * PI / 4), stdComplex(1.0,7 * PI / 4), stdComplex(1.0,6 * PI / 4),
        stdComplex(1.0,PI / 2), stdComplex(1.0,PI / 3), stdComplex(1.0,PI / 4), stdComplex(1.0,PI / 6), stdComplex(1.0, PI), stdComplex(1.0,5 * PI / 4), stdComplex(1.0,7 * PI / 4), stdComplex(1.0,6 * PI / 4),
        stdComplex(1.0,PI / 2), stdComplex(1.0,PI / 3), stdComplex(1.0,PI / 4), stdComplex(1.0,PI / 6), stdComplex(1.0, PI), stdComplex(1.0,5 * PI / 4), stdComplex(1.0,7 * PI / 4), stdComplex(1.0,6 * PI / 4)
    };

    stdComplex target[Dim][Dim] =
    {
        stdComplex(0,0), stdComplex(0,0), stdComplex(0,0), stdComplex(0,0), stdComplex(0,0), stdComplex(0,0), stdComplex(0,0), stdComplex(0,0),
        stdComplex(0,0), stdComplex(0,0), stdComplex(0,0), stdComplex(0,0), stdComplex(0,0), stdComplex(0,0), stdComplex(0,0), stdComplex(0,0),
        stdComplex(0,0), stdComplex(0,0), stdComplex(0.4,0), stdComplex(0.4,0), stdComplex(0.4,0), stdComplex(0,0), stdComplex(0,0), stdComplex(0,0),
        stdComplex(0,0), stdComplex(0,0), stdComplex(0.4,0), stdComplex(1,0), stdComplex(1,0), stdComplex(0.4,0), stdComplex(0,0), stdComplex(0,0),
        stdComplex(0,0), stdComplex(0,0), stdComplex(0.4,0), stdComplex(1,0), stdComplex(1,0), stdComplex(0.4,0), stdComplex(0,0), stdComplex(0,0),
        stdComplex(0,0), stdComplex(0,0), stdComplex(0,0), stdComplex(0.4,0), stdComplex(0.4,0), stdComplex(0.4,0), stdComplex(0,0), stdComplex(0,0),
        stdComplex(0,0), stdComplex(0,0), stdComplex(0,0), stdComplex(0,0), stdComplex(0,0), stdComplex(0,0), stdComplex(0,0), stdComplex(0,0),
        stdComplex(0,0), stdComplex(0,0), stdComplex(0,0), stdComplex(0,0), stdComplex(0,0), stdComplex(0,0), stdComplex(0,0), stdComplex(0,0)
    };

    stdComplex ISO[Dim][Dim];  // Отладочный массив  // Буфер хранения образов после преобразования Фурье
    memcpy(ISO, test, Dim * Dim * sizeof(stdComplex)); // Скопируем исходный массив чтобы сделать преобразование

    // Создаём из массива масок и амплитуд начального распределения массив вида a + ib.
    for (size_t i = 0; i < Dim; i++)
    {
        for (size_t j = 0; j < Dim; j++)
            ISO[i][j] = std::polar(real((ISO[i][j])), imag(ISO[i][j]));

    }

    // Выводим полученное.
    std::cout << " Initial matrix" << std::endl;
    for (size_t i = 0; i < Dim; i++)
    {
        for (size_t j = 0; j < Dim; j++)
            std::cout << ISO[i][j];
        std::cout << std::endl;
    }

    unsigned nx_mem_size = sizeof(stdComplex) * Dim;    // Если делать 2D-копирование с разделением
    size_t d_pitch, h_pitch=nx_mem_size ;           // Если делать 2D-копирование с разделением
    size_t size = Dim * Dim * sizeof(cufftDoubleComplex); // Копирование без разделения
    cufftDoubleComplex *dataInOutCuda;
    cufftDoubleComplex *data;
    cufftDoubleComplex * d_target;
    cufftDoubleComplex * d_basic;
    //gpuErrchk(cudaMallocPitch((void**)&dataInOutCuda, &d_pitch, nx_mem_size, Dim));
    //gpuErrchk(cudaMallocPitch ((void**) &data, &d_pitch, nx_mem_size, Dim));
    //gpuErrchk(cudaMallocPitch((void**)&d_target, &d_pitch, nx_mem_size, Dim));
    //gpuErrchk(cudaMallocPitch((void**)&d_basic, &d_pitch, nx_mem_size, Dim));
    //gpuErrchk(cudaMemcpy2D(d_target, d_pitch, target, h_pitch, nx_mem_size, Dim, cudaMemcpyHostToDevice));
    //gpuErrchk(cudaMemcpy2D(dataInOutCuda,d_pitch,  ISO, h_pitch, nx_mem_size, Dim, cudaMemcpyHostToDevice));
    //gpuErrchk(cudaMemcpy2D(d_basic, d_pitch, test, h_pitch, nx_mem_size, Dim, cudaMemcpyHostToDevice));
    cudaMalloc((void**)&dataInOutCuda, size);
    cudaMalloc((void**)&data, size);
    cudaMalloc((void**)&d_target, size);
    cudaMalloc((void**)&d_basic, size);
    cudaMemcpy(dataInOutCuda, ISO, size, cudaMemcpyHostToDevice);   // Копируем подбираемое распределение на карту
    cudaMemcpy(d_target, target, size, cudaMemcpyHostToDevice);     // Копируем целевое распределение  на карту
    cudaMemcpy(d_basic, test, size, cudaMemcpyHostToDevice);        // Копируем  входное распределение на карту
    // std::cout<<std::endl << "host : " << h_pitch << " device : " << d_pitch << std::endl; /// Отладка
    ///----------Device code

    Dim3 Grd(iDivUp(Dim, BLOCKSIZE), iDivUp(Dim, BLOCKSIZE));
    Dim3 Blk(BLOCKSIZE, BLOCKSIZE);
    // Выделяем поле прямых и обратных преобразований.
    cufftHandle planFFT, planIFFT;
    cufftPlan2d(&planFFT, Dim , Dim  , CUFFT_Z2Z);
    cufftPlan2d(&planIFFT, Dim , Dim , CUFFT_Z2Z);

    for (size_t i = 0; i < 100; i++)
    {
        cufftExecZ2Z(planFFT, dataInOutCuda, data, CUFFT_FORWARD); //  Прямое преобразование Фурье для входящего распределения
        cudaThreadSynchronize();
        MyKernel << <Grd, Blk >> > (d_target, data, d_pitch, Dim, true); // Ядро, заменяющее полученную после прямого преобразования матрицу матрицей искомого
        cudaThreadSynchronize();
        cufftExecZ2Z(planIFFT, data, dataInOutCuda, CUFFT_INVERSE); // Обратное преобразование Фурье с заменёнными амплитудами
        cudaThreadSynchronize();
        MyKernel << <Grd, Blk >> > (d_basic, dataInOutCuda, d_pitch, Dim, false); // Ядро, заменяющее полученную после обратного преобразования матрицу
                                                                                    // входным
        cudaThreadSynchronize();
    }
    stdComplex Tester[Dim][Dim];
    cudaMemcpy(Tester, dataInOutCuda, size, cudaMemcpyDeviceToHost); // Читаем с GPU
    ///   --- end device code

    // Выводим полученное
    std::cout << std::endl << "Here's Your successes " << std::endl << std::endl;
    for (size_t i = 0; i < Dim; i++)
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
}*/

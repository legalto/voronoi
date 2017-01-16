#include <stdio.h>
#include "voronoi.h"
#include <vector>

__global__ void voronoi_kernel(unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep, COORDS_T* sites){

	//Get index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	float distance = 0.0;
	auto closest_site = 0;
	auto closest = 1.0e30f;

	// Make sure threads don't access memory outside of the image
	if((xIndex<width) && (yIndex<height))
	{
		//Location of colored pixel in input
		const int color_tid = yIndex * colorWidthStep + (3 * xIndex);

		#pragma unroll
		for (int index = 0; index < sites->next; ++index) {

	    		 distance = sqrtf( (xIndex - sites->x[index])*(xIndex - sites->x[index]) +
	    						  (yIndex - sites->y[index])*(yIndex - sites->y[index]) );

	    		 if (distance < closest) {
	    			closest_site = index;
	    			closest = distance;
	    		}
	    }

		// Determine pixel index of closest site
		const int site_result = sites->y[closest_site]*colorWidthStep + (3*sites->x[closest_site]);

		// Copy site pixel value into output pixel
		output[color_tid]		= input[site_result];
		output[color_tid + 1]	= input[site_result + 1];
		output[color_tid + 2]	= input[site_result + 2];

	}
}

__global__ void voronoi_kernel_shared_mem(unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep, COORDS_T* sites){

	__shared__ COORDS_T *sites_shared;
	sites_shared = sites;
	__syncthreads();


	//Get index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	float distance = 0.0;
	auto closest_site = 0;
	auto closest = 1.0e30f;

	// Make sure threads don't access memory outside of the image
	if((xIndex<width) && (yIndex<height))
	{
		//Location of colored pixel in input
		const int color_tid = yIndex * colorWidthStep + (3 * xIndex);

		#pragma unroll
		for (int index = 0; index < sites_shared->next; ++index) {

	    		 distance = sqrtf( (xIndex - sites_shared->x[index])*(xIndex - sites_shared->x[index]) +
	    						  (yIndex - sites_shared->y[index])*(yIndex - sites_shared->y[index]) );

	    		 if (distance < closest) {
	    			closest_site = index;
	    			closest = distance;
	    		}
	    }

		// Determine pixel index of closest site
		const int site_result = sites_shared->y[closest_site]*colorWidthStep + (3*sites_shared->x[closest_site]);

		// Copy site pixel value into output pixel
		output[color_tid]		= input[site_result];
		output[color_tid + 1]	= input[site_result + 1];
		output[color_tid + 2]	= input[site_result + 2];

	}
}

Voronoi::Voronoi(const cv::Mat& input) {

	const int bytes = input.step * input.rows;

	// Allocate memory on the gpu
	cudaMalloc<COORDS_T>(&d_sites, sizeof(COORDS_T));
	cudaMalloc<unsigned char>(&d_input, bytes);
	cudaMalloc<unsigned char>(&d_output, bytes);

	// Copy host data to device
	cudaMemcpy(d_input, input.ptr(), bytes, cudaMemcpyHostToDevice);
}

Voronoi::~Voronoi() {
	clean();
}


void Voronoi::voronoi_gpu(cv::Mat& output){
	const int bytes = output.step * output.rows;

	// Set block size (256 threads per block)
	const dim3 block(16,16);

	const dim3 grid((output.cols + block.x - 1)/block.x, (output.rows + block.y -1)/block.y);

	voronoi_kernel<<<grid,block>>>(d_input, d_output, output.cols, output.rows, output.step, d_sites);
	//voronoi_kernel_shared_mem<<<grid,block>>>(d_input, d_output, output.cols, output.rows, output.step, d_sites);
	cudaDeviceSynchronize();

	cudaMemcpy(output.ptr(), d_output, bytes, cudaMemcpyDeviceToHost);

}

void Voronoi::update(COORDS_T *sites){
	// Copy new site data to the device
	cudaMemcpy(d_sites, sites, sizeof(COORDS_T), cudaMemcpyHostToDevice);
}

void Voronoi::clean(){
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_sites);
}


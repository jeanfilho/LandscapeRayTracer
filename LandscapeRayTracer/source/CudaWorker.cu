#pragma once

#include "CudaWorker.h"

__device__ glm::fvec3* d_pixel_array;

__global__
void castRay()
{

}

void CudaWorker::par_castRay(glm::fvec3* pixel_array, const Camera* cam, const Grid<Grid<PointData*>*>* grid, int window_height, int window_width)
{
	if (cudaMalloc(&d_pixel_array, window_height*window_width * 3 * sizeof(float)) != cudaSuccess)
	{
		cout << "Error allocating memory in device" << endl;
	}
	cudaMemcpy(d_pixel_array, pixel_array, window_height*window_width * 3 * sizeof(float), cudaMemcpyHostToDevice);

	castRay << <4, 4 >> >();

	cudaFree(d_pixel_array);
}
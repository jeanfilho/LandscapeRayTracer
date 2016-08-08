#pragma once

#include "CudaWorker.h"

using namespace std;

__device__ glm::vec3* d_pixel_array;
__device__ Grid<Grid<PointData*>*> *d_grid;
thrust::device_vector<glm::vec3> d_points;

__device__ glm::vec3* d_cam_pos;
__device__ glm::vec3* d_cam_dir;
__device__ int *d_window_height, *d_window_width;

__global__
void mainLoop()
{

}

__global__
void allocateGrids()
{

}

/*
	Load points from HDD, store them in points array and transfer them to the GPU
*/
cudaError_t CudaWorker::loadPoints(float *max_height, float *min_height, thrust::host_vector<PointData> *points)
{
	cudaError_t result;

	d_points = thrust::device_vector<glm::vec3>();

	std::ifstream file("../Data/data");
	std::string line, data;
	float x, y, z;
	int count = 0;

	while (getline(file, line) && !line.empty())
	{
		size_t start = 0, end = line.find(" ");
		data = line.substr(start, end - start);
		x = (stof(data));

		start = end + 1;
		end = line.find(" ", start + 1);
		data = line.substr(start, end - start);
		z = (stof(data));

		start = end + 1;
		data = line.substr(start, end - start);
		y = (stof(data));

		points->push_back(PointData(glm::vec3(x,y,z), glm::vec3(0,1,1)));
		d_points.push_back(glm::vec3(x, y, z));

		if (y > *max_height) *max_height = z;
		if (y < *min_height) *min_height = z;
	}
	file.close();

	result = cudaMalloc(&d_grid, sizeof(Grid<Grid<PointData*>*>));
	if (result != cudaSuccess)
	{
		cout << "Error allocating grid on device: cudaError " << result;
		return result;
	}

	allocateGrids << <100, 1 >> >();


	return result;
}

/*
	Start the main loop on the GPU
*/
cudaError_t CudaWorker::startRoutine(int window_height, int window_witdh)
{
	cudaError_t result = cudaSuccess;

	return result;
}

/*
	Called when exiting the main loop on the GPU
*/
cudaError_t CudaWorker::exitRoutine()
{
	cudaError_t result = cudaSuccess;

	return result;
}
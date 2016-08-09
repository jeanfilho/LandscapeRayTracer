#pragma once

#include "CudaWorker.h"

using namespace std;

__device__ int d_window_height, d_window_width;
__device__ bool continueLoop = true;
__device__ glm::vec3 *d_pixel_array;
__device__ Grid<Grid<bool>*> *d_grid;
__device__ Camera *d_cam;

__global__
void mainLoop()
{
	while (continueLoop)
	{

	}

	free(d_cam);
	free(d_grid);
	free(d_pixel_array);
}

__global__
void setUp(PointData* parray, int psize, int wh, int ww)
{
	d_window_height = wh;
	d_window_width = ww;

	d_grid = new Grid<Grid<bool>*>(glm::vec3(0, 0, 0), 100, 10000, NULL);
	float x, y, z;
	int	coarse_x, coarse_y, coarse_z,
		fine_x, fine_y, fine_z;
	Grid<bool>* subgrid;


	for (int i = 0; i < psize; i++)
	{
		x = parray[i].position.x;
		y = parray[i].position.x;
		z = parray[i].position.x;

		coarse_x = int(x / (*d_grid).cell_size);
		coarse_y = int(y / (*d_grid).cell_size);
		coarse_z = int(z / (*d_grid).cell_size);

		fine_x = int(x - coarse_x * (*d_grid).cell_size);
		fine_y = int(y - coarse_y * (*d_grid).cell_size);
		fine_z = int(z - coarse_z * (*d_grid).cell_size);

		if ((*d_grid)(coarse_x, coarse_y, coarse_z) == NULL)
			(*d_grid)(coarse_x, coarse_y, coarse_z) = new Grid<bool>(glm::vec3(coarse_x, coarse_y, coarse_z) * (*d_grid).cell_size, 100, (*d_grid).cell_size, NULL);

		subgrid = (*d_grid)(coarse_x, coarse_y, coarse_z);

		if ((*subgrid)(fine_x, fine_y, fine_z) == NULL)
			(*subgrid)(fine_x, fine_y, fine_z) = true;
	}
}

/*
	Load points from HDD, store them in points array and transfer them to the GPU
*/
cudaError_t CudaWorker::loadPoints(float *max_height, float *min_height, int window_height, int window_width, Camera cam)
{
	cudaError_t result = cudaSuccess;
	
	thrust::device_vector<PointData> d_points;
	ifstream file("../Data/data");
	string line, data;
	float x, y, z;

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

		d_points.push_back(PointData(glm::vec3(x, y, z), glm::vec3(0, 1, 1)));

		if (y > *max_height) *max_height = z;
		if (y < *min_height) *min_height = z;
	}
	file.close();

	setUp << <1, 1, 0 >> >(thrust::raw_pointer_cast(d_points.data()), d_points.size(), window_height, window_width);
	cudaDeviceSynchronize();

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
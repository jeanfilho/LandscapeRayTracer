#include "CudaWorker.h"

__device__ Grid<Grid<bool>*> *d_grid = NULL;
__device__ int d_wh, d_ww;
Camera *d_cam;
glm::vec3 *d_pixel_array;
thrust::device_vector<PointData> d_points;

__global__ void d_setUp(PointData* d_point_array, int size, int wh, int ww)
{
	float x, y, z;
	int	coarse_x, coarse_y, coarse_z,
		fine_x, fine_y, fine_z;

	d_ww = ww;
	d_wh = wh;

	d_grid = new Grid<Grid<bool>*>(glm::vec3(0, 0, 0), 1000, 10000, 0);
	Grid<bool>* subgrid;
	

	for (int i = 0; i < size; i++)
	{
		x = d_point_array[i].position.x;
		y = d_point_array[i].position.x;
		z = d_point_array[i].position.x;

		coarse_x = int(x / (*d_grid).cell_size);
		coarse_y = int(y / (*d_grid).cell_size);
		coarse_z = int(z / (*d_grid).cell_size);

		fine_x = int(x - coarse_x * (*d_grid).cell_size);
		fine_y = int(y - coarse_y * (*d_grid).cell_size);
		fine_z = int(z - coarse_z * (*d_grid).cell_size);

		subgrid = (*d_grid)(coarse_x, coarse_y, coarse_z);

		if (subgrid == NULL)
			subgrid = new Grid<bool>(glm::vec3(0, 0, 0), 100, 100, true);
		else
			(*subgrid)(fine_x, fine_y, fine_z) = true;
	}
}

__global__ void d_castRays(glm::vec3 *d_pixel_array, Camera *d_cam)
{
	bool inGrid = true, isFinished = false;
	glm::vec3 ray_pos, ray_dir;
	glm::ivec3 grid_pos;
	Grid<bool>* subgrid;

	while (inGrid && !isFinished)
	{
		grid_pos = d_grid->castRay(ray_pos, ray_dir, &ray_pos);
		if (grid_pos.x < 0)
			inGrid = false;
		else
		{
			subgrid = (*d_grid)(grid_pos);
			grid_pos = subgrid->castRay(ray_pos, ray_dir, &ray_pos);
			if (grid_pos.x >= 0)
				isFinished = true;
		}
	}

	if (inGrid)
		d_pixel_array[blockIdx.y * d_ww + blockIdx.x] = glm::vec3(0, 1, 1);
	else
		d_pixel_array[blockIdx.y * d_ww + blockIdx.x] = glm::vec3(0, 0, 0);
}

/*
Load points from HDD, store them in points array and transfer them to the GPU
*/
void CudaWorker::setUp(glm::vec3 *h_pixel_array, Camera cam, int window_height, int window_width)
{
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
	}
	file.close();

	checkCudaErrors(cudaMalloc(&d_pixel_array, window_height * window_width * sizeof(glm::vec3)));
	checkCudaErrors(cudaMalloc(&d_cam, sizeof(Camera)));
	checkCudaErrors(cudaMemcpy(d_pixel_array, h_pixel_array, window_height * window_width * sizeof(glm::vec3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_cam, &cam, sizeof(Camera), cudaMemcpyHostToDevice));
	
	d_setUp << <1, 1 >> >(thrust::raw_pointer_cast(d_points.data()), d_points.size(), window_height, window_width);
}

void CudaWorker::updatePixelArray(glm::vec3 *h_pixel_array, int window_height, int window_width)
{
	dim3 numBlocks(window_height, window_width);
	d_castRays << <numBlocks, 1 >> >(d_pixel_array, d_cam);

	cudaDeviceSynchronize();

	checkCudaErrors(cudaMemcpy(h_pixel_array, d_pixel_array, window_height * window_width * sizeof(glm::vec3), cudaMemcpyDeviceToHost));
}

void CudaWorker::release()
{
	cudaFree(d_pixel_array);
	cudaFree(d_cam);
}
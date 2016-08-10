#pragma once
#define GLM_FORCE_CUDA
#include <utility>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include <thrust\swap.h>

#include <glm\glm.hpp>
#include <glm\trigonometric.hpp>

/*
This class describes a NxNxN Grid

Implementation: http://www.scratchapixel.com/lessons/advanced-rendering/introduction-acceleration-structure/grid
*/
template<class T>
class Grid
{
public:
	
	__host__ __device__ Grid<T>(glm::vec3 origin, int resolution, float size, T fill);
	__host__ __device__ ~Grid<T>();

	inline __host__ __device__ T & operator()(int x, int y, int z);
	inline __host__ __device__ T & operator()(glm::ivec3 idx);
	inline __host__ __device__ Grid<T> & operator=(const Grid<T> &obj);
	inline __host__ __device__ Grid<T> & operator=(Grid<T> obj);

	/*-------------------------------------------------------------------------
	Cast a ray through the grid
	
	Output:
	@intersection: the point of intersection between the ray and the cell grid face
	@Return:
		- coordinates of the intersection
		- glm::ivec3(-1,-1,-1) otherwise 
	-------------------------------------------------------------------------*/
	__host__ __device__ glm::ivec3 castRay(const glm::vec3 &ray_origin, const glm::vec3 &ray_direction, glm::vec3 *intersection);

	const int resolution;
	const float size, cell_size;
	glm::vec3 origin;

	T unitialized_value;

protected:

	/*-------------------------------------------------------------------------
	Output:
	@t: current distances to the next ray-cell intersection
	@d: distances between two cells in X/Y/Z axis along the ray 
	@return:
		- start position of the ray in the grid
		- glm::vec3(-1,-1,-1) if the ray does not intersect with the grid
	-------------------------------------------------------------------------*/
	__host__ __device__
	glm::ivec3 getRaycastParameters(
		const glm::vec3 &ray_origin,
		const glm::vec3 &ray_direction,
		glm::vec3 *t,
		glm::vec3 *d);

	T * grid;
};

template<class T>
__host__ __device__ Grid<T>::Grid(glm::vec3 origin, int resolution, float size, T fill)
	: resolution(resolution), origin(origin), size(size), cell_size(size/resolution)
{
	grid = new T[resolution*resolution*resolution];
	for (int i = 0; i < resolution*resolution*resolution; i++)
		grid[i] = fill;
}

template<class T>
__host__ __device__ Grid<T>::~Grid()
{
	delete[] (grid);
}

template<class T>
__host__ __device__ T & Grid<T>::operator()(int x, int y, int z)
{
	return grid[z * resolution * resolution + y * resolution + x];
}

template<class T>
__host__ __device__ T & Grid<T>::operator()(glm::ivec3 idx)
{
	return (*this)(idx.x, idx.y, idx.z);
}

template<class T>
__host__ __device__ inline Grid<T> & Grid<T>::operator=(const Grid<T> & obj)
{
	Grid<T> temp(obj);
	thrust::swap(temp, *this);
	return *this;
}

template<class T>
__host__ __device__ inline Grid<T> & Grid<T>::operator=(Grid<T> obj)
{
	thrust::swap(obj, *this);
	return *this;
}

template<class T>
__host__ __device__ glm::ivec3 Grid<T>::castRay(const glm::vec3 &ray_origin, const glm::vec3 &ray_dir, glm::vec3 *intersection)
{
	glm::ivec3 result;
	glm::vec3 t, d, ray_direction;
	float t_total = 0;

	ray_direction = glm::normalize(ray_dir);

	result = getRaycastParameters(ray_origin, ray_direction, &t, &d);

	if (result.x >= resolution || result.x < 0
		|| result.y >= resolution || result.y < 0
		|| result.z >= resolution || result.z < 0)
		return glm::vec3(-1, -1, -1);

	while ((*this)(result) == unitialized_value)
	{
		if (t.x < t.y)
		{
			if (t.x < t.z)
			{
				result.x += (ray_direction.x > 0) - (ray_direction.x < 0);
				t_total = t.x;
				t.x += d.x;
				if (result.x >= resolution || result.x < 0)
					return glm::vec3(-1, -1, -1);
			}
			else
			{
				result.z += (ray_direction.z > 0) - (ray_direction.z < 0);
				t_total = t.z;
				t.z += d.z;
				if (result.z >= resolution || result.z < 0)
					return glm::vec3(-1, -1, -1);
			}
		}
		else
		{
			if (t.y < t.z)
			{
				result.y += (ray_direction.y > 0) - (ray_direction.y < 0);
				t_total = t.y;
				t.y += d.y;
				if (result.y >= resolution || result.y < 0)
					return glm::vec3(-1, -1, -1);
			}
			else
			{
				result.z += (ray_direction.z > 0) - (ray_direction.z < 0);
				t_total = t.z;
				t.z += d.z;
				if (result.z >= resolution || result.z < 0)
					return glm::vec3(-1, -1, -1);
			}
		}
	}
	*intersection = ray_origin + (t_total + 1) * ray_direction;

	return result;
}

template<class T>
__host__ __device__ glm::ivec3 Grid<T>::getRaycastParameters(const glm::vec3 & ray_origin, const glm::vec3 & ray_direction, glm::vec3 *t, glm::vec3 *d)
{
	float origin_grid, origin_cell;
	int i;
	glm::ivec3 result;

	for (i = 0; i < ray_origin.length(); i++)
		result[i] = int((ray_origin[i] - origin[i]) / cell_size);

	for (i = 0; i < t->length(); i++)
	{
		origin_grid = ray_origin[i] - origin[i];
		origin_cell = origin_grid / cell_size;

		if (ray_direction[i] == 0)
		{
			(*t)[i] = FLT_MAX;
			(*d)[i] = 0;
		}
		else
		{
			if (ray_direction[i] > 0)
			{
				(*d)[i] = cell_size / ray_direction[i];
				(*t)[i] = ((glm::floor(origin_cell) + 1) * cell_size - origin_grid) / ray_direction[i];
			}
			else
			{
				(*d)[i] = -cell_size / ray_direction[i];
				(*t)[i] = (glm::floor(origin_cell) * cell_size - origin_grid) / ray_direction[i];
			}
		}
	}

	return result;
}



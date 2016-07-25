#pragma once
#include <glm\common.hpp>
#include <utility>
/*
This class describes a NxNxN Grid

Implementation: http://www.scratchapixel.com/lessons/advanced-rendering/introduction-acceleration-structure/grid
*/

template<class T> class Grid
{
public:
	Grid<T>(glm::vec3 origin, int resolution, float size, T unitialized_value);
	~Grid<T>();

	T & operator()(int x, int y, int z);
	T & operator()(glm::ivec3 idx);

	/*-------------------------------------------------------------------------
	Cast a ray through the grid
	
	Output:
	@intersection: the point of intersection between the ray and the cell grid face
	@Return:
		- coordinates of the intersection
		- glm::ivec3(-1,-1,-1) otherwise 
	-------------------------------------------------------------------------*/
	glm::ivec3 castRay(const glm::vec3 &ray_origin, const glm::vec3 &ray_direction, glm::vec3 *intersection);



	const int resolution;
	const float size, cell_size;
	const glm::vec3 origin;

	const T unitialized_value;
	T *grid;

protected:

	/*-------------------------------------------------------------------------
	Output:
	@t: current distances to the next ray-cell intersection
	@d: distances between two cells in X/Y/Z axis along the ray 
	@return:
		- start position of the ray in the grid
		- glm::vec3(-1,-1,-1) if the ray does not intersect with the grid
	-------------------------------------------------------------------------*/
	glm::ivec3 getRaycastParameters(
		const glm::vec3 &ray_origin,
		const glm::vec3 &ray_direction,
		glm::vec3 *t,
		glm::vec3 *d);
};


#include "Grid.h"

template<class T>
Grid<T>::Grid(glm::vec3 origin, int resolution, float size, T unitialized_value)
	: resolution(resolution), origin(origin), size(size), cell_size(size/resolution), unitialized_value(unitialized_value)
{
	grid = new T[resolution*resolution*resolution];
	for (int i = 0; i < resolution*resolution*resolution; i++)
		grid[i] = unitialized_value;
}

template<class T>
Grid<T>::~Grid()
{
	delete(grid);
}

template<class T>
T & Grid<T>::operator()(int x, int y, int z)
{
	return grid[z * resolution * resolution + y * resolution + x];
}

template<class T>
T & Grid<T>::operator()(glm::ivec3 idx)
{
	return (*this)(idx.x, idx.y, idx.z);
}

template<class T>
glm::ivec3 Grid<T>::castRay(const glm::vec3 &ray_origin, const glm::vec3 &ray_direction, glm::vec3 *intersection)
{
	glm::ivec3 result;
	glm::vec3 t, d;
	glm::bvec3 isZero;
	float t_total = 0;

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
				result.x += (int)glm::sign(ray_direction.x);
				t_total += t.x;
				t.x += d.x;
				if (result.x >= resolution || result.x < 0)
					return glm::vec3(-1, -1, -1);
			}
			else
			{
				result.z += (int)glm::sign(ray_direction.z);
				t_total += t.z;
				t.z += d.z;
				if (result.z >= resolution || result.z < 0)
					return glm::vec3(-1, -1, -1);
			}
		}
		else
		{
			if (t.y < t.z)
			{
				result.y += (int)glm::sign(ray_direction.y);
				t_total += t.y;
				t.y += d.y;
				if (result.y >= resolution || result.y < 0)
					return glm::vec3(-1, -1, -1);
			}
			else
			{
				result.z += (int)glm::sign(ray_direction.z);
				t_total += t.z;
				t.z += d.z;
				if (result.z >= resolution || result.z < 0)
					return glm::vec3(-1, -1, -1);
			}
		}
	}

	*intersection = ray_origin + t_total * ray_direction;

	return result;
}

template<class T>
glm::ivec3 Grid<T>::getRaycastParameters(const glm::vec3 & ray_origin, const glm::vec3 & ray_direction, glm::vec3 *t, glm::vec3 *d)
{
	float origin_grid, origin_cell;
	glm::ivec3 result;

	for (int i = 0; i < t->length(); i++)
	{
		origin_grid = ray_origin[i] - ray_origin[i];
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



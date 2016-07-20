#include "Grid.h"

template<class T>
Grid<T>::Grid(glm::vec3 origin, int resolution, float size)
{
	grid = new std::pair<bool, T>[size*size*size](0, NULL);
	this.origin = origin;
	this.size = size;
	this.gresolution = resolution;
	this.cell_size = size / resolution;
}

template<class T>
Grid<T>::~Grid()
{
	delete(grid);
}

template<class T>
T & Grid<T>::operator()(int x, int y, int z)
{
	return &(grid[z * grid_resolution * grid_resolution + y * grid_resolution + x].value);
}

template<class T>
T & Grid<T>::operator()(glm::ivec3 idx)
{
	return this(idx.x, idx.y, idx.z);
}

template<class T>
glm::ivec3 Grid<T>::castRay(glm::vec3 ray_origin, glm::vec3 ray_direction)
{
	glm::ivec3 result;
	glm::vec3 t, d;
	glm::bvec3 isZero;

	result = getRaycastParameters(ray_origin, ray_direction, t, d, isZero);

	if (x >= coarse_grid_resolution || x < 0
		|| y >= coarse_grid_resolution || y < 0
		|| z >= coarse_grid_resolution || z < 0)
		return glm::vec3(-1, -1, -1);

	while (get_coarse_voxel(x, y, z) == 0)
	{
		if (tx < ty)
		{
			if (tx < tz)
			{
				x += glm::sign(ray_direction.x);
				t = tx;
				tx += dx;
				if (x >= coarse_grid_resolution || x < 0)
					return glm::vec3(-1, -1, -1);
			}
			else
			{
				z += glm::sign(ray_direction.z);
				t = tz;
				tz += dz;
				if (z >= coarse_grid_resolution || z < 0)
					return glm::vec3(-1, -1, -1);
			}
		}
		else
		{
			if (ty < tz)
			{
				y += glm::sign(ray_direction.y);
				t = ty;
				ty += dy;
				if (y >= coarse_grid_resolution || y < 0)
					return glm::vec3(-1, -1, -1);
			}
			else
			{
				z += glm::sign(ray_direction.z);
				t = tz;
				tz += dz;
				if (z >= coarse_grid_resolution || z < 0)
					return glm::vec3(-1, -1, -1);
			}
		}
	}

	return result;
}

template<class T>
glm::ivec3 Grid<T>::getRaycastParameters(const glm::vec3 & ray_origin, const glm::vec3 & ray_direction, glm::vec3 & t, glm::vec3 & d, glm::bvec3 & isZero)
{
	float origin_grid, origin_cell;
	glm::ivec3 result;

	result = projectRayToGrid(ray_origin, ray_direction);
	if (result.x < 0)
		return result;

	for (int i = 0; i < t.length; i++)
	{
		result[i] = (int)(ray_origin[i] / cell_size);
		origin_grid = ray_origin[i] - origin[i];
		origin_cell = origin_grid / cell_size;

		if (ray_dir == 0)
		{
			t = -1;
			d = 0;
			isZero[i] = 1;
		}
		else
		{
			isZero[i] = 0;
			if (ray_dir > 0)
			{
				d = cell_size / ray_dir;
				t = ((glm::floor(org_cell) + 1) * cell_size - org_grid) / ray_dir;
			}
			else
			{
				d = -cell_size / ray_dir;
				t = (glm::floor(org_cell) * cell_size - org_grid) / ray_dir;
			}
		}
	}

	return result;
}

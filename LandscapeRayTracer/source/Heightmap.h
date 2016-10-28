#pragma once
#include <glm\glm.hpp>
#include <vector>

using namespace std;

class Heightmap
{
public:
	Heightmap(int resolution, int levels, glm::vec3 origin, float cell_size) : resolution(resolution), levels(levels), origin(origin), cell_size(cell_size)
	{
		grids.resize(levels);
		for (int i = 0; i < levels; ++i)
		{
			grids[i] = new float[pow((resolution / (pow(2, i))), 2)]();
		}
	}

	~Heightmap()
	{
		for (int i = 0; i < levels; ++i)
		{
			delete(grids[i]);
		}
	}

	glm::vec3 trace_ray(glm::vec3 ray_origin, glm::vec3 ray_direction)
	{
		glm::vec3 t, d, color(1, 0, 0);
		int LOD = 0, cells_walked = 0;
		float t_total, current_height = 0.0f;
		ray_direction = glm::normalize(ray_direction);

		glm::ivec2 grid_position = get_parameters(&t, &d, ray_origin, ray_direction);


		// check if ray is out of bounds
		if (is_ray_out_of_boundaries(grid_position))
			return glm::vec3(0, 0, 0);

		while (current_height > get_grid_height(LOD, grid_position.x, grid_position.y))
		{
			if (t.x < t.y)
			{
				grid_position.x++;
				t.x += d.x;
				t_total = t.x;
			}
			else
			{
				grid_position.y++;
				t.y += d.y;
				t_total = t.y;
			}
			
			// check if ray is out of bounds
			if (is_ray_out_of_boundaries(grid_position))
				return glm::vec3(0, 0, 0);
			
			// decreases the LOD
			cells_walked++;
			if (cells_walked > 4 && LOD < levels)
			{
				cells_walked = 0;
				LOD++;
			}
		}

		return color;
	}

	glm::ivec2 get_parameters(glm::vec3 *t, glm::vec3 *d, glm::vec3 ray_origin, glm::vec3 ray_direction)
	{
		float origin_grid, origin_cell;
		glm::ivec2 grid_position;

		for (int i = 0; i < 2; i++)
			grid_position[i] = int((ray_origin[i] - origin[i]) / cell_size);

		for (int i = 0; i < t->length(); i++)
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

		return grid_position;
	}

	float get_grid_height(int LOD, int x, int y)
	{
		return (grids[LOD])[(int)(resolution / (pow(2, LOD))) * y + x];
	}

	bool is_ray_out_of_boundaries(glm::ivec2 grid_position)
	{
		return grid_position.x >= resolution || grid_position.x < 0
			|| grid_position.y >= resolution || grid_position.y < 0;
	}

protected:
	int resolution;
	int levels;
	float cell_size;

	glm::vec3 origin;

	vector<float*> grids;
};
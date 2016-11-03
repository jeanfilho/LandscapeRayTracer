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

	int LOD_change = INT_MAX;
	glm::vec3 trace_ray(glm::vec3 ray_origin, glm::vec3 ray_direction)
	{
		glm::vec2 t, d;
		glm::vec3 color(1, 0, 0);
		int LOD = 0;
		float t_total = 0,
			current_height = ray_origin.z;
		glm::ivec2 cells_walked(0,0);
		ray_direction = glm::normalize(ray_direction);

		glm::ivec2 grid_position = get_parameters(&t, &d, ray_origin, ray_direction);


		// check if ray is out of bounds
		if (is_ray_out_of_boundaries(grid_position, LOD))
			return glm::vec3(0, 0, 0);

		while (current_height > get_grid_height(LOD, grid_position.x, grid_position.y))
		{
			if (t.x < t.y)
			{
				grid_position.x += glm::sign(ray_direction.x);
				t.x += d.x;
				t_total = t.x;
				cells_walked.x++;
			}
			else
			{
				grid_position.y += glm::sign(ray_direction.y);
				t.y += d.y;
				t_total = t.y;
				cells_walked.y++;
			}
			current_height = t_total * ray_direction.z + ray_origin.z;
			
			// decreases the LOD
			if ((cells_walked.x > LOD_change || cells_walked.y > LOD_change) && LOD < levels - 1)
			{
				grid_position /= 2;
				LOD++;
				d *= 2;
				cells_walked.x = cells_walked.y = 0;
			}

			// check if ray is out of bounds
			if (is_ray_out_of_boundaries(grid_position, LOD))
				return glm::vec3(0, 0, 0);
		} 
		return color;
	}

	glm::ivec2 get_parameters(glm::vec2 *t, glm::vec2 *d, glm::vec3 ray_origin, glm::vec3 ray_direction)
	{
		float origin_grid, origin_cell;
		glm::ivec2 grid_position;

		for (int i = 0; i < grid_position.length(); i++)
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

	bool is_ray_out_of_boundaries(glm::ivec2 grid_position, int LOD)
	{
		int bound = resolution / pow(2, LOD);
		return grid_position.x >= bound || grid_position.x < 0
			|| grid_position.y >= bound || grid_position.y < 0;
	}

protected:
	int resolution;
	int levels;
	float cell_size;

	glm::vec3 origin;

	vector<float*> grids;
};
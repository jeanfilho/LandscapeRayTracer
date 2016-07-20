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
	Grid<T>(glm::vec3 origin, int resolution, float size);
	~Grid<T>();

	T & operator()(int x, int y, int z);
	T & operator()(glm::ivec3 idx);

	class Element
	{
		bool filled;
		T value;

		void operator=(T value)
		{
			this.value = value;
		}
	};

	/*-------------------------------------------------------------------------
	Cast a ray through the grid
	
	@Return:
		- coordinates of the intersection
		- glm::ivec3(-1,-1,-1) otherwise 
	-------------------------------------------------------------------------*/
	glm::ivec3 castRay(glm::vec3 ray_origin, glm::vec3 ray_direction);



	const int resolution;
	const float size, cell_size;
	const glm::vec3 origin;

protected:

	/*-------------------------------------------------------------------------
	@t: current distances to the next ray-cell intersection
	@d: distances between two cells in X/Y/Z axis along the ray 

	@return:
		- start position of the ray in the grid
		- glm::vec3(-1,-1,-1) if the ray does not intersect with the grid
	-------------------------------------------------------------------------*/
	glm::ivec3 getRaycastParameters(
		const glm::vec3 &ray_origin,
		const glm::vec3 &ray_direction,
		glm::vec3 &t,
		glm::vec3 &d,
		glm::bvec3 &isZero);



	Element *grid;
};

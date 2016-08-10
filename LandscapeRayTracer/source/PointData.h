#pragma once
#define GLM_FORCE_CUDA
#include <glm\glm.hpp>
#include <thrust\swap.h>

class PointData
{
public:
	__host__ __device__
	PointData(glm::vec3 position, glm::vec3 color) : color(color), position(position) {}

	__host__ __device__
	~PointData() {}

	__host__ __device__
	PointData& operator=(const PointData& rs)
	{
		PointData temp(rs);
		thrust::swap(temp, *this);
		return *this;
	}

	const glm::vec3 color, position;
};


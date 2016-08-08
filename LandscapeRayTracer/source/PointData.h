#pragma once
#include <glm\glm.hpp>

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
		swap(temp, *this);
		return *this;
	}

	const glm::vec3 color, position;
};


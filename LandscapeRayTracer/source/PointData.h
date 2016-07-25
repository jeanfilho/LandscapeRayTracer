#pragma once
#include <glm\common.hpp>

class PointData
{
public:
	PointData(glm::vec3 color) : color(color) {}
	~PointData() {}

	const glm::vec3 color;
};


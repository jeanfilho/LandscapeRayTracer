#pragma once
#include "glm\common.hpp"

class Camera
{
public:
	Camera(glm::vec3 position, glm::vec3 forward, glm::vec3 up, float frame_distance, float frame_height, float frame_width);
	~Camera();

	void rotate(glm::vec3 axis, float angle);

	const glm::vec3 &forward, &up, &right;
	const float &frame_distance, &frame_height, &frame_width;

	glm::vec3 position;

protected:
	glm::vec3 _forward, _up, _right;
	float _frame_distance, _frame_height, _frame_width;
};


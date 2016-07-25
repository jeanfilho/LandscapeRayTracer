#pragma once
#include "glm\common.hpp"

class Camera
{
public:
	Camera(glm::vec3 position, glm::vec3 forward, glm::vec3 up, float frame_distance, float frame_height, float frame_width)
		: position(position), _forward(forward), _up(up), _frame_distance(frame_distance), _frame_height(frame_height), _frame_width(frame_width) {}
	~Camera() {};

	void rotate(glm::vec3 axis, float angle);

	const glm::vec3
		&forward = _forward,
		&up = _up,
		&right = _right;
	const float
		&frame_distance = _frame_distance,
		&frame_height = _frame_height,
		&frame_width = _frame_width;

	glm::vec3 position;

protected:
	glm::vec3 _forward, _up, _right;
	float _frame_distance, _frame_height, _frame_width;
};


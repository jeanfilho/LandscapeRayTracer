#pragma once
#include "glm\glm.hpp"

class Camera
{
public:
	~Camera() {};
	Camera(int window_width, int window_height, glm::vec3 up, glm::vec3 forward, glm::vec3 position, float frame_distance) :
		window_width(window_width), window_height(window_height),
		forward(glm::normalize(forward)), up(glm::normalize(up)), right(glm::cross(forward, up)),
		position(position), frame_distance(frame_distance)
	{};


	//  variables representing the window size
	int window_width;
	int window_height;
	float frame_distance;

	glm::vec3 up, right, forward, position;
};
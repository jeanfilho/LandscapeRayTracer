#pragma once
#include "glm\glm.hpp"

class Camera
{
public:
	~Camera() {};
	Camera(int window_width, int window_height) : window_width(window_width), window_height(window_height) {};


	//  variables representing the window size
	int window_width;
	int window_height;
	


};
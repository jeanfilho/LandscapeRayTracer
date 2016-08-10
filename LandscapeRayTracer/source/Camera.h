#pragma once
#define GLM_FORCE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust\swap.h>

#include "glm\glm.hpp"


class Camera
{
public:
	__host__ __device__
	Camera() : position(glm::vec3(0, 0, 0)), _forward(glm::vec3(1, 0, 0)), _up(glm::vec3(0, 1, 0)), _right(glm::normalize(glm::cross(_up, _forward))),
		_frame_distance(100), _frame_height(512), _frame_width(512){}

	__host__ __device__
	Camera(glm::vec3 position, glm::vec3 forward, glm::vec3 up, float frame_distance, float frame_height, float frame_width)
		: position(position), _forward(glm::normalize(forward)), _up(glm::normalize(up)), _right(glm::normalize(glm::cross(_up, _forward))),
		_frame_distance(frame_distance), _frame_height(frame_height), _frame_width(frame_width){}
	
	__host__ __device__
	~Camera() {};

	__host__ __device__ Camera& operator=(Camera obj)
	{
		thrust::swap(*this, obj);
		return *this;
	}

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


#pragma once

#define GLM_FORCE_CUDA

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

#include "Camera.h"
#include "Grid.h"
#include "PointData.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "thrust\device_vector.h"
#include "helper_cuda.h"
#include "device_launch_parameters.h"

#include "glm\glm.hpp"



using namespace std;
namespace CudaWorker
{
	void setUp(glm::vec3 *h_pixel_array, Camera cam, int window_height, int window_width);
	void updatePixelArray(glm::vec3 *h_pixel_array, int window_height, int window_width);
	void release();
}
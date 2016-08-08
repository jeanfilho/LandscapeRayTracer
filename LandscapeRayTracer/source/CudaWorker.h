#pragma once

#include <iostream>

#include "Camera.h";
#include "Grid.h";
#include "PointData.h";

#include "glm\common.hpp";
#include "glm\geometric.hpp";

#include "cuda.h";
#include "cuda_runtime.h";
#include "cuda_runtime_api.h";

using namespace std;

class CudaWorker
{
public:
	CudaWorker() {}
	~CudaWorker() {};

	static void par_castRay(glm::fvec3* pixel_array, const Camera* cam, const Grid<Grid<PointData*>*>* grid, int window_height, int window_width);
};
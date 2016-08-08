#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

#include "Camera.h"
#include "Grid.h"
#include "PointData.h"

#include "glm\common.hpp"
#include "glm\geometric.hpp"

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "thrust\device_vector.h"
#include "thrust\host_vector.h"

using namespace std;
/*
	This class handles all communication with the GPU and CUDA operations
*/
class CudaWorker
{
public:

	static cudaError_t loadPoints(float* max_height, float *min_height, thrust::host_vector<PointData> *points);
	static cudaError_t startRoutine(int window_height, int window_witdh);
	static cudaError_t exitRoutine();

private:
	CudaWorker() {}
	~CudaWorker() {}
};
#pragma once
#include <glm\glm.hpp>
#include <vector>

using namespace std;

class Heightmap
{
public:
	~Heightmap();
	Heightmap(int dimension, int levels, glm::vec3 origin) : dimension(dimension), levels(levels), origin(origin)
	{
		grids.resize(levels);
	}

protected:
	int dimension;
	int levels;

	glm::vec3 origin;

	vector<float*> grids;
};
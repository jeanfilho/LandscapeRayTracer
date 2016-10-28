#pragma once
#include <glm\glm.hpp>
#include <vector>

using namespace std;

class Heightmap
{
public:
	Heightmap(int dimension, int levels, glm::vec3 origin) : dimension(dimension), levels(levels), origin(origin)
	{
		grids.resize(levels);
		for(int i = 0; i < levels; ++i)
		{
			grids[i] = new float[(dimension / (2 ^ i)) ^ 2]();
		}
	}

	~Heightmap()
	{
		for (int i = 0; i < levels; ++i)
		{
			delete(grids[i]);
		}
	}



protected:
	int dimension;
	int levels;

	glm::vec3 origin;

	vector<float*> grids;
};
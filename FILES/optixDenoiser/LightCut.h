#pragma once
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_vector_types.h>
#include "prd.h"


class LightCut
{
public:
	LightCut(const std::vector<Photon> &photonList);
	void writeToBuffer(optix::Buffer buffer);
	~LightCut();
private:
	// compress heap to array
	LightcutNode *m_tree = nullptr;
	int m_treeSize = 0;
};


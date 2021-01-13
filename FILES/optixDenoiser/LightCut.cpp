#include "LightCut.h"
#include <optixu/optixu_math_namespace.h>    
using namespace optix;

LightCut::LightCut(const std::vector<Photon>& photonList) {
	// clustering using Lloyd algorithm
	const int k = 5;

	// Randomly choosing marker
	int markerIndex[k];
	float3 markerPosition[k];
	std::vector<bool> markerIndexHash(photonList.size(), false);
	for (int i = 0; i < k; ++i) {
		size_t index;
		do {
			index = ((float)rand() / (float)(RAND_MAX - 1) * photonList.size());
		} while (markerIndexHash[index]);
		markerIndex[i] = index;
		markerIndexHash[index] = true;
		markerPosition[i] = photonList[index].position;
	}
	std::vector<int> clusterList(photonList.size());
	int iteration = 10;
	while (iteration--) {
		float3 summationPosition[k] = {};
		int clusterSize[k] = {};
		for (int photonId = 0; photonId < photonList.size(); ++photonId) {
				// Compute each point's belong to what cluster
			float minDistance = 1e9;
			int clusterId = 0;
			for (int i = 0; i < k; ++i) {
				float distanceToMarker = length(photonList[photonId].position - photonList[markerIndex[i]].position);
				if (distanceToMarker < minDistance) {
					minDistance = distanceToMarker;
					clusterList[photonId] = clusterId = i;
				}
			}
			// Update cluster centroid - calculate individual cluster info
			clusterSize[clusterId]++;
			summationPosition[clusterId] += photonList[photonId].position;
		}

		// Update cluster centroid
		for (int clusterId = 0; clusterId < k; ++clusterId) {
			markerPosition[clusterId] = summationPosition[clusterId] / clusterSize[clusterId];
		}
	}

	// Place actual point to cluster
	std::vector<std::vector< Photon>> cluster(k, std::vector<Photon>());
	for (int i = 0; i < photonList.size(); ++i) {
		cluster[clusterList[i]].push_back(photonList[i]);
	}
	// Build heap tree
	m_tree = new LightcutNode[2*photonList.size()];
	m_treeSize = 2 * photonList.size();
	int currentTreePos = m_treeSize;
	int currentCluster = 0, currentClusterPoint = 0;
	for (int i = photonList.size()-1; i >= 0; --i) {
		// FIXME need to verify lightcut info
		LightcutNode node;
		if ((--cluster.end())->empty()) {
			cluster.pop_back();
		}
		node.lightcut = *(--((--cluster.end())->end()));
		(--cluster.end())->pop_back();
		// FIXME intensity photon energy to 3D
		//node.intensity = node.lightcut.energy * node.lightcut.color;
		node.intensity = node.lightcut.color;
		node.minBoundBox = node.maxBoundBox = node.lightcut.position;
		node.representativeId = --currentTreePos;
		node.leftTreeId = node.representativeId;
		node.rightTreeId = node.representativeId;
		node.isLeaf = 1;
		m_tree[node.representativeId] = node;
	}
	--currentTreePos;
	while (currentTreePos) {
		int leftTreeId = currentTreePos * 2;
		int rightTreeId = leftTreeId + 1;

		LightcutNode node;
		node.lightcut =  m_tree[leftTreeId].lightcut;
		node.intensity = m_tree[leftTreeId].intensity + ((rightTreeId < m_treeSize) ? m_tree[rightTreeId].intensity : make_float3(0.0, 0.0, 0.0));
		node.minBoundBox = rightTreeId < m_treeSize ? fminf(m_tree[leftTreeId].minBoundBox,  m_tree[rightTreeId].minBoundBox) : m_tree[leftTreeId].minBoundBox;
		node.maxBoundBox = rightTreeId < m_treeSize ? fmaxf(m_tree[leftTreeId].maxBoundBox, m_tree[rightTreeId].maxBoundBox) : m_tree[leftTreeId].maxBoundBox;
		node.representativeId = rightTreeId < m_treeSize && length(m_tree[rightTreeId].intensity) > length(m_tree[leftTreeId].intensity) ? rightTreeId : leftTreeId;
		node.leftTreeId = leftTreeId;
		node.rightTreeId = rightTreeId < m_treeSize ?leftTreeId + 1:currentTreePos;
		node.isLeaf = 0;
		m_tree[currentTreePos] = node;
		currentTreePos--;
	}
}

LightCut::~LightCut() {
	delete[] m_tree;
}

void LightCut::writeToBuffer(Buffer buffer) {
	LightcutNode* lightCutBuffer = (LightcutNode * )(buffer->map());
	for (int i = 0; i < m_treeSize; ++i) {
		printf("%d %d\n", m_tree[i].leftTreeId, m_tree[i].rightTreeId);
		lightCutBuffer[i] = m_tree[i];
	}
	buffer->unmap();
}
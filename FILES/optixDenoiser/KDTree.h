//
// Created by Edge on 2020/11/20.
//

#pragma once

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_vector_types.h>
// Edge Edited - START
#include <cmath>
// Edge Edited - END
#include "prd.h"

class KDTree {
public:
    KDTree(const std::vector<Photon>& photonList) {
        m_treeSize = photonList.size();
        m_tree = new KDTreeNode[m_treeSize];

        for (int photonId = 0; photonId < photonList.size();++photonId) {
            m_tree[photonId].photonIndex = photonId;
            m_tree[photonId].isLeaf = true;
            m_tree[photonId].leftTreeIndex = INVALID_INDEX;
            m_tree[photonId].rightTreeIndex = INVALID_INDEX;
            if (photonId) {
                insert(ROOT, photonId, photonList, 0);
            }
        }
    }

    ~KDTree() {
        delete[] m_tree;
    }

    void writeToBuffer(optix::Buffer buffer) {
        KDTreeNode* KDTreeBuffer = (KDTreeNode*)(buffer->map());
        for (int i = 0; i < m_treeSize; ++i) {
            KDTreeBuffer[i] = m_tree[i];
        }
        buffer->unmap();
    }


private:
    void insert(int nodeIndex, int photonIndex, const std::vector<Photon>& photonList, int depth){
        int dim = depth % DIMENSION;
        // Edge Edited - START
        m_max_depth = std::max(m_max_depth, depth+1);
        // Edge Edited - END
        optix::float3 nodeOrigin = photonList[nodeIndex].position;
        float nodeOriginFloat[3] = {nodeOrigin.x, nodeOrigin.y, nodeOrigin.z};
        optix::float3 photonOrigin = photonList[photonIndex].position;
        float photonOriginFloat[3] = { photonOrigin.x, photonOrigin.y, photonOrigin.z };

        if (m_tree[nodeIndex].isLeaf) {
            if (photonOriginFloat[dim] < nodeOriginFloat[dim]) {
                m_tree[nodeIndex].leftTreeIndex = photonIndex;
            } else {
                m_tree[nodeIndex].rightTreeIndex = photonIndex;
            }
            m_tree[nodeIndex].isLeaf = false;
        } else {
            if (photonOriginFloat[dim] < nodeOriginFloat[dim]) {
                if (m_tree[nodeIndex].leftTreeIndex == INVALID_INDEX) {
                    m_tree[nodeIndex].leftTreeIndex = photonIndex;
                } else {
                    insert(m_tree[nodeIndex].leftTreeIndex, photonIndex, photonList, depth + 1);
                }
            } else {
                if (m_tree[nodeIndex].rightTreeIndex == INVALID_INDEX) {
                    m_tree[nodeIndex].rightTreeIndex = photonIndex;
                } else {
                    insert(m_tree[nodeIndex].rightTreeIndex, photonIndex, photonList, depth + 1);
                }
            }
        }
    }

    KDTreeNode *m_tree;
    int m_treeSize;

    static const int DIMENSION = 3;
    static const int ROOT = 0;
    static const int INVALID_INDEX = -1;

    // Edge Edited - START
public:
    int m_max_depth;
    // Edge Edited - END
};

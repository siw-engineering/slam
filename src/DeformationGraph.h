#include <vector>
#include <Eigen/Dense>
#include "GraphNode.h"


class DeformationGraph{

  public:
    DeformationGraph(int k, std::vector<Eigen::Vector3f>* sourceVertices);
    virtual ~DeformationGraph();
    std::vector<GraphNode*>& getGraph();
    std::vector<unsigned long long int>& getGraphTimes();
    void initialiseGraph(std::vector<Eigen::Vector3f>* customGraph, std::vector<unsigned long long int>* graphTimeMap);
    void connectGraphSeq();

    // Number of neighbours
    const int k;
  private:
  	bool initialised;

  	// Maps vertex indices to neighbours and weights
    // std::vector<std::vector<VertexWeightMap> > vertexMap;
    std::vector<Eigen::Vector3f>* sourceVertices;
    

    // Graph itself
    std::vector<GraphNode> graphNodes;
    std::vector<GraphNode*> graph;


    std::vector<Eigen::Vector3f>* graphCloud;
    std::vector<unsigned long long int> sampledGraphTimes;
    unsigned int lastPointCount;
};
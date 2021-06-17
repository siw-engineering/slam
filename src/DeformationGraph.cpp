#include "DeformationGraph.h"


DeformationGraph::DeformationGraph(int k, std::vector<Eigen::Vector3f>* sourceVertices)
    : k(k),
      initialised(false),
      graphCloud(new std::vector<Eigen::Vector3f>) {}

DeformationGraph::~DeformationGraph() {
  if (initialised) {
    graphNodes.clear();
  }

  delete graphCloud;

}

std::vector<GraphNode*>& DeformationGraph::getGraph() { return graph; }

std::vector<unsigned long long int>& DeformationGraph::getGraphTimes() { return sampledGraphTimes; }


void DeformationGraph::initialiseGraph(std::vector<Eigen::Vector3f>* customGraph, std::vector<unsigned long long int>* graphTimeMap) {
  graphCloud->clear();

  sampledGraphTimes.clear();

  sampledGraphTimes.insert(sampledGraphTimes.end(), graphTimeMap->begin(), graphTimeMap->end());

  graphCloud->insert(graphCloud->end(), customGraph->begin(), customGraph->end());

  graphNodes.clear();

  graph.clear();

  graphNodes.resize(graphCloud->size());

  for (unsigned int i = 0; i < graphCloud->size(); i++) {
    graphNodes[i].id = i;

    graphNodes[i].enabled = true;

    graphNodes[i].position = graphCloud->at(i);

    graphNodes[i].translation = Eigen::Vector3f::Zero();

    graphNodes[i].rotation.setIdentity();

    graph.push_back(&graphNodes[i]);
  }

  connectGraphSeq();

  initialised = true;
}


void DeformationGraph::connectGraphSeq() {
  for (int i = 0; i < k / 2; i++) {
    for (int n = 0; n < k + 1; n++) {
      if (i == n) {
        continue;
      }

      graphNodes[i].neighbours.push_back(n);
    }
  }

  for (unsigned int i = k / 2; i < graphCloud->size() - (k / 2); i++) {
    for (int n = 0; n < k / 2; n++) {
      graphNodes[i].neighbours.push_back(i - (n + 1));
      graphNodes[i].neighbours.push_back(i + (n + 1));
    }
  }

  for (unsigned int i = graphCloud->size() - (k / 2); i < graphCloud->size(); i++) {
    for (unsigned int n = graphCloud->size() - (k + 1); n < graphCloud->size(); n++) {
      if (i == n) {
        continue;
      }

      graphNodes[i].neighbours.push_back(n);
    }
  }
}

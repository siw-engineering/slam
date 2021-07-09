#include "Deformation.h"

Deformation::Deformation()
  : def(4, &pointPool),
    sample_hst(new float[1024*4]),
    graphPosePoints(new std::vector<Eigen::Vector3f>)
  {
  	// float* sample_hst = new float[1024*4];
	
}

Deformation::~Deformation()
{
	// delete graphPosePoints;
}
std::vector<GraphNode*>& Deformation::getGraph() { return def.getGraph(); }

void Deformation::sampleGraphFrom(Deformation & other)
{
	float* otherVerts = other.getVertices();
	int sampleRate = 5;

	if (other.getCount() / sampleRate  > def.k)
	{
		for (int i = 0; i< other.getCount(); i += sampleRate)
		{
			Eigen::Vector3f newPoint(otherVerts[i*4], otherVerts[i*4+1], otherVerts[i*4+2]);
			graphPosePoints->push_back(newPoint);
            // if(i > 0 && sample_hst[i*4 + 3] < graphPoseTimes.back())
            // {
            //     // assert(false && "Assumption failed");
            // }
            graphPoseTimes.push_back(otherVerts[i*4 + 3]);

		}
		
	    def.initialiseGraph(graphPosePoints, &graphPoseTimes);

	    graphPoseTimes.clear();
	    graphPosePoints->clear();

	}

} 

void Deformation::sampleGraphModel(DeviceArray<float>& model_buffer, int count/**, int* g_count**/) {
	
	sample_points.create(1024 * 4);
	graph_count = 0;
	SampleGraph(model_buffer, count, sample_points, &graph_count);
	sample_points.download(sample_hst);

	if (graph_count > 4){
		for (int i = 0; i < graph_count; i++){
			Eigen::Vector3f newPoint(sample_hst[i*4], sample_hst[i*4 + 1], sample_hst[i*4 + 2]); 
			graphPosePoints->push_back(newPoint);

			// assumption is failing due to modelbuffer index ordering
            // if(i > 0 && sample_hst[i*4 + 3] < graphPoseTimes.back())
            // {
            //     assert(false && "Assumption failed");
            // }

            graphPoseTimes.push_back(sample_hst[i*4 + 3]);

		}

    def.initialiseGraph(graphPosePoints, &graphPoseTimes);

    graphPoseTimes.clear();
    graphPosePoints->clear();
	}

}
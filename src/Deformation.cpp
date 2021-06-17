#include "Deformation.h"

Deformation::Deformation()
  : def(4, &pointPool),
    graphPosePoints(new std::vector<Eigen::Vector3f>)
  {

	sample_points.create(1024 * 4);
}

std::vector<GraphNode*>& Deformation::getGraph() { return def.getGraph(); }


void Deformation::sampleGraphModel(DeviceArray<float>& model_buffer, int count/**, int* g_count**/) {
	
	int g_count = 0;
	SampleGraph(model_buffer, count, sample_points, &g_count);
	std::cout << " called deformation grah .....  - " << g_count  << "   , model cout - " << count << std::endl;
	float* sample_hst = new float[1024*4];
	sample_points.download(sample_hst);
	
	if (g_count > 4){

		for (int i = 0; i < g_count*4; i+=4){
			Eigen::Vector3f newPoint; 
			newPoint[0] = sample_hst[i];
			newPoint[1] = sample_hst[i + 1];
			newPoint[2] = sample_hst[i + 2];

			// Eigen::Vector<unsigned long long int> time;
			// time = sample_hst[i + 3];
			// std::cout << sample_hst[i + 3] /* << " : " << graphPoseTimes.back()*/<< std::endl;
			graphPosePoints->push_back(newPoint);

			// if (i > 0 && sample_hst[i + 3] < graphPoseTimes.back()){
			// 	std::cout << sample_hst[i + 3] << std::endl;
			// 	assert(false && "Assumption failed");
			// }
			graphPoseTimes.push_back(sample_hst[i + 3]);

			// std::cout << newPoint << std::endl;
			// std::cout << " completed " << std::endl;
		}
    def.initialiseGraph(graphPosePoints, &graphPoseTimes);

    graphPoseTimes.clear();
    graphPosePoints->clear();
	}	

}
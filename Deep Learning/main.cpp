#include <iostream>
#include "datasets.h"
#include "DLmodel.h"


using std::cout;
using std::endl;
using namespace DeepLearning::Datasets;
using namespace DeepLearning::models;


int main() {
	
	auto data = Mnist::load_data();
	
	data.test_input /= 255;
	data.train_input /= 255;
	
	SequentialModel Md;

	Md.add(new Dense(128, 784, "relu"));
	Md.add(new Dense(10, 128, "softmax"));
	Md.compile();
	Md.fit(data.train_input, data.train_target,10, 0.01f);
	float s = Md.score(data.test_input, data.test_target);

	cout << "score: " << s << endl;
	

	return 0;
}
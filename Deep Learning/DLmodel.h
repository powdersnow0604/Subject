#ifndef __DLMODEL_H__
#define __DLMODEL_H__

#include <vector>
#include "layers.h"
#include <string>


namespace DeepLearning {
	namespace models {
		
		using namespace Layers;

		class SequentialModel {
			std::vector<__basic_layer*> layers;
			float learning_rate;
			size_t epoch;

		public:
			SequentialModel(std::string loss_function = "L2", size_t epoch = 10, float _learning_rate = 0.01);
		};



	}
}


#endif
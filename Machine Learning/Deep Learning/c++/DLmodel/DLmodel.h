#ifndef __DLMODEL_H__
#define __DLMODEL_H__

#include <vector>
#include "layers.h"
#include <string>
#include "ndArray.hpp"


namespace DeepLearning {
	namespace models {
		
		using namespace Layers;

		class SequentialModel {
			std::vector<__basic_layer*> layers;
			__basic_loss_func* loss_function;
			

		public:
			SequentialModel() : loss_function(nullptr) {}
			~SequentialModel() noexcept;
			void add(__basic_layer* layer);
			void compile(std::string _loss_function = "sparse_categorical_crossentropy");
			void fit(const na::ndArray<float>& input, const na::ndArray<float>& target, size_t epoch = 10, float _learning_rate = 0.01);
			na::ndArray<float> predict(const na::ndArray<float>& input);
			float score(const na::ndArray<float>& input, const na::ndArray<float>& target);
		};



	}
}


#endif
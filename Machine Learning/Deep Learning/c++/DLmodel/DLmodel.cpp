#include "DLmodel.h"
#include <iostream>


namespace DeepLearning {
	namespace models {

		SequentialModel::~SequentialModel() noexcept
		{
			for (size_t i = layers.size() - 1; ; --i) {
				delete layers[i];

				if (i == 0) break;
			}

			delete loss_function;
		}

		void SequentialModel::add(__basic_layer* layer)
		{
			layers.push_back(layer);
		}

		void SequentialModel::compile(std::string _loss_function)
		{
			if (_loss_function == "sparse_categorical_crossentropy") {
				loss_function = new sparse_categorical_crossentropy();
			}
			else {
				loss_function = new sparse_categorical_crossentropy();
			}
		}

		void SequentialModel::fit(const na::ndArray<float>& input, const na::ndArray<float>& target, size_t epoch, float _learning_rate)
		{
			assert(layers.size() != 0);

			auto input_shp = input.shape();
			auto target_shp = target.shape();

			//수정 필요
			assert(input_shp[0] == target_shp[0]);
			assert(input_shp[1] == layers[0]->input_size);

			size_t i, j, m;
			na::ndArray<size_t> randind = na::range(0ull, input_shp[0]);

			std::cout << std::endl;
			for(size_t k = 1; k <= epoch; ++k) {

				randind.shuffle();

				for (i = 0; i < input_shp[0]; ++i) {
					//forwarding
					layers[0]->forward(input[randind.at(i)]);
					for (j = 1; j < layers.size(); ++j) {
						layers[j]->forward(layers[j - 1]->output());
					}

					//loss
					loss_function->diff(layers.back()->getC_Z(), layers.back()->output(), target[randind.at(i)]);

					//back propagation
					for (m = layers.size() - 1; m > 0; --m) {
						layers[m]->back_propagation(layers[m - 1], _learning_rate);
					}
					layers[0]->back_propagation(input[randind.at(i)], _learning_rate);
				}
				std::cout << "epoch " << k << " done\n" << std::endl;
			}
		}

		na::ndArray<float> SequentialModel::predict(const na::ndArray<float>& input)
		{

			//수정 필요
			auto input_shp = input.shape();
			assert(input_shp[1] == layers[0]->input_size);

			size_t i, j;
			std::vector<float> result;
			
			for (i = 0; i < input_shp[0]; ++i) {
				//forwarding
				layers[0]->forward(input[i]);
				for (j = 1; j < layers.size(); ++j) {
					layers[j]->forward(layers[j - 1]->output());
				}
				result.push_back(float(layers.back()->output().argmax()));
			}
				
			return na::array(result);
		}

		float SequentialModel::score(const na::ndArray<float>& input, const na::ndArray<float>& target)
		{
			size_t instance_size = input.shape()[0];
			assert(instance_size == target.shape()[0]);

			auto pred = predict(input);

			size_t cnt = 0;
			
			for (size_t i = instance_size - 1; ; --i) {
				if (pred.at(i) == target.at(i)) ++cnt;
				if (i == 0) break;
			}

			return (float)cnt / instance_size;
		}

	}
} 
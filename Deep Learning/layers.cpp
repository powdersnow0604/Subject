#include "layers.h"



namespace DeepLearning {
	namespace Layers {

		Dense::Dense(const size_t _layer_size, const size_t _input_size, const std::string& _activation_function):
			weights(na::random::uniform({ layer_size, input_size }, -0.2, 0.2)), bias(na::random::uniform({ layer_size }, -0.2, 0.2))
		{
			layer_size = _layer_size;
			input_size = _input_size;

			Z.alloc({ layer_size });
			A.alloc({ layer_size });
			C_Z.alloc({ layer_size });

			if (_activation_function == "sigmoid") {
				activation = new sigmoid();
			}
			else if (_activation_function == "softmax") {
				activation = new softmax();
			}
			else {
				activation = new sigmoid();
			}
		}

		void Dense::forward(const na::ndArray<float>& input)
		{
			size_t i;
			for (i = layer_size - 1; i != 0; --i) {
				Z.at(i) = weights[i].dot(input);
			}
			Z.at(0) = weights[0].dot(input);

			Z += bias;

			activation->forward(A, Z);
		}
		
		void Dense::back_propagation(const __basic_layer* _prev, double learning_rate)
		{
			size_t i, j;
			Dense* prev = (Dense*)_prev;

			prev->C_Z.copy(weights[0] * C_Z.at(0));
			for (i = layer_size - 1; i != 0; --i) {
				prev->C_Z += weights[i] * C_Z.at(i);
			}

			activation->diff(prev->C_Z, prev->A);

			for (i = layer_size - 1; i != 0; --i) {
				weights[i] -= prev->A * C_Z.at(i) * learning_rate;
			}
			weights[0] -= prev->A * C_Z.at(0) * learning_rate;

			bias -= C_Z * learning_rate;
		}

		Dense::~Dense() noexcept
		{
			delete activation;
		}
	}
};
#ifndef __LAYERS_H__
#define __LAYERS_H__

#include "ndArray.hpp"
#include "string"
#include "activation_function.h"
#include "loss_function.h"


namespace DeepLearning {
	namespace Layers {
		using namespace ActivationFunction;
		using namespace LossFunction;

		class __basic_layer {
		public:
			//수정 필요
			const size_t layer_size;
			const size_t input_size;
			virtual void forward(const na::ndArray<float>& input) = 0;
			virtual void back_propagation(const __basic_layer* _prev, float learning_rate) = 0;
			virtual void back_propagation(const na::ndArray<float>& input, float learning_rate) = 0;
			virtual const na::ndArray<float>& output() = 0;
			virtual na::ndArray<float>& getC_Z() = 0;

			__basic_layer(size_t _layer_size, size_t _input_size) : layer_size(_layer_size), input_size(_input_size) {}
		};


		class Dense : public __basic_layer {

			na::ndArray<float> weights;
			na::ndArray<float> bias;
			na::ndArray<float> Z;
			na::ndArray<float> A;
			na::ndArray<float> C_Z;

			__basic_activation_func* activation;

		public:
			
			Dense(const size_t _layer_size, const size_t _input_size, const std::string& _activation_function = "sigmoid");
			void forward(const na::ndArray<float>&  input);
			void back_propagation(const __basic_layer* _prev, float learning_rate);
			void back_propagation(const na::ndArray<float>& input, float learning_rate);
			const na::ndArray<float>& output() { return A; }
			na::ndArray<float>& getC_Z() { return C_Z; }
			~Dense() noexcept;
		};

	}
};


#endif


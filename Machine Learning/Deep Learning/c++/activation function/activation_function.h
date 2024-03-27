#ifndef __ACTIVATION_FUNCTION_H__
#define __ACTIVATION_FUNCTION_H__

#include "ndArray.hpp"

namespace DeepLearning {
	namespace ActivationFunction {

		class __basic_activation_func {
		public:
			virtual void forward(na::ndArray<float>& dst, const na::ndArray<float>& src) = 0;
			virtual void diff(na::ndArray<float>& dst, const na::ndArray<float>& src) = 0;
		};

		class sigmoid : public __basic_activation_func {
		public:
			void forward(na::ndArray<float>& dst, const na::ndArray<float>& src);
			void diff(na::ndArray<float>& dst, const na::ndArray<float>& src);
		};

		class softmax : public __basic_activation_func {
			void forward(na::ndArray<float>& dst, const na::ndArray<float>& src);
			void diff(na::ndArray<float>& dst, const na::ndArray<float>& src);
		};

		class Relu : public __basic_activation_func {
			void forward(na::ndArray<float>& dst, const na::ndArray<float>& src);
			void diff(na::ndArray<float>& dst, const na::ndArray<float>& src);
		};

	}
};

#endif
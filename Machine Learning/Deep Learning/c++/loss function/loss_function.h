#ifndef __ERR0R_FUNCTION_H__
#define __ERR0R_FUNCTION_H__

#include "ndArray.hpp"

namespace DeepLearning {
	namespace LossFunction {

		class __basic_loss_func {
		public:
			virtual float forward(const na::ndArray<float>& src, const na::ndArray<float>& target) = 0;
			virtual void diff(na::ndArray<float>& dst, const na::ndArray<float>& src, const na::ndArray<float>& target) = 0;
		};

		class sparse_categorical_crossentropy : public __basic_loss_func {
		public:
			float forward(const na::ndArray<float>& src, const na::ndArray<float>& target);
			void diff(na::ndArray<float>& dst, const na::ndArray<float>& src, const na::ndArray<float>& target);
		};

		float L2_loss(const na::ndArray<float>& pred, const na::ndArray<float>& target);

		na::ndArray<float> L2_loss_diff(const na::ndArray<float>& pred, const na::ndArray<float>& target);
	}
}



#endif
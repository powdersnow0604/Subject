#ifndef __ERR0R_FUNCTION_H__
#define __ERR0R_FUNCTION_H__

#include "ndArray.hpp"

namespace DeepLearning {
	namespace LossFunction {
		float L2_loss(const na::ndArray<float>& pred, const na::ndArray<float>& target);

		na::ndArray<float> L2_loss_diff(const na::ndArray<float>& pred, const na::ndArray<float>& target);
	}
}



#endif
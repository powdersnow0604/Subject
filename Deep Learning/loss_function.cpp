#include "loss_function.h"


namespace DeepLearning {
	namespace LossFunction {
		float L2_loss(const na::ndArray<float>& pred, const na::ndArray<float>& target)
		{
			float sum = (pred.at(0) * target.at(0)) * (pred.at(0) * target.at(0));
			for (size_t i = pred.raw_shape().back() - 1; i != 0; --i) {
				sum += (pred.at(i) * target.at(i)) * (pred.at(i) * target.at(i));
			}

			return sum;
		}

		na::ndArray<float> L2_loss_diff(const na::ndArray<float>& pred, const na::ndArray<float>& target)
		{
			return (pred - target) * 2;
		}
	}
}
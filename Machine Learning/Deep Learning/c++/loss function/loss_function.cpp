#include "loss_function.h"
#include <cmath>
#include <iostream>

namespace DeepLearning {
	namespace LossFunction {

		float sparse_categorical_crossentropy::forward(const na::ndArray<float>& src, const na::ndArray<float>& target)
		{
			size_t cls = (size_t)target.at(0);
			return -std::log(src.at(cls));
		}

		void sparse_categorical_crossentropy::diff(na::ndArray<float>& dst, const na::ndArray<float>& src, const na::ndArray<float>& target)
		{
			static size_t cnt = 0;
			++cnt;
			assert(dst.total_size() == src.total_size());
			size_t cls = (size_t)target.at(0);
			for (size_t i = src.total_size() - 1; ; --i) {
				if (i == cls) {
					dst.at(i) = src.at(i) - 1.0f;
				}
				else {
					dst.at(i) = src.at(i);
				}

				if (i == 0) break;
			}
		}



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
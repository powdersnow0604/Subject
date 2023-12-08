#include "activation_function.h"
#include <cmath>
#include <cassert>

namespace DeepLearning {
	namespace ActivationFunction {

		void sigmoid::forward(na::ndArray<float>& dst, const na::ndArray<float>& src)
		{
			assert(dst.total_size() == src.total_size());

			for (size_t i = dst.total_size() - 1; i != 0; --i) {
				dst.at(i) = 1 / (1 + std::exp(-src.at(i)));
			}
			dst.at(0) = 1 / (1 + std::exp(-src.at(0)));
		}

		void sigmoid::diff(na::ndArray<float>& dst, const na::ndArray<float>& src)
		{
			assert(dst.total_size() == src.total_size());

			for (size_t i = dst.total_size() - 1; i != 0; --i) {
				dst.at(i) *= src.at(i) * (1 - src.at(i));
			}
			dst.at(0) *= src.at(0) * (1 - src.at(0));
		}

		void softmax::forward(na::ndArray<float>& dst, const na::ndArray<float>& src)
		{
			assert(dst.total_size() == src.total_size());

			dst.at(0) = std::exp(src.at(0));
			float sum = dst.at(0);
			for (size_t i = src.total_size() - 1; i != 0; --i) {
				dst.at(i) = std::exp(src.at(i));
				sum += dst.at(0);
			}

			dst /= sum;
		}

		void softmax::diff(na::ndArray<float>& dst, const na::ndArray<float>& src)
		{
			assert(dst.total_size() == src.total_size());

			for (size_t i = dst.total_size() - 1; i != 0; --i) {
				dst.at(i) *= src.at(i) * (1 - src.at(i));
			}
			dst.at(0) *= src.at(0) * (1 - src.at(0));
		}

	}
};
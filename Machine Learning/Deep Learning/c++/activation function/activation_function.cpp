#include "activation_function.h"
#include <cmath>
#include <cassert>
#include <iostream>

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
				sum += dst.at(i);
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

		void Relu::forward(na::ndArray<float>& dst, const na::ndArray<float>& src)
		{
			assert(dst.total_size() == src.total_size());

			dst.at(0) = src.at(0) > 0 ? src.at(0) : 0;
			for (size_t i = dst.total_size() - 1; i != 0; --i) {
				dst.at(i) = src.at(i) > 0 ? src.at(i) : 0;
			}
		}

		void Relu::diff(na::ndArray<float>& dst, const na::ndArray<float>& src)
		{
			assert(dst.total_size() == src.total_size());

			dst.at(0) *= src.at(0) > 0 ? 1 : 0;
			for (size_t i = dst.total_size() - 1; i != 0; --i) {
				dst.at(i) *= src.at(i) > 0 ? 1 : 0;
			}
		}

	}
};
#ifndef __NDARRAY_RANDOM_HPP__
#define __NDARRAY_RANDOM_HPP__

namespace na {
	namespace random {
		/////////////////////////////////////////////////////////////////////		declaration		///////////////////////////////////////////////////////////////////

		template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
		ndArray<T> uniform(std::initializer_list<size_t> shape, const T s = 0., const T e = 1.);

		/////////////////////////////////////////////////////////////////////		definition		///////////////////////////////////////////////////////////////////

		template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> >
		ndArray<T> uniform(std::initializer_list<size_t> shape, const T s, const T e)
		{
			assert(shape.size() != 0);

			std::mt19937 gen{ std::random_device()() };
			std::conditional_t<std::is_floating_point_v<T>, std::uniform_real_distribution<T>,
				std::uniform_int_distribution<T>> dist{ s, e };

			ndArray<T> res;
			res.alloc(shape);

			size_t size = res.raw_shape().back();
			T* data = (T*)res.data();
			for (size_t i = size - 1; i != 0; --i) {
				data[i] = dist(gen);
			}
			data[0] = dist(gen);

			return res;
		}
	}
}

#endif
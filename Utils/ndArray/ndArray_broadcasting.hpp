#ifndef __NDARRAY_BROADCASTING_HPP__
#define __NDARRAY_BROADCASTING_HPP__
#include <iostream>

namespace na {

	struct bad_broadcasting {};


	template <typename T>
	class broadcast {
		const ndArray<T>& arr;
		size_t dim;
		size_t target_dim_size; //maybe redundant
		size_t dim_size; // maybe redundant
		size_t dim_below_size;
		__ndArray_shape_view target_shape;

	public:
		static constexpr bool is_leaf = false;

		broadcast(const ndArray<T>& __ndarray) : arr(__ndarray), dim(0), target_dim_size(0), dim_size(0), dim_below_size(0) {}

		template<typename E>
		void shape_resolution(const ndArrayExpression<E>& other)
		{
			const __ndArray_shape& shp = arr.raw_shape();
			const __ndArray_shape& other_shape = other.raw_shape();
			target_shape.init(other.raw_shape());

			if (other_shape.size() == shp.size()) {
				size_t i = 0;
				for (; i != shp.size(); ++i) {
					if (shp[i] != other_shape[i]) break;
				}

				if (i == shp.size()) {
					dim = 0;
					return;
				}

				dim = i++;

				for (; i != shp.size(); ++i) {
					assert(other_shape[i] / other_shape[i - 1] == shp[i] / shp[i - 1]);
				}

				target_dim_size = other_shape[dim];
				dim_size = shp[dim];
				dim_below_size = shp[dim - 1];
			}
			else if (other_shape.size() - shp.size() == 1) {
				for (size_t i = shp.size() - 1; i != 0; --i) {
					assert(other_shape[i] == shp[i]);
				}
				dim = shp.size();
				target_dim_size = other_shape[dim];
				dim_size = dim_below_size = shp.back();
			}
			else {
				throw bad_broadcasting();
			}

		}

		// no boundary check; member functions are not intended to be used by user
		T at(size_t i) const
		{
			if (dim == 0) return arr.item[i];

			size_t index = i / target_dim_size;
			size_t sub_index = (i - index * target_dim_size) % dim_size;
			return arr.item[index * dim_size + sub_index];
		}

		na::ndArray<T> operator[](size_t i) const
		{
			if (dim == 0) return arr[i];

			na::ndArray<T> res;
			res.alloc(target_shape.dim_decreased(target_shape.size() - 1));

			if (dim == target_shape.size() - 1) {
				size_t dim_elements = dim_size / dim_below_size;
				size_t index = (i % dim_elements) * dim_below_size;
				for (size_t j = 0; j < res.total_size(); ++j) {
					res.item[j] = arr.item[index + j];
				}
			}
			else {
				//size_t dim_elements = dim_size / dim_below_size;
				size_t offset = i * arr._shape[arr.dim() - 1];
				for (size_t j = 0, arr_j = 0; j < res.total_size(); j += target_dim_size, arr_j += dim_size) {

					for (size_t k = 0; k < target_dim_size; k += dim_below_size) {
						size_t index = k % dim_size;
						for (size_t l = 0; l < dim_below_size; ++l) {
							res.item[j + k + l] = arr.item[offset + arr_j + index + l];
						}
					}
					/*
					for (size_t k = 0; k < res.dim_size(dim); ++k) {
						size_t index = (k % dim_elements) * dim_below_size;
						for (size_t l = 0; l < dim_below_size; ++l) {
							res.item[j + k * dim_below_size + l] = arr.item[offset + arr_j + index + l];
						}
					}
					*/
				}
			}

			
			
			return res;
		}
		

		const __ndArray_shape_view& raw_shape() const { return target_shape; }
	};



#pragma region support_broadcasting_arithmetic_operator
	template <typename E1, typename E2>
	ndArraySum<E1, broadcast<E2>>
		operator+(ndArrayExpression<E1> const& u, broadcast<E2>&& v) {
		v.shape_resolution(u);
		return ndArraySum<E1, broadcast<E2>>(*static_cast<const E1*>(&u), v);
	}

	template <typename E1, typename E2>
	ndArrayMul<E1, broadcast<E2>>
		operator*(ndArrayExpression<E1> const& u, broadcast<E2> const& v) {
		v.shape_resolution(u);
		return ndArrayMul<E1, broadcast<E2>>(*static_cast<const E1*>(&u), v);
	}

	template <typename E1, typename E2>
	ndArraySub<E1, broadcast<E2>>
		operator-(ndArrayExpression<E1> const& u, broadcast<E2> const& v) {
		v.shape_resolution(u);
		return ndArraySub<E1, broadcast<E2>>(*static_cast<const E1*>(&u), v);
	}

	template <typename E1, typename E2>
	ndArrayDiv<E1, broadcast<E2>>
		operator/(ndArrayExpression<E1> const& u, broadcast<E2> const& v) {
		v.shape_resolution(u);
		return ndArrayDiv<E1, broadcast<E2>>(*static_cast<const E1*>(&u), v);
	}

	template <typename E1, typename E2>
	ndArraySum<broadcast<E1>, E2>
		operator+(broadcast<E1>&& u, ndArrayExpression<E2> const& v) {
		u.shape_resolution(v);
		return ndArraySum<broadcast<E1>, E2>(u, *static_cast<const E2*>(&v));
	}

	template <typename E1, typename E2>
	ndArrayMul<broadcast<E1>, E2>
		operator*(broadcast<E1>&& u, ndArrayExpression<E2> const& v) {
		u.shape_resolution(v);
		return ndArrayMul<broadcast<E1>, E2>(u, *static_cast<const E2*>(&v));
	}

	template <typename E1, typename E2>
	ndArraySub<broadcast<E1>, E2>
		operator-(broadcast<E1>&& u, ndArrayExpression<E2> const& v) {
		u.shape_resolution(v);
		return ndArraySub<broadcast<E1>, E2>(u, *static_cast<const E2*>(&v));
	}

	template <typename E1, typename E2>
	ndArrayDiv<broadcast<E1>, E2>
		operator/(broadcast<E1>&& u, ndArrayExpression<E2> const& v) {
		u.shape_resolution(v);
		return ndArrayDiv<broadcast<E1>, E2>(u, *static_cast<const E2*>(&v));
	}
	
	/*
	template <typename T>
	template <typename E>
	ndArray<T>& ndArray<T>::operator+= (const ndArrayExpression<E>& other)
	{
		assert(_shape == other.raw_shape());
		for (size_t i = _shape.back() - 1; i != 0; --i) {
			item[i] = static_cast<T>(item[i] + other.at(i));
		}
		item[0] = static_cast<T>(item[0] + other.at(0));

		return *this;
	}

	template <typename T>
	template <typename E>
	ndArray<T>& ndArray<T>::operator-= (const ndArrayExpression<E>& other)
	{
		assert(_shape == other.raw_shape());
		for (size_t i = _shape.back() - 1; i != 0; --i) {
			item[i] = static_cast<T>(item[i] - other.at(i));
		}
		item[0] = static_cast<T>(item[0] - other.at(0));

		return *this;
	}

	template <typename T>
	template <typename E>
	ndArray<T>& ndArray<T>::operator*= (const ndArrayExpression<E>& other)
	{
		assert(_shape == other.raw_shape());
		for (size_t i = _shape.back() - 1; i != 0; --i) {
			item[i] = static_cast<T>(item[i] * other.at(i));
		}
		item[0] = static_cast<T>(item[0] * other.at(0));

		return *this;
	}

	template <typename T>
	template <typename E>
	ndArray<T>& ndArray<T>::operator/= (const ndArrayExpression<E>& other)
	{
		assert(_shape == other.raw_shape());
		for (size_t i = _shape.back() - 1; i != 0; --i) {
			item[i] = static_cast<T>(item[i] / other.at(i));
		}
		item[0] = static_cast<T>(item[0] / other.at(0));

		return *this;
	}

	template <typename T>
	template <typename E, std::enable_if_t<std::is_arithmetic_v<E>, bool>>
	ndArray<T>& ndArray<T>::operator+= (const E scalar)
	{
		for (size_t i = _shape.back() - 1; i != 0; --i) {
			item[i] = static_cast<T>(item[i] + scalar);
		}
		item[0] = static_cast<T>(item[0] + scalar);

		return *this;
	}

	template <typename T>
	template <typename E, std::enable_if_t<std::is_arithmetic_v<E>, bool>>
	ndArray<T>& ndArray<T>::operator-= (const E scalar)
	{
		for (size_t i = _shape.back() - 1; i != 0; --i) {
			item[i] = static_cast<T>(item[i] - scalar);
		}
		item[0] = static_cast<T>(item[0] - scalar);

		return *this;
	}

	template <typename T>
	template <typename E, std::enable_if_t<std::is_arithmetic_v<E>, bool>>
	ndArray<T>& ndArray<T>::operator*= (const E scalar)
	{
		for (size_t i = _shape.back() - 1; i != 0; --i) {
			item[i] = static_cast<T>(item[i] * scalar);
		}
		item[0] = static_cast<T>(item[0] * scalar);

		return *this;
	}

	template <typename T>
	template <typename E, std::enable_if_t<std::is_arithmetic_v<E>, bool>>
	ndArray<T>& ndArray<T>::operator/= (const E scalar)
	{
		for (size_t i = _shape.back() - 1; i != 0; --i) {
			item[i] = static_cast<T>(item[i] / scalar);
		}
		item[0] = static_cast<T>(item[0] / scalar);

		return *this;
	}
	*/
#pragma endregion
}

#endif
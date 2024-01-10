//c++ 17
#ifndef __NDARRAY_HPP__
#define __NDARRAY_HPP__


#include <memory>
#include <iostream>
#include <cassert>
#include <vector>
#include <random>
#include <utility>


/////////////////////////////////////////////////////////////////////		shape		///////////////////////////////////////////////////////////////////

#include "ndArray_shape.h"


/////////////////////////////////////////////////////////////////////		std::vector supporters		///////////////////////////////////////////////////////////////////

#include "ndArray_supporters_for_stdvector.hpp"

/////////////////////////////////////////////////////////////////////		allocator		///////////////////////////////////////////////////////////////////

#include "ndArray_allocator.h"


namespace na {
	/////////////////////////////////////////////////////////////////////		forward declaration		///////////////////////////////////////////////////////////////////

	namespace linalg {
		template<typename U, typename V>
		class __ndArray_inner_base;

		template<typename U, typename V>
		class __ndArray_outer_base;
	}

	/////////////////////////////////////////////////////////////////////		extern variables		///////////////////////////////////////////////////////////////////
	
	extern 	__ndArray_allocator __ndArray_allocator_instance;

	/////////////////////////////////////////////////////////////////////		helper class		///////////////////////////////////////////////////////////////////

	template <typename E>
	class ndArrayExpression {
	public:
		static constexpr bool is_leaf = false;
		auto operator[](size_t i) const { return static_cast<E const&>(*this)[i]; }
		auto at(size_t i) const { return static_cast<E const&>(*this).at(i); }
		//std::vector<size_t> shape() const { return static_cast<E const&>(*this).shape(); }
		const auto& raw_shape() const { return static_cast<E const&>(*this).raw_shape(); }
		
		template <typename U>
		operator U() const {
			assert(this->raw_shape().size() == 1);
			return this->at(0);
		}
		
	};



	template <typename E>
	class ndArrayScalarExpression : public ndArrayExpression<ndArrayScalarExpression<E>> {
		const E u;
	public:
		static constexpr bool is_leaf = true;
		ndArrayScalarExpression(const E& _u) : u(_u) {}
		E operator[](size_t i) const { return u; }
		operator E() { return u; }
	};

	

	/////////////////////////////////////////////////////////////////////		ndArray		///////////////////////////////////////////////////////////////////
	template <typename T>
	class ndArray : public ndArrayExpression<ndArray<T>> {
		T* item;
		T* original;
		size_t* ref_cnt;
		__ndArray_shape _shape;

		ndArray(T* _item, T* _original, size_t* _ref_cnt, const __ndArray_shape& shp, size_t diff_dim = 1) :item(_item), original(_original), ref_cnt(_ref_cnt), _shape(shp, 0, shp.size() - diff_dim)
		{
			++(*ref_cnt);
		}
		void _memcpy(void* dst, void* src, size_t size) const;

	public:
		template <typename E>
		friend class broadcast;

		template <typename E>
		friend class ndArray;

		static constexpr bool is_leaf = true;

		//functions
		template <typename E>
		ndArray(ndArrayExpression<E> const& expr);
		
		ndArray(const ndArray<T>& other) : item(other.item), original(other.original), ref_cnt(other.ref_cnt), _shape(other._shape) { if (ref_cnt) ++(*ref_cnt); };

		ndArray() : item(nullptr), original(nullptr), ref_cnt(nullptr) {}

		~ndArray() noexcept;

		T& at(size_t i) { return item[i]; }

		T& at_s(size_t i) { assert(i < _shape.back()); return item[i]; }

		const T& at(size_t i) const { return item[i]; }

		const T& at_s(size_t i) const { assert(i < _shape.back()); return item[i]; }

		ndArray<T> access(size_t i) const;

		ndArray<T> access(std::initializer_list<size_t> list) const;

		const __ndArray_shape& raw_shape() const { return _shape; }

		std::vector<size_t> shape() const;

		size_t dim() const { return _shape.size() - 1; }

		ndArray<T>& shrink_dim() { _shape.shrink_dim_to_fit(); return *this; }

		ndArray<T>& alloc(std::initializer_list<size_t> list);

		ndArray<T>& alloc(const std::vector<size_t>& vec);

		ndArray<T>& alloc(const __ndArray_shape& vec);

		template<typename E>
		linalg::__ndArray_inner_base<ndArray<T>, E> dot(const ndArrayExpression<E>& other);

		template<typename E>
		linalg::__ndArray_outer_base<ndArray<T>, E> outer(const ndArrayExpression<E>& other);

		ndArray<T> copy() const;

		template <typename E>
		ndArray<T>& copy(const ndArray<E>& other);

		template <typename E>
		ndArray<T>& copy(const ndArrayExpression<E>& other);

		ndArray<T>& copy(const T* other);

		template <typename E>
		ndArray<T>& shallow_copy(const ndArray<E>& other);

		ndArray<T>& reshape(std::initializer_list<size_t> list);

		const T* data() const { return item; }

		size_t total_size() const { return _shape.back(); }

		size_t dim_size(size_t i) const { assert(i < _shape.size()); return _shape[i] / _shape[i - 1]; }

		ndArray<T> sum(size_t dim = 1) const;

		ndArray<T>& square();

		ndArray<size_t> argmax(size_t dim = 1) const;

		ndArray<size_t> argmin(size_t dim = 1) const;

		ndArray<T>& shuffle(size_t dim = 1, bool synchronize = false);
		
		ndArray<T>& transpose();

		#pragma region ndArray_operators
		ndArray<T> operator[](size_t i) const;

		ndArray<T> operator[](std::initializer_list<size_t> list) const;

		ndArray<T>& operator=(const ndArray<T>& other);

		template <typename E>
		ndArray<T>& operator=(const ndArrayExpression<E>& other);

		template<typename E>
		T& operator=(const E v) { assert(_shape.size() == 1); *item = v; return *item; }

		//operator T& () { assert(_shape.size() == 1); return *item; }

		template <typename E>
		ndArray<T>& operator+= (const ndArrayExpression<E>& other);

		template <typename E>
		ndArray<T>& operator-= (const ndArrayExpression<E>& other);

		template <typename E>
		ndArray<T>& operator*= (const ndArrayExpression<E>& other);

		template <typename E>
		ndArray<T>& operator/= (const ndArrayExpression<E>& other);

		template <typename E>
		ndArray<T>& operator+= (const E scalar);

		template <typename E>
		ndArray<T>& operator-= (const E scalar);

		template <typename E>
		ndArray<T>& operator*= (const E scalar);

		template <typename E>
		ndArray<T>& operator/= (const E scalar);

		#pragma endregion
	};

	/////////////////////////////////////////////////////////////////////		member fuction		///////////////////////////////////////////////////////////////////

	template <typename T>
	template <typename E>
	ndArray<T>::ndArray(ndArrayExpression<E> const& expr):_shape(expr.raw_shape()) {
		//item = original = (T*)malloc(sizeof(T) * _shape.back() + sizeof(size_t));
		item = original = (T*)__ndArray_allocator_instance.allocate(sizeof(T) * _shape.back() + sizeof(size_t));
		assert(item != nullptr);
		ref_cnt = (size_t*)(item + _shape.back());
		*ref_cnt = 1;

		for (size_t i = _shape.back() - 1; i != 0; --i) {
			item[i] = expr.at(i);
		}
		item[0] = expr.at(0);
	}

	template <typename T>
	ndArray<T>::~ndArray() noexcept
	{
		if (original != nullptr && --(*ref_cnt) == 0) {
			//free(original);
			__ndArray_allocator_instance.deallocate(original, _shape.back());
		}
	}

	template <typename T>
	ndArray<T> ndArray<T>::operator[](size_t i) const
	{
		assert(_shape.size() != 1);
		return ndArray<T>(item + i * _shape[_shape.size() - 2], original, ref_cnt, _shape);
	}

	template <typename T>
	ndArray<T> ndArray<T>::operator[](std::initializer_list<size_t> list) const
	{
		assert(list.size() != 0);
		assert(_shape.size() > list.size());

		auto list_i = list.begin();
		size_t shp_i = _shape.size() - 2;
		size_t offset = *list_i++ * _shape[shp_i--];

		for (; list_i != list.end(); ++list_i, --shp_i) {
			offset += *list_i * _shape[shp_i];
		}
		
		return ndArray<T>(item + offset, original, ref_cnt, _shape, list.size());
	}

	template <typename T>
	ndArray<T> ndArray<T>::access(size_t i) const
	{
		assert(_shape.size() != 1);
		assert(i < _shape.back());
		return ndArray<T>(item + i * _shape[_shape.size() - 2], original, ref_cnt, _shape);
	}

	template <typename T>
	ndArray<T> ndArray<T>::access(std::initializer_list<size_t> list) const
	{
		assert(list.size() != 0);
		assert(_shape.size() > list.size());

		auto list_i = list.begin();
		size_t shp_i = _shape.size() - 2;
		assert(*list_i < _shape[shp_i] / _shape[shp_i - 1]);
		size_t offset = *list_i++ * _shape[shp_i--];

		for (; list_i != list.end(); ++list_i, --shp_i) {
			assert(*list_i < _shape[shp_i] / _shape[shp_i - 1]);
			offset += *list_i * _shape[shp_i];
		}

		return ndArray<T>(item + offset, original, ref_cnt, _shape, list.size());
	}

	template <typename T>
	ndArray<T>& ndArray<T>::operator=(const ndArray<T>& other)
	{
		assert(_shape == other._shape);

		_memcpy(item, other.item, _shape.back());

		return *this;
	}

	template <typename T>
	template <typename E>
	ndArray<T>& ndArray<T>::operator=(const ndArrayExpression<E>& other)
	{
		assert(_shape == other.raw_shape());

		for (size_t i = _shape.back() - 1; i != 0; --i) {
			item[i] = other.at(i);
		}
		item[0] = other.at(0);

		return *this;
	}

	template <typename T>
	std::vector<size_t> ndArray<T>::shape() const
	{
		std::vector<size_t> shp; shp.reserve(_shape.size() - 1);
		for (size_t i = _shape.size() - 1; i != 0; --i) {
			shp.push_back(_shape[i] / _shape[i - 1]);
		}

		return shp;
	}

	template <typename T>
	ndArray<T>& ndArray<T>::alloc(std::initializer_list<size_t> list)
	{
		assert(list.size() != 0);
		if (original != nullptr && --(*ref_cnt) == 0) {
			free(original);
		}

		_shape.init(list);

		//item = original = (T*)malloc(sizeof(T) * _shape.back() + sizeof(size_t));
		item = original = (T*)__ndArray_allocator_instance.allocate(sizeof(T) * _shape.back() + sizeof(size_t));
		assert(item != nullptr);
		ref_cnt = (size_t*)(item + _shape.back());
		*ref_cnt = 1;

		return *this;
	}

	template <typename T>
	ndArray<T>& ndArray<T>::alloc(const std::vector<size_t>& vec)
	{
		assert(vec.size() != 0);
		if (original != nullptr && --(*ref_cnt) == 0) {
			free(original);
		}

		_shape.init(vec);

		//item = original = (T*)malloc(sizeof(T) * _shape.back() + sizeof(size_t));
		item = original = (T*)__ndArray_allocator_instance.allocate(sizeof(T) * _shape.back() + sizeof(size_t));
		assert(item != nullptr);
		ref_cnt = (size_t*)(item + _shape.back());
		*ref_cnt = 1;

		return *this;
	}

	template <typename T>
	ndArray<T>& ndArray<T>::alloc(const __ndArray_shape& shp)
	{
		if (original != nullptr && --(*ref_cnt) == 0) {
			free(original);
		}

		_shape = shp;

		//item = original = (T*)malloc(sizeof(T) * _shape.back() + sizeof(size_t));
		item = original = (T*)__ndArray_allocator_instance.allocate(sizeof(T) * _shape.back() + sizeof(size_t));
		assert(item != nullptr);
		ref_cnt = (size_t*)(item + _shape.back());
		*ref_cnt = 1;

		return *this;
	}

	template <typename T>
	ndArray<T> ndArray<T>::copy() const
	{
		ndArray<T> cpy;
		cpy.alloc(_shape);

		_memcpy(cpy.item, item, _shape.back());

		return cpy;
	}

	template <typename T>
	template <typename E>
	ndArray<T>& ndArray<T>::copy(const ndArray<E>& other)
	{
		assert(_shape == other._shape);

		for (size_t i = _shape.back() - 1; i != 0; --i) {
			item[i] = other.at(i);
		}
		item[0] = other.at(0);

		return *this;
	}

	template <typename T>
	template <typename E>
	ndArray<T>& ndArray<T>::copy(const ndArrayExpression<E>& other)
	{
		assert(_shape == other.raw_shape());

		for (size_t i = _shape.back() - 1; i != 0; --i) {
			item[i] = other.at(i);
		}
		item[0] = other.at(0);

		return *this;
	}

	template <typename T>
	ndArray<T>& ndArray<T>::copy(const T* other)
	{
		_memcpy(item, (void*)other, _shape.back());
		return *this;
	}

	template <typename T>
	template <typename E>
	ndArray<T>& ndArray<T>::shallow_copy(const ndArray<E>& other)
	{
		if (original != nullptr && --(*ref_cnt) == 0) {
			//free(original);
			__ndArray_allocator_instance.deallocate(original, _shape.back());
		}

		item = other.item;
		original = other.original;
		_shape = other._shape;

		return *this;
	}

	template <typename T>
	void ndArray<T>::_memcpy(void* dst, void* src, size_t size) const
	{
		size *= sizeof(T);
		size_t i = 0;
		for (; i < size >> 3; ++i) {
			(static_cast<size_t*>(dst))[i] = (static_cast<size_t*>(src))[i];
		}

		dst = static_cast<size_t*>(dst) + i;
		src = static_cast<size_t*>(src) + i;


		for (i = 0; i < ((size & 4) >> 2); ++i) {
			(static_cast<unsigned int*>(dst))[i] = (static_cast<unsigned int*>(src))[i];
		}

		dst = static_cast<unsigned int*>(dst) + i;
		src = static_cast<unsigned int*>(src) + i;

		for (i = 0; i < (size & 3); ++i) {
			(static_cast<unsigned char*>(dst))[i] = (static_cast<unsigned char*>(src))[i];
		}
	}

	template <typename T>
	ndArray<T>& ndArray<T>::reshape(std::initializer_list<size_t> list)
	{
		_shape.reshape(list);

		return *this;
	}

	template <typename T>
	ndArray<T> ndArray<T>::sum(size_t dim) const
	{
		assert(0 < dim && dim < _shape.size());

		size_t i, j, k;
		size_t curr_i;

		ndArray<T> res;
		res.alloc(_shape.dim_erased(dim));
		
		for (i = 0, curr_i = 0; i < res._shape.back(); i += _shape[dim-1], curr_i += _shape[dim]) {
			for (k = 0; k < _shape[dim - 1]; ++k) {
				res.item[i + k] = item[curr_i + k];
			}

			for (j = _shape[dim-1]; j < _shape[dim]; j += _shape[dim-1]) {
				for (k = 0; k < _shape[dim - 1]; ++k) {
					res.item[i + k] += item[curr_i + j + k];
				}
			}
		}

		return res;
	}

	template<typename T>
	ndArray<T>& ndArray<T>::square()
	{
		for (size_t i = _shape.back() - 1; i != 0; --i) {
			item[i] *= item[i];
		}
		item[0] *= item[0];
		
		return *this;
	}

	template<typename T>
	ndArray<size_t> ndArray<T>::argmax(size_t dim) const
	{
		assert(0 < dim && dim < _shape.size());
		size_t i, j, k;
		size_t curr_i;

		ndArray<size_t> res;
		res.alloc(_shape.dim_erased(dim));
		std::vector<T> temp(_shape[dim - 1]);

		for (i = 0, curr_i = 0; i < res._shape.back(); i += _shape[dim - 1], curr_i += _shape[dim]) {

			for (k = 0; k < _shape[dim - 1]; ++k) {
				temp[k] = item[curr_i + k];
			}

			for (j = _shape[dim - 1]; j < _shape[dim]; j += _shape[dim - 1]) {
				for (k = 0; k < _shape[dim - 1]; ++k) {
					if (temp[k] < item[curr_i + j + k]) {
						res.item[i + k] = j / _shape[dim - 1];
						temp[k] = item[curr_i + j + k];
					}
				}
			}
		}

		return res;
	}

	template<typename T>
	ndArray<size_t> ndArray<T>::argmin(size_t dim) const
	{
		assert(0 < dim && dim < _shape.size());
		size_t i, j, k;
		size_t curr_i;

		ndArray<size_t> res;
		res.alloc(_shape.dim_erased(dim));
		std::vector<T> temp(_shape[dim - 1]);

		for (i = 0, curr_i = 0; i < res._shape.back(); i += _shape[dim - 1], curr_i += _shape[dim]) {

			for (k = 0; k < _shape[dim - 1]; ++k) {
				temp[k] = item[curr_i + k];
			}

			for (j = _shape[dim - 1]; j < _shape[dim]; j += _shape[dim - 1]) {
				for (k = 0; k < _shape[dim - 1]; ++k) {
					if (temp[k] > item[curr_i + j + k]) {
						res.item[i + k] = j / _shape[dim - 1];
						temp[k] = item[curr_i + j + k];
					}
				}
			}
		}

		return res;
	}

	template <typename T>
	ndArray<T>& ndArray<T>::shuffle(size_t dim, bool synchronize)
	{
		size_t i, j, temp_t;
		size_t dim_size = _shape[dim] / _shape[dim - 1];
		std::mt19937 gen{ std::random_device()() };
		std::uniform_int_distribution<size_t> dist;
		std::uniform_int<size_t>::param_type params;
		size_t randnum;

		T* temp = (T*)malloc(sizeof(T) * _shape[dim]);
		std::vector<size_t> randind; randind.reserve(dim_size);

		for (i = dim_size - 1; i != 0; --i) {
			randind.push_back(i);
		}
		randind.push_back(0);

		if (synchronize) {
			for (j = dim_size - 1; j != 0; --j) {
				params._Init(0, j);
				randnum = dist(gen, params);
				temp_t = randind[j];
				randind[j] = randind[randnum];
				randind[randnum] = temp_t;
			}
		}


		for (i = 0; i < _shape.back(); i += _shape[dim]) {

			if (!synchronize) {
				for (j = dim_size - 1; j != 0; --j) {
					params._Init(0, j);
					randnum = dist(gen, params);
					temp_t = randind[j];
					randind[j] = randind[randnum];
					randind[randnum] = temp_t;
				}
			}

			for (j = dim_size - 1; ; --j) {
				_memcpy(temp + j * _shape[dim - 1], item + i + randind[j] * _shape[dim - 1], _shape[dim - 1]);
				if (j == 0) break;
			}
			_memcpy(item + i, temp, _shape[dim]);

		}

		free(temp);

		return *this;
	}


	/////////////////////////////////////////////////////////////////////		declaration		///////////////////////////////////////////////////////////////////


	template <typename T>
	void __support_array_func(const std::vector<T>& vec, __supporter_vector_element_type_v<std::vector<T>>* item, const __ndArray_shape& shp);

	template<typename E>
	ndArray<__supporter_vector_element_type_v<std::vector<E>>> array(const std::vector<E>& vec);

	template<typename T>
	void __support_ndArray_print(T* item, const std::vector<size_t>& shp, const __ndArray_shape& raw_shp, size_t dim, size_t highest_dim);

	template<typename T>
	void __support_ndArrayExpression_print(const ndArrayExpression<T>& item, size_t offset, 
		const std::vector<size_t>& shp, const __ndArray_shape& raw_shp, size_t dim, size_t highest_dim);

	template<typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
	ndArray<T> range(T start, const T end, const T interval = 1);

	template<typename T = double>
	ndArray<T> homogeneous(std::initializer_list<size_t> shp, T val = 0.);

	/////////////////////////////////////////////////////////////////////		definition		///////////////////////////////////////////////////////////////////

	template <typename T>
	void __support_array_func(const std::vector<T>& vec, __supporter_vector_element_type_v<std::vector<T>>* item, const __ndArray_shape& shp) {
		if constexpr (__supporter_dim_v<std::vector<T>> == 1) {
			if (std::is_arithmetic_v< __supporter_vector_element_type_v<std::vector<T>>>) {
				for (size_t i = vec.size() - 1; i != 0; --i) {
					item[i] = vec[i];
				}
				item[0] = vec.front();

				return;
			}
			else {
				const T* vec_item = vec.data();
				for (size_t i = vec.size() - 1; i != 0; --i) {
					memcpy_s(item + i, sizeof(T), vec_item + i, sizeof(T));
				}
				memcpy_s(item, sizeof(T), vec_item, sizeof(T));
			}
			for (size_t i = vec.size()-1; i != 0; --i) {
				item[i] = vec[i];
			}
			item[0] = vec.front();

			return;
		}
		else {
			for (size_t i = vec.size() - 1; i != 0; --i) {
				__support_array_func(vec[i], item + i * shp[__supporter_dim_v<std::vector<T>> -1], shp);
			}
			__support_array_func(vec[0], item, shp);
		}
	}
	
	template <typename E>
	ndArray<__supporter_vector_element_type_v<std::vector<E>>> array(const std::vector<E>& vec) {
		assert(__supporter_is_rect_vector(vec));
		std::vector<size_t> shp; shp.reserve(__supporter_dim_v<std::vector<E>>);
		__supporter_calc_shape(vec, shp);
	
		ndArray<__supporter_vector_element_type_v<std::vector<E>>> res;
		res.alloc(shp);

		__support_array_func(vec, (__supporter_vector_element_type_v<std::vector<E>>*) res.data(), res.raw_shape());

		return res;
	}

	template<typename T>
	void __support_ndArray_print(T* item, const std::vector<size_t>& shp, const __ndArray_shape& raw_shp, size_t dim, size_t highest_dim)
	{
		using std::cout;
		using std::endl;

		if (dim == 1) {
			cout << '[';
			for (size_t i = 0; ; ++i) {
				cout << item[i];
				if (i == shp.back() - 1) break;
				cout << ", ";
			}
			cout << ']';

			if (1 == highest_dim) {
				cout << endl;
				return;
			}
		}
		else {
			size_t ind = highest_dim - dim;
			std::cout << '[';
			for (size_t i = 0; i < shp[ind]; ++i) {
				if (i != 0) {
					for (size_t j = 0; j < highest_dim - dim + 1; ++j)
						std::cout << ' ';
				}

				__support_ndArray_print(item + i * raw_shp[dim-1], shp, raw_shp, dim-1, highest_dim);

				if (i != shp[ind] - 1) {
					std::cout << "," << std::endl;
					for (size_t j = 0; j < dim / 3; ++j) std::cout << std::endl;
				}

			}
			std::cout << ']';

			if (dim == highest_dim) {
				std::cout << std::endl;
				return;
			}
		}
	}

	template<typename T>
	std::ostream& operator << (std::ostream& out, const ndArray<T>& arr)
	{
		const std::vector<size_t>& shp = arr.shape();
		if (shp.size() == 0) return out << arr.at(0);

		__support_ndArray_print((T*)arr.data(), shp, arr.raw_shape(), shp.size(), shp.size());

		return out;
	}

	template<typename T>
	void __support_ndArrayExpression_print(const ndArrayExpression<T>& item, size_t offset,
		const std::vector<size_t>& shp, const __ndArray_shape& raw_shp, size_t dim, size_t highest_dim)
	{
		using std::cout;
		using std::endl;

		if (dim == 1) {
			cout << '[';
			for (size_t i = 0; ; ++i) {
				cout << item.at(offset + i);
				if (i == shp.back() - 1) break;
				cout << ", ";
			}
			cout << ']';

			if (1 == highest_dim) {
				cout << endl;
				return;
			}
		}
		else {
			size_t ind = highest_dim - dim;
			std::cout << '[';
			for (size_t i = 0; i < shp[ind]; ++i) {
				if (i != 0) {
					for (size_t j = 0; j < highest_dim - dim + 1; ++j)
						std::cout << ' ';
				}

				__support_ndArrayExpression_print(item, offset + i * raw_shp[dim - 1], shp, raw_shp, dim - 1, highest_dim);

				if (i != shp[ind] - 1) {
					std::cout << "," << std::endl;
					for (size_t j = 0; j < dim / 3; ++j) std::cout << std::endl;
				}

			}
			std::cout << ']';

			if (dim == highest_dim) {
				std::cout << std::endl;
				return;
			}
		}
	}

	template<typename T>
	std::ostream& operator << (std::ostream& out, const ndArrayExpression<T>& arr)
	{
		const std::vector<size_t>& shp = __ndArray_shape::to_vector(arr.raw_shape());
		if (shp.size() == 0) return out << arr.at(0);

		__support_ndArrayExpression_print(arr, 0, shp, arr.raw_shape(), shp.size(), shp.size());

		return out;
	}

	template<typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool>>
	ndArray<T> range(T start, const T end, const T interval)
	{
		ndArray<T> res;

		if (interval == 0 || (interval > 0 && end < start) || (interval < 0 && end > start))
			return res;

		size_t size, i = 0;
		double temp = ((double)end - start) / (double)interval;
		if (temp - (size_t)temp) ++temp;
		size = (size_t)temp;

		
		res.alloc({ size });
		T* data = (T*)res.data();

		if(start < end) {
			for (; start < end; start += interval) {
				data[i++] = start;
			}
		}
		else if (start > end) {
			for (; start > end; start += interval) {
				data[i++] = start;
			}
		}

		return res;
	}

	template<typename T>
	ndArray<T> homogeneous(std::initializer_list<size_t> shp, T val)
	{
		ndArray<T> res;
		
		res.alloc(shp);

		if (std::is_arithmetic_v<T>) {
			for (size_t i = res.total_size() - 1; i != 0; --i) {
				res.at(i) = val;
			}
			res.at(0) = val;
		}
		else {
			T* ptr = (T*)res.data();

			for (size_t i = res.total_size() - 1; i != 0; --i) {
				memcpy_s(ptr + i, sizeof(T), &val, sizeof(T));
			}
			memcpy_s(ptr, sizeof(T), &val, sizeof(T));
		}

		return res;
	}
}

/////////////////////////////////////////////////////////////////////		vectorize operation		///////////////////////////////////////////////////////////////////

#include "ndArray_vectorize_operation.hpp"

/////////////////////////////////////////////////////////////////////		random		///////////////////////////////////////////////////////////////////

#include "ndArray_random.hpp"

/////////////////////////////////////////////////////////////////////		broadcasting		///////////////////////////////////////////////////////////////////

#include "ndArray_broadcasting.hpp"

/////////////////////////////////////////////////////////////////////		linear algebra		///////////////////////////////////////////////////////////////////

#include "ndArray_linalg.hpp"


#endif
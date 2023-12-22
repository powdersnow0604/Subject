//c++ 17
#ifndef __NDARRAY_HPP__
#define __NDARRAY_HPP__


#include <memory>
#include <stdlib.h>
#include <iostream>
#include <cassert>
#include <vector>
#include <random>
#include <utility>
#include "ndArray_shape.h"


namespace na {
	/////////////////////////////////////////////////////////////////////		forward declaration		///////////////////////////////////////////////////////////////////
	

	/////////////////////////////////////////////////////////////////////		helper class		///////////////////////////////////////////////////////////////////
	template <typename E>
	class ndArrayExpression {
	public:
		static constexpr bool is_leaf = false;
		auto operator[](size_t i) const { return static_cast<E const&>(*this)[i]; }
		auto at(size_t i) const { return static_cast<E const&>(*this).at(i); }
		//std::vector<size_t> shape() const { return static_cast<E const&>(*this).shape(); }
		const __ndArray_shape& raw_shape() const { return static_cast<E const&>(*this).raw_shape(); }
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



	#pragma region support_arithmetic_operation_class
	template <typename E1, typename E2>
	class ndArraySum : public ndArrayExpression<ndArraySum<E1, E2> > {
		typename std::conditional_t<E1::is_leaf, const E1, const E1&> _u;
		typename std::conditional_t<E2::is_leaf, const E2, const E2&> _v;

	public:
		static constexpr bool is_leaf = false;
		ndArraySum(E1 const& u, E2 const& v) : _u(u), _v(v) {
			assert(u.raw_shape() == v.raw_shape());
		}
		decltype(auto) operator[](size_t i) const { return _u[i] + _v[i]; }
		decltype(auto) at(size_t i) const { return _u.at(i) + _v.at(i); }
		const __ndArray_shape& raw_shape()               const { return _u.raw_shape(); }
	};

	template <typename E1, typename E2>
	class ndArrayMul : public ndArrayExpression<ndArrayMul<E1, E2> > {
		typename std::conditional_t<E1::is_leaf, const E1, const E1&> _u;
		typename std::conditional_t<E2::is_leaf, const E2, const E2&> _v;

	public:
		static constexpr bool is_leaf = false;
		ndArrayMul(E1 const& u, E2 const& v) : _u(u), _v(v) {
			assert(u.raw_shape() == v.raw_shape());
		}
		decltype(auto) operator[](size_t i) const { return _u[i] * _v[i]; }
		decltype(auto) at(size_t i) const { return _u.at(i) * _v.at(i); }
		const __ndArray_shape& raw_shape()               const { return _u.raw_shape(); }
	};

	template <typename E1, typename E2>
	class ndArraySub : public ndArrayExpression<ndArraySub<E1, E2> > {
		typename std::conditional_t<E1::is_leaf, const E1, const E1&> _u;
		typename std::conditional_t<E2::is_leaf, const E2, const E2&> _v;

	public:
		static constexpr bool is_leaf = false;
		ndArraySub(E1 const& u, E2 const& v) : _u(u), _v(v) {
			assert(u.raw_shape() == v.raw_shape());
		}
		decltype(auto) operator[](size_t i) const { return _u[i] - _v[i]; }
		decltype(auto) at(size_t i) const { return _u.at(i) - _v.at(i); }
		const __ndArray_shape& raw_shape()               const { return _u.raw_shape(); }
	};

	template <typename E1, typename E2>
	class ndArrayDiv : public ndArrayExpression<ndArrayDiv<E1, E2> > {
		typename std::conditional_t<E1::is_leaf, const E1, const E1&> _u;
		typename std::conditional_t<E2::is_leaf, const E2, const E2&> _v;

	public:
		static constexpr bool is_leaf = false;
		ndArrayDiv(E1 const& u, E2 const& v) : _u(u), _v(v) {
			assert(u.raw_shape() == v.raw_shape());
		}
		decltype(auto) operator[](size_t i) const { return _u[i] / _v[i]; }
		decltype(auto) at(size_t i) const { return _u.at(i) / _v.at(i); }
		const __ndArray_shape& raw_shape()               const { return _u.raw_shape(); }
	};



	
	template <typename E1, typename E2, std::enable_if_t<std::is_arithmetic_v<E2>, bool> = true>
	class ndArrayScalarSum : public ndArrayExpression<ndArrayScalarSum<E1, E2> > {
		typename std::conditional_t<E1::is_leaf, const E1, const E1&> _u;
		const E2 _v;

	public:
		static constexpr bool is_leaf = false;
		ndArrayScalarSum(E1 const& u, E2 const v) : _u(u), _v(v) {}
		decltype(auto) operator[](size_t i) const { return _u[i] + _v; }
		decltype(auto) at(size_t i) const { return _u.at(i) + _v; }
		const __ndArray_shape& raw_shape()               const { return _u.raw_shape(); }
	};
	
	template <typename E1, typename E2, std::enable_if_t<std::is_arithmetic_v<E2>, bool> = true>
	class ndArrayScalarMul : public ndArrayExpression<ndArrayScalarMul<E1, E2> > {
		typename std::conditional_t<E1::is_leaf, const E1, const E1&> _u;
		const E2 _v;

	public:
		static constexpr bool is_leaf = false;
		ndArrayScalarMul(E1 const& u, E2 const v) : _u(u), _v(v) {}
		decltype(auto) operator[](size_t i) const { return _u[i] * _v; }
		decltype(auto) at(size_t i) const { return _u.at(i) * _v; }
		const __ndArray_shape& raw_shape()               const { return _u.raw_shape(); }
	};

	template <typename E1, typename E2, std::enable_if_t<std::is_arithmetic_v<E2>, bool> = true>
	class ndArrayScalarSub : public ndArrayExpression<ndArrayScalarSub<E1, E2> > {
		typename std::conditional_t<E1::is_leaf, const E1, const E1&> _u;
		const E2 _v;

	public:
		static constexpr bool is_leaf = false;
		ndArrayScalarSub(E1 const& u, E2 const v) : _u(u), _v(v) {}
		decltype(auto) operator[](size_t i) const { return _u[i] - _v; }
		decltype(auto) at(size_t i) const { return _u.at(i) - _v; }
		const __ndArray_shape& raw_shape()               const { return _u.raw_shape(); }
	};
	
	template <typename E1, typename E2, std::enable_if_t<std::is_arithmetic_v<E2>, bool> = true>
	class ndArrayScalarDiv : public ndArrayExpression<ndArrayScalarDiv<E1, E2> > {
		typename std::conditional_t<E1::is_leaf, const E1, const E1&> _u;
		const E2 _v;

	public:
		static constexpr bool is_leaf = false;
		ndArrayScalarDiv(E1 const& u, E2 const v) : _u(u), _v(v) {}
		decltype(auto) operator[](size_t i) const { return _u[i] / _v; }
		decltype(auto) at(size_t i) const { return _u.at(i) / _v; }
		const __ndArray_shape& raw_shape()               const { return _u.raw_shape(); }
	};
	
	#pragma endregion 	


	
	#pragma region supporters_for_vector
	template<typename >
	struct __supporter_is_vector : std::false_type {};

	template<typename T>
	struct __supporter_is_vector<std::vector<T>> : std::true_type {};

	template< class T >
	inline constexpr bool __supporter_is_vector_v = __supporter_is_vector<T>::value;


	template<typename T>
	struct __supporter_vector_element_type {
		using value_type = T;
	};

	template<typename T>
	struct __supporter_vector_element_type <std::vector<T>> {
		using value_type = typename __supporter_vector_element_type<T>::value_type;
	};

	template<typename T>
	using __supporter_vector_element_type_v = typename __supporter_vector_element_type<T>::value_type;


	template<typename T>
	struct __supporter_dim {
		static constexpr size_t value = 0;
	};

	template<typename T>
	struct __supporter_dim<std::vector<T>> {
		static constexpr size_t value = __supporter_dim<T>::value + 1;
	};

	template<typename T>
	constexpr size_t __supporter_dim_v = __supporter_dim<T>::value;


	template <typename T>
	bool __supporter_is_rect_vector(const std::vector<T>& vec) {

		if constexpr (__supporter_dim<std::vector<T>>::value == 1) {
			return true;
		}
		else if constexpr (__supporter_dim<std::vector<T>>::value == 2) {
			const size_t size = vec[0].size();
			for (size_t i = 1; i < vec.size(); ++i) {
				if (size != vec[i].size()) return false;
			}
			return true;
		}
		else {
			const size_t size = vec[0].size();
			for (size_t i = 1; i < vec.size(); ++i) {
				if (!__supporter_is_rect_vector(vec[i])) return false;
				if (size != vec[i].size()) return false;
			}
			return true;
		}
	}


	template <typename T>
	void __supporter_calc_shape(const std::vector<T>& vec, std::vector<size_t>& shp) {
		if constexpr (__supporter_dim<std::vector<T>>::value == 1) {
			shp.push_back(vec.size());
			return;
		}
		else {
			shp.push_back(vec.size());
			__supporter_calc_shape(vec[0], shp);
		}
	}
	#pragma endregion



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
		void _memcpy(void* dst, void* src, size_t size);

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

		T& at(size_t i) const { return item[i]; }

		const __ndArray_shape& raw_shape() const { return _shape; }

		std::vector<size_t> shape() const;

		void alloc(std::initializer_list<size_t> list);

		void alloc(const std::vector<size_t>& vec);

		T dot(const ndArray<T>& other);

		ndArray<T> copy();

		template <typename E>
		ndArray<T>& copy(const ndArray<E>& other);

		template <typename E>
		ndArray<T>& copy(const ndArrayExpression<E>& other);

		template <typename E>
		ndArray<T>& shallow_copy(const ndArray<E>& other);

		ndArray<T>& reshape(std::initializer_list<size_t> list);

		const T* data() const { return item; }

		size_t total_size() const { return _shape.back(); }

		ndArray<T> sum(size_t dim = 1) const;

		ndArray<T>& square();

		ndArray<size_t> argmax(size_t dim = 1) const;

		ndArray<T>& shuffle(size_t dim = 1, bool synchronize = false);

		#pragma region ndArray_operators
		ndArray<T> operator[](size_t i) const;

		ndArray<T> operator[](std::initializer_list<size_t> list) const;

		ndArray<T>& operator=(const ndArray<T>& other);

		template <typename E>
		ndArray<T>& operator=(const ndArrayExpression<E>& other);

		template<typename E, std::enable_if_t<std::is_arithmetic_v<E>, bool> = true>
		T& operator=(const E v) { assert(_shape.size() == 1); *item = v; return *item; }

		operator T& () { assert(_shape.size() == 1); return *item; }

		template <typename E>
		ndArray<T>& operator+= (const ndArrayExpression<E>& other);

		template <typename E>
		ndArray<T>& operator-= (const ndArrayExpression<E>& other);

		template <typename E>
		ndArray<T>& operator*= (const ndArrayExpression<E>& other);

		template <typename E>
		ndArray<T>& operator/= (const ndArrayExpression<E>& other);

		template <typename E, std::enable_if_t<std::is_arithmetic_v<E>, bool> = true>
		ndArray<T>& operator+= (const E scalar);

		template <typename E, std::enable_if_t<std::is_arithmetic_v<E>, bool> = true>
		ndArray<T>& operator-= (const E scalar);

		template <typename E, std::enable_if_t<std::is_arithmetic_v<E>, bool> = true>
		ndArray<T>& operator*= (const E scalar);

		template <typename E, std::enable_if_t<std::is_arithmetic_v<E>, bool> = true>
		ndArray<T>& operator/= (const E scalar);

		#pragma endregion
	};

	/*
	template <typename T>
	class broadcast {
		static constexpr bool is_leaf = false;

		const ndArray<T>* ndarray;
		size_t dim;
		size_t iter;
		std::vector<size_t> _shape;

		broadcast(const ndArray<T>* __ndarray, const std::vector<size_t>& other) : ndarray(__ndarray), dim(0) {
			
		}

		T at(size_t i) const;

		const std::vector<size_t>& raw_shape() const { return _shape; }
	};
	*/


	/////////////////////////////////////////////////////////////////////		declaration		///////////////////////////////////////////////////////////////////


	template <typename T>
	void __support_array_func(const std::vector<T>& vec, __supporter_vector_element_type_v<std::vector<T>>* item, const __ndArray_shape& shp);

	template<typename E>
	ndArray<__supporter_vector_element_type_v<std::vector<E>>> array(const std::vector<E>& vec);

	template<typename T>
	void __support_ndArray_print(T* item, const std::vector<size_t>& shp, const __ndArray_shape& raw_shp, size_t dim, size_t highest_dim);

	template<typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
	ndArray<T> range(T start, const T end, const T interval = 1);

	/////////////////////////////////////////////////////////////////////		definition		///////////////////////////////////////////////////////////////////

	template <typename T>
	template <typename E>
	ndArray<T>::ndArray(ndArrayExpression<E> const& expr):_shape(expr.raw_shape()) {
		item = original = (T*)malloc(sizeof(T) * _shape.back() + sizeof(size_t));
		assert(item != nullptr);
		ref_cnt = (size_t*)(item + _shape.back());
		*ref_cnt = 1;

		for (size_t i = _shape.back() - 1; i != 0; --i) {
			item[i] = static_cast<T>(expr.at(i));
		}
		item[0] = static_cast<T>(expr.at(0));
	}

	template <typename T>
	ndArray<T>::~ndArray() noexcept
	{
		if (original != nullptr && --(*ref_cnt) == 0) {
			free(original);
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
			item[i] = static_cast<T>(other.at(i));
		}
		item[0] = static_cast<T>(other.at(0));

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
	void ndArray<T>::alloc(std::initializer_list<size_t> list)
	{
		assert(list.size() != 0);
		if (original != nullptr && --(*ref_cnt) == 0) {
			free(original);
		}

		_shape.init(list);

		item = original = (T*)malloc(sizeof(T) * _shape.back() + sizeof(size_t));
		assert(item != nullptr);
		ref_cnt = (size_t*)(item + _shape.back());
		*ref_cnt = 1;
	}

	template <typename T>
	void ndArray<T>::alloc(const std::vector<size_t>& vec)
	{
		assert(vec.size() != 0);
		if (original != nullptr && --(*ref_cnt) == 0) {
			free(original);
		}

		_shape.init(vec);

		item = original = (T*)malloc(sizeof(T) * _shape.back() + sizeof(size_t));
		assert(item != nullptr);
		ref_cnt = (size_t*)(item + _shape.back());
		*ref_cnt = 1;
	}

	template <typename T>
	T ndArray<T>::dot(const ndArray<T>& other)
	{
		assert(_shape.size() == 2 && _shape == other._shape);

		T sum = 0;
		for (size_t i = _shape.back() - 1; i != 0 ; --i) {
			sum += other.item[i] * item[i];
		}
		sum += other.item[0] * item[0];

		return sum;
	}

	template <typename T>
	ndArray<T> ndArray<T>::copy()
	{
		ndArray<T> cpy;
		cpy._shape = _shape;

		cpy.item = cpy.original = (T*)malloc(sizeof(T) * cpy._shape.back() + sizeof(size_t));
		assert(cpy.item != nullptr);
		cpy.ref_cnt = (size_t*)(cpy.item + cpy._shape.back());
		*(cpy.ref_cnt) = 1;

		_memcpy(cpy.item, item, _shape.back());

		return cpy;
	}

	template <typename T>
	template <typename E>
	ndArray<T>& ndArray<T>::copy(const ndArray<E>& other)
	{
		assert(_shape == other._shape);

		for (size_t i = _shape.back() - 1; i != 0; --i) {
			item[i] = static_cast<T>(other.at(i));
		}
		item[0] = static_cast<T>(other.at(0));

		return *this;
	}

	template <typename T>
	template <typename E>
	ndArray<T>& ndArray<T>::copy(const ndArrayExpression<E>& other)
	{
		assert(_shape == other.raw_shape());

		for (size_t i = _shape.back() - 1; i != 0; --i) {
			item[i] = static_cast<T>(other.at(i));
		}
		item[0] = static_cast<T>(other.at(0));

		return *this;
	}

	template <typename T>
	template <typename E>
	ndArray<T>& ndArray<T>::shallow_copy(const ndArray<E>& other)
	{
		if (original != nullptr && --(*ref_cnt) == 0) {
			free(original);
		}

		item = other.item;
		original = other.original;
		_shape = other._shape;

		return *this;
	}

	template <typename T>
	void ndArray<T>::_memcpy(void* dst, void* src, size_t size)
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
		size_t i, j, k;
		size_t dim_size = _shape[dim] / _shape[dim - 1];
		size_t curr_i;

		ndArray<T> res;
		res._shape = _shape.dim_erased(dim);

		res.item = res.original = (T*)calloc(res._shape.back() + sizeof(size_t) / sizeof(T), sizeof(T));
		assert(res.original != nullptr);
		
		res.ref_cnt = (size_t*)(res.original + res._shape.back());
		++(*(res.ref_cnt));
		
		for (i = 0; i < res._shape.back(); i += _shape[dim-1]) {
			curr_i = i * dim_size;
			for (j = 0; j < _shape[dim]; j += _shape[dim-1]) {
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
		size_t i, j, k;
		size_t dim_size = _shape[dim] / _shape[dim - 1];
		size_t curr_i;

		ndArray<size_t> res;
		std::vector<T> temp(_shape[dim - 1]);
		res._shape = _shape.dim_erased(dim);

		res.item = res.original = (size_t*)calloc(res._shape.back() + 1, sizeof(size_t));
		assert(res.original != nullptr);

		res.ref_cnt = (res.original + res._shape.back());
		++(*(res.ref_cnt));

		for (i = 0; i < res._shape.back(); i += _shape[dim - 1]) {
			curr_i = i * dim_size;

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



	#pragma region support_arithmetic_operator
	template <typename E1, typename E2>
	ndArraySum<E1, E2>
		operator+(ndArrayExpression<E1> const& u, ndArrayExpression<E2> const& v) {
		return ndArraySum<E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));
	}

	template <typename E1, typename E2>
	ndArrayMul<E1, E2>
		operator*(ndArrayExpression<E1> const& u, ndArrayExpression<E2> const& v) {
		return ndArrayMul<E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));
	}

	template <typename E1, typename E2>
	ndArraySub<E1, E2>
		operator-(ndArrayExpression<E1> const& u, ndArrayExpression<E2> const& v) {
		return ndArraySub<E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));
	}

	template <typename E1, typename E2>
	ndArrayDiv<E1, E2>
		operator/(ndArrayExpression<E1> const& u, ndArrayExpression<E2> const& v) {
		return ndArrayDiv<E1, E2>(*static_cast<const E1*>(&u), *static_cast<const E2*>(&v));
	}
	
	
	template <typename E1, typename E2, std::enable_if_t<std::is_arithmetic_v<E2>, bool> = true>
	ndArrayScalarSum<E1, E2>
		operator+(ndArrayExpression<E1> const& u, E2 const v) {
		return ndArrayScalarSum<E1, E2>(*static_cast<const E1*>(&u), v);
	}
	
	template <typename E1, typename E2, std::enable_if_t<std::is_arithmetic_v<E2>, bool> = true>
	ndArrayScalarSub<E1, E2>
		operator-(ndArrayExpression<E1> const& u, E2 const v) {
		return ndArrayScalarSub<E1, E2>(*static_cast<const E1*>(&u), v);
	}
	
	template <typename E1, typename E2, std::enable_if_t<std::is_arithmetic_v<E2>, bool> = true>
	ndArrayScalarMul<E1, E2>
		operator*(ndArrayExpression<E1> const& u, E2 const v) {
		return ndArrayScalarMul<E1, E2>(*static_cast<const E1*>(&u), v);
	}
	
	template <typename E1, typename E2, std::enable_if_t<std::is_arithmetic_v<E2>, bool> = true>
	ndArrayScalarDiv<E1, E2>
		operator/(ndArrayExpression<E1> const& u, E2 const v) {
		return ndArrayScalarDiv<E1, E2>(*static_cast<const E1*>(&u), v);
	}

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
	
	#pragma endregion


	template <typename T>
	void __support_array_func(const std::vector<T>& vec, __supporter_vector_element_type_v<std::vector<T>>* item, const __ndArray_shape& shp) {
		if constexpr (__supporter_dim_v<std::vector<T>> == 1) {
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
			for (size_t i = 0; i != shp.back(); ++i) {
				cout << item[i];
				if (i != shp.back() - 1) cout << ", ";
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



	/////////////////////////////////////////////////////////////////////		namespace random		///////////////////////////////////////////////////////////////////

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
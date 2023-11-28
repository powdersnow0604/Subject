//c++ 14
#ifndef __NDARRAY_HPP__
#define __NDARRAY_HPP__


#include <memory>
#include <stdlib.h>
#include <iostream>
#include <cassert>
#include <vector>


namespace na {
	/////////////////////////////////////////////////////////////////////		declaration		///////////////////////////////////////////////////////////////////
	template <typename E>
	class ndArrayTypeWrapper;

	/////////////////////////////////////////////////////////////////////		helper class		///////////////////////////////////////////////////////////////////
	template <typename E>
	class ndArrayExpression {
	public:
		static constexpr bool is_leaf = false;
		auto operator[](size_t i) const { return static_cast<E const&>(*this)[i]; }
		auto operator()(size_t i) const { return static_cast<E const&>(*this)(i); }
		std::vector<size_t> shape() const { return static_cast<E const&>(*this).shape(); }
		const std::vector<size_t>& raw_shape() const { return static_cast<E const&>(*this).raw_shape(); }
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
		decltype(auto) operator()(size_t i) const { return _u(i) + _v(i); }
		const std::vector<size_t>& raw_shape()               const { return _u.raw_shape(); }
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
		decltype(auto) operator()(size_t i) const { return _u(i) * _v(i); }
		const std::vector<size_t>& raw_shape()               const { return _u.raw_shape(); }
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
		decltype(auto) operator()(size_t i) const { return _u(i) - _v(i); }
		const std::vector<size_t>& raw_shape()               const { return _u.raw_shape(); }
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
		decltype(auto) operator()(size_t i) const { return _u(i) / _v(i); }
		const std::vector<size_t>& raw_shape()               const { return _u.raw_shape(); }
	};



	
	template <typename E1, typename E2, std::enable_if_t<std::is_arithmetic_v<E2>, bool> = true>
	class ndArrayScalarSum : public ndArrayExpression<ndArrayScalarSum<E1, E2> > {
		typename std::conditional_t<E1::is_leaf, const E1, const E1&> _u;
		const E2 _v;

	public:
		static constexpr bool is_leaf = false;
		ndArrayScalarSum(E1 const& u, E2 const v) : _u(u), _v(v) {}
		decltype(auto) operator[](size_t i) const { return _u[i] + _v; }
		decltype(auto) operator()(size_t i) const { return _u(i) + _v; }
		const std::vector<size_t>& raw_shape()               const { return _u.raw_shape(); }
	};
	
	template <typename E1, typename E2, std::enable_if_t<std::is_arithmetic_v<E2>, bool> = true>
	class ndArrayScalarMul : public ndArrayExpression<ndArrayScalarMul<E1, E2> > {
		typename std::conditional_t<E1::is_leaf, const E1, const E1&> _u;
		const E2 _v;

	public:
		static constexpr bool is_leaf = false;
		ndArrayScalarMul(E1 const& u, E2 const v) : _u(u), _v(v) {}
		decltype(auto) operator[](size_t i) const { return _u[i] * _v; }
		decltype(auto) operator()(size_t i) const { return _u(i) * _v; }
		const std::vector<size_t>& raw_shape()               const { return _u.raw_shape(); }
	};

	template <typename E1, typename E2, std::enable_if_t<std::is_arithmetic_v<E2>, bool> = true>
	class ndArrayScalarSub : public ndArrayExpression<ndArrayScalarSub<E1, E2> > {
		typename std::conditional_t<E1::is_leaf, const E1, const E1&> _u;
		const E2 _v;

	public:
		static constexpr bool is_leaf = false;
		ndArrayScalarSub(E1 const& u, E2 const v) : _u(u), _v(v) {}
		decltype(auto) operator[](size_t i) const { return _u[i] - _v; }
		decltype(auto) operator()(size_t i) const { return _u(i) - _v; }
		const std::vector<size_t>& raw_shape()               const { return _u.raw_shape(); }
	};
	
	template <typename E1, typename E2, std::enable_if_t<std::is_arithmetic_v<E2>, bool> = true>
	class ndArrayScalarDiv : public ndArrayExpression<ndArrayScalarDiv<E1, E2> > {
		typename std::conditional_t<E1::is_leaf, const E1, const E1&> _u;
		const E2 _v;

	public:
		static constexpr bool is_leaf = false;
		ndArrayScalarDiv(E1 const& u, E2 const v) : _u(u), _v(v) {}
		decltype(auto) operator[](size_t i) const { return _u[i] / _v; }
		decltype(auto) operator()(size_t i) const { return _u(i) / _v; }
		const std::vector<size_t>& raw_shape()               const { return _u.raw_shape(); }
	};
	
	#pragma endregion 	


	
	/////////////////////////////////////////////////////////////////////		ndArray		///////////////////////////////////////////////////////////////////
	template <typename T>
	class ndArray : public ndArrayExpression<ndArray<T>> {
		T* item;
		T* original;
		size_t* ref_cnt;
		std::vector<size_t> _shape;
		
	public:
		#pragma region freind_declaration
		friend class ndArrayTypeWrapper<T>;

		template<typename E1, typename E2>
		friend class ndArraySum;

		template<typename E1, typename E2>
		friend class ndArrayMul;

		template<typename E1, typename E2>
		friend class ndArraySub;

		template<typename E1, typename E2>
		friend class ndArrayDiv;
		
		template<typename E1, typename E2, std::enable_if_t<std::is_arithmetic_v<E2>, bool>>
		friend class ndArrayScalarSum;

		template<typename E1, typename E2, std::enable_if_t<std::is_arithmetic_v<E2>, bool>>
		friend class ndArrayScalarMul;

		template<typename E1, typename E2, std::enable_if_t<std::is_arithmetic_v<E2>, bool>>
		friend class ndArrayScalarSum;

		template<typename E1, typename E2, std::enable_if_t<std::is_arithmetic_v<E2>, bool>>
		friend class ndArrayScalarMul;
		#pragma endregion

		static constexpr bool is_leaf = true;

		//functions
		ndArray(std::initializer_list<size_t> list);

		ndArray(const std::vector<size_t>& vec);

		template <typename E>
		ndArray(ndArrayExpression<E> const& expr);

		ndArray(const ndArray<T>& other) : item(other.item), original(other.original), ref_cnt(other.ref_cnt), _shape(other._shape) { ++(*ref_cnt); };

		ndArray() : item(nullptr), original(nullptr), ref_cnt(nullptr) {}

		~ndArray() noexcept;

		void assign(std::initializer_list<T> list);

		template <typename E>
		void assign(const std::vector<E>& vec);

		ndArrayTypeWrapper<T> operator[](size_t index) const;

		T operator()(size_t i) const { return item[i]; }

		const std::vector<size_t>& raw_shape() const { return _shape; }

		std::vector<size_t> shape() const;

		T dot(const ndArray<T>& other);

		ndArray<T> copy();

		void reshape(std::initializer_list<size_t> list);
	};



	template <typename E>
	class ndArrayTypeWrapper {
		size_t dim;
		E* item;
		const ndArray<E>* array;
	public:
		friend class ndArray<E>;

		//functions
		ndArrayTypeWrapper(size_t index, size_t _dim, E* _item, const ndArray<E>* _array) : dim(_dim), array(_array), item(_item + index * _array->_shape[_dim]) {};

		operator E&() { assert(dim == 0);  return *item; }

		template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
		ndArrayTypeWrapper<E>& operator=(const T v) { assert(dim == 0); *item = v; return *this; }

		ndArrayTypeWrapper<E> operator[](size_t index) {
			assert(dim != 0);
			return ndArrayTypeWrapper(index, dim - 1, item, array);
		}

	};


	/////////////////////////////////////////////////////////////////////		definition		///////////////////////////////////////////////////////////////////

	template <typename T>
	ndArray<T>::ndArray(std::initializer_list<size_t> list)
	{
		assert(list.size() != 0);

		_shape.reserve(list.size() + 1);
		_shape.push_back(1);
		for (auto iter = std::crbegin(list); iter != std::crend(list); ++iter) {
			_shape.push_back(*iter * _shape.back());
		}
		
		item = original = (T*)malloc(sizeof(T) * _shape.back() + sizeof(size_t));
		assert(item != nullptr);
		ref_cnt = (size_t*)(item + _shape.back());
		*ref_cnt = 1;
	}

	template <typename T>
	ndArray<T>::ndArray(const std::vector<size_t>& vec)
	{
		assert(vec.size() != 0);

		_shape.reserve(vec.size() + 1);
		_shape.push_back(1);
		for (auto iter = vec.crbegin(); iter != vec.crend(); ++iter) {
			_shape.push_back(*iter * _shape.back());
		}

		item = original = (T*)malloc(sizeof(T) * _shape.back() + sizeof(size_t));
		assert(item != nullptr);
		ref_cnt = (size_t*)(item + _shape.back());
		*ref_cnt = 1;
	}

	template <typename T>
	template <typename E>
	ndArray<T>::ndArray(ndArrayExpression<E> const& expr) {
		_shape = expr.raw_shape();

		item = original = (T*)malloc(sizeof(T) * _shape.back() + sizeof(size_t));
		assert(item != nullptr);
		ref_cnt = (size_t*)(item + _shape.back());
		*ref_cnt = 1;

		for (size_t i = _shape.back() - 1; i != 0; --i) {
			item[i] = expr(i);
		}
		item[0] = expr(0);
	}

	template <typename T>
	void ndArray<T>::assign(std::initializer_list<T> list)
	{
		assert(list.size() == _shape.back());
		size_t i = 0;
		for (const auto& elem : list) {
			item[i++] = elem;
		}
	}

	template <typename T>
	template <typename E>
	void ndArray<T>::assign(const std::vector<E>& vec)
	{
		assert(vec.size() == _shape.back());
		size_t i = 0;
		for (const auto& elem : vec) {
			item[i++] = elem;
		}
	}

	template <typename T>
	ndArray<T>::~ndArray() noexcept
	{
		if (--(*ref_cnt) == 0 && original != nullptr) {
			free(original);
		}
	}
	
	template <typename T>
	ndArrayTypeWrapper<T> ndArray<T>::operator[](size_t index) const
	{ 
		return ndArrayTypeWrapper<T>(index, _shape.size()-2, item, this);
	};

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

		for (size_t i = _shape.back() - 1; i != 0; --i) {
			cpy.item[i] = item[i];
		}
		cpy.item[0] = item[0];

		return  cpy;
	}

	template <typename T>
	std::ostream& operator << (std::ostream& out, const ndArray<T>& vec)
	{
		return out;
	}

	template <typename T>
	void ndArray<T>::reshape(std::initializer_list<size_t> list)
	{
		size_t sum = 0;
		for (auto& elem : list) {
			sum += elem;
		}

		assert(_shape.back() == sum);

		_shape.resize(list.size());
		size_t i = 0;
		for (auto& elem : list) {
			_shape[i++] = elem;
		}
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
	
	#pragma endregion
}

#endif
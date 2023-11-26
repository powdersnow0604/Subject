//c++ 14
#ifndef __NDARRAY_HPP__
#define __NDARRAY_HPP__


#include <memory>
#include <stdlib.h>
#include <iostream>
#include <cassert>
#include <vector>
#include <algorithm>


namespace na {

	//helper class
	template <typename E>
	class ndArrayExpression {
	public:
		double operator[](size_t i) const { return static_cast<E const&>(*this)[i]; }
		size_t size() const { return static_cast<E const&>(*this).size(); }
	};


	template <typename E1, typename E2>
	class ndArraySum : public ndArrayExpression<ndArraySum<E1, E2> > {
		// cref if leaf, copy otherwise
		const E1& _u;
		const E2& _v;

	public:
		ndArraySum(E1 const& u, E2 const& v) : _u(u), _v(v) {
			assert(u.size() == v.size());
		}
		decltype(auto) operator[](size_t i) const { return _u[i] + _v[i]; }
		size_t size()               const { return _v.size(); }
	};

	template <typename E1, typename E2>
	class ndArrayMul : public ndArrayExpression<ndArrayMul<E1, E2> > {
		// cref if leaf, copy otherwise
		const E1& _u;
		const E2& _v;

	public:
		ndArrayMul(E1 const& u, E2 const& v) : _u(u), _v(v) {
			assert(u.size() == v.size());
		}
		decltype(auto) operator[](size_t i) const { return _u[i] * _v[i]; }
		size_t size()               const { return _v.size(); }
	};

	template <typename E1, typename E2>
	class ndArraySub : public ndArrayExpression<ndArraySub<E1, E2> > {
		// cref if leaf, copy otherwise
		const E1& _u;
		const E2& _v;

	public:
		ndArraySub(E1 const& u, E2 const& v) : _u(u), _v(v) {
			assert(u.size() == v.size());
		}
		decltype(auto) operator[](size_t i) const { return _u[i] - _v[i]; }
		size_t size()               const { return _v.size(); }
	};


	template <typename E1, typename E2>
	class ndArrayDiv : public ndArrayExpression<ndArrayDiv<E1, E2> > {
		// cref if leaf, copy otherwise
		const E1& _u;
		const E2& _v;

	public:
		ndArrayDiv(E1 const& u, E2 const& v) : _u(u), _v(v) {
			assert(u.size() == v.size());
		}
		decltype(auto) operator[](size_t i) const { return _u[i] / _v[i]; }
		size_t size()               const { return _v.size(); }
	};


	//declaration
	
	

	
	//ndArray
	template <typename T>
	class ndArray : public ndArrayExpression<ndArray<T>> {
		T* item;
		T* original;
		size_t* ref_cnt;
		size_t total_size;
		std::vector<size_t> shape;
		
	public:
		ndArray(std::initializer_list<size_t> list);
		~ndArray() noexcept;
		void assign(std::initializer_list<T> list);
		T& operator[](size_t index);
		const T& operator[](size_t index) const;
		size_t size() const { return total_size; }

		template <typename E>
		ndArray(ndArrayExpression<E> const& expr) {
			total_size = expr.size();
			item = (T*)malloc(sizeof(T) * total_size + sizeof(size_t));
			assert(item != nullptr);
			ref_cnt = (size_t*)(item + total_size);
			++(*ref_cnt);

			for (size_t i = total_size-1; i != 0; --i) {
				item[i] = expr[i];
			}
			item[0] = expr[0];
		}
	};




	/////////////////////////////////////////////////////////////////////		definition		///////////////////////////////////////////////////////////////////

	template <typename T>
	ndArray<T>::ndArray(std::initializer_list<size_t> list)
	{
		shape.reserve(list.size());
		total_size = 1;
		for (const auto& elem : list) {
			shape.push_back(elem);
			total_size *= elem;
		}
		
		item = original = (T*)malloc(sizeof(T) * total_size + sizeof(size_t));
		assert(item != nullptr);
		ref_cnt = (size_t*)(item + total_size);
		++(*ref_cnt);
	}

	template <typename T>
	void ndArray<T>::assign(std::initializer_list<T> list)
	{
		assert(list.size() == total_size);
		size_t i = 0;
		for (const auto& elem : list) {
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
	T& ndArray<T>::operator[](size_t index)
	{ 
		assert(shape.size() == 0);
		return item[index];

	};

	template <typename T>
	const T& ndArray<T>::operator[](size_t index) const 
	{ 
		return item[index];
	}


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

}

#endif
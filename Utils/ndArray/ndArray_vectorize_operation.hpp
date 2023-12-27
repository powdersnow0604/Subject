#ifndef __NDARRAY_VECTORIZE_OPERATOR__
#define __NDARRAY_VECTORIZE_OPERATOR__

namespace na {
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
		const auto& raw_shape()               const { return _u.raw_shape(); }
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
		const auto& raw_shape()               const { return _u.raw_shape(); }
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
		const auto& raw_shape()               const { return _u.raw_shape(); }
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
		const auto& raw_shape()               const { return _u.raw_shape(); }
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
		const auto& raw_shape()               const { return _u.raw_shape(); }
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
		const auto& raw_shape()               const { return _u.raw_shape(); }
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
		const auto& raw_shape()               const { return _u.raw_shape(); }
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
		const auto& raw_shape()               const { return _u.raw_shape(); }
	};

#pragma endregion 	


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


	template <typename E1, typename E2>
	ndArrayScalarSum<E1, E2>
		operator+(ndArrayExpression<E1> const& u, E2 const v) {
		return ndArrayScalarSum<E1, E2>(*static_cast<const E1*>(&u), v);
	}

	template <typename E1, typename E2>
	ndArrayScalarSub<E1, E2>
		operator-(ndArrayExpression<E1> const& u, E2 const v) {
		return ndArrayScalarSub<E1, E2>(*static_cast<const E1*>(&u), v);
	}

	template <typename E1, typename E2>
	ndArrayScalarMul<E1, E2>
		operator*(ndArrayExpression<E1> const& u, E2 const v) {
		return ndArrayScalarMul<E1, E2>(*static_cast<const E1*>(&u), v);
	}

	template <typename E1, typename E2>
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
			//item[i] = static_cast<T>(item[i] + other.at(i));
			item[i] += other.at(i);
		}
		//item[0] = static_cast<T>(item[0] + other.at(0));
		item[0] += other.at(0);

		return *this;
	}

	template <typename T>
	template <typename E>
	ndArray<T>& ndArray<T>::operator-= (const ndArrayExpression<E>& other)
	{
		assert(_shape == other.raw_shape());
		for (size_t i = _shape.back() - 1; i != 0; --i) {
			//item[i] = static_cast<T>(item[i] - other.at(i));
			item[i] -= other.at(i);
		}
		//item[0] = static_cast<T>(item[0] - other.at(0));
		item[0] -= other.at(0);

		return *this;
	}

	template <typename T>
	template <typename E>
	ndArray<T>& ndArray<T>::operator*= (const ndArrayExpression<E>& other)
	{
		assert(_shape == other.raw_shape());
		for (size_t i = _shape.back() - 1; i != 0; --i) {
			//item[i] = static_cast<T>(item[i] * other.at(i));
			item[i] *= other.at(i);
		}
		//item[0] = static_cast<T>(item[0] * other.at(0));
		item[0] *= other.at(0);

		return *this;
	}

	template <typename T>
	template <typename E>
	ndArray<T>& ndArray<T>::operator/= (const ndArrayExpression<E>& other)
	{
		assert(_shape == other.raw_shape());
		for (size_t i = _shape.back() - 1; i != 0; --i) {
			//item[i] = static_cast<T>(item[i] / other.at(i));
			item[i] /= other.at(i);
		}
		//item[0] = static_cast<T>(item[0] / other.at(0));
		item[0] /= other.at(0);

		return *this;
	}

	template <typename T>
	template <typename E>
	ndArray<T>& ndArray<T>::operator+= (const E scalar)
	{
		for (size_t i = _shape.back() - 1; i != 0; --i) {
			//item[i] = static_cast<T>(item[i] + scalar);
			item[i] += scalar;
		}
		//item[0] = static_cast<T>(item[0] + scalar);
		item[0] += scalar;


		return *this;
	}

	template <typename T>
	template <typename E>
	ndArray<T>& ndArray<T>::operator-= (const E scalar)
	{
		for (size_t i = _shape.back() - 1; i != 0; --i) {
			//item[i] = static_cast<T>(item[i] - scalar);
			item[i] -= scalar;
		}
		//item[0] = static_cast<T>(item[0] - scalar);
		item[0] -= scalar;

		return *this;
	}

	template <typename T>
	template <typename E>
	ndArray<T>& ndArray<T>::operator*= (const E scalar)
	{
		for (size_t i = _shape.back() - 1; i != 0; --i) {
			//item[i] = static_cast<T>(item[i] * scalar);
			item[i] *= scalar;
		}
		//item[0] = static_cast<T>(item[0] * scalar);
		item[0] *= scalar;

		return *this;
	}

	template <typename T>
	template <typename E>
	ndArray<T>& ndArray<T>::operator/= (const E scalar)
	{
		for (size_t i = _shape.back() - 1; i != 0; --i) {
			//item[i] = static_cast<T>(item[i] / scalar);
			item[i] /= scalar;
		}
		//item[0] = static_cast<T>(item[0] / scalar);
		item[0] /= scalar;

		return *this;
	}

#pragma endregion
}


#endif

#ifndef __NDARRAYTYPEWRAPPER_H__
#define __NDARRAYTYPEWRAPPER_H__


//forward declaration
template <typename E>
class ndArrayTypeWrapper;



//ndArray

//in class
template<typename E>
friend class ndArrayTypeWrapper;

ndArray(const ndArrayTypeWrapper<T>& other);

ndArrayTypeWrapper<T> operator[](size_t index) const;


//outside class
template <typename T>
ndArray<T>::ndArray(const ndArrayTypeWrapper<T>& other)
{
	item = other.item;
	original = other.array->original;
	ref_cnt = other.array->ref_cnt;
	++(*ref_cnt);
	_shape = other.array->_shape;
	_shape.resize(other.dim + 1);
}

template <typename T>
ndArrayTypeWrapper<T> ndArray<T>::operator[](size_t index) const
{
	return ndArrayTypeWrapper<T>(index, _shape.size() - 2, item, this);
};


/////////


template <typename E>
class ndArrayTypeWrapper {
	size_t dim;
	E* item;
	const ndArray<E>* array;
public:
	friend class ndArray<E>;

	//functions
	ndArrayTypeWrapper(size_t index, size_t _dim, E* _item, const ndArray<E>* _array) : dim(_dim), array(_array), item(_item + index * _array->_shape[_dim]) {};

	operator E& () { assert(dim == 0);  return *item; }

	E at(size_t i) const { return item[i]; }

	template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
	ndArrayTypeWrapper<E>& operator=(const T v) { assert(dim == 0); *item = v; return *this; }

	template <typename T>
	ndArrayTypeWrapper<E>& operator=(const ndArrayTypeWrapper<T>& v);

	ndArrayTypeWrapper<E>& operator=(const ndArrayTypeWrapper<E>& v);

	template <typename T>
	ndArrayTypeWrapper<E>& operator=(const ndArray<T>& v);

	ndArrayTypeWrapper<E> operator[](size_t index) {
		assert(dim != 0);
		return ndArrayTypeWrapper(index, dim - 1, item, array);
	}
};

template<typename E>
template <typename T>
ndArrayTypeWrapper<E>& ndArrayTypeWrapper<E>::operator=(const ndArrayTypeWrapper<T>& v)
{
	assert(dim == v.dim);
	for (size_t i = dim; i != 0; --i) {
		assert(this->array->_shape[i] == v.array->_shape[i]);
	}

	for (size_t i = this->array->_shape[dim] - 1; i != 0; --i) {
		item[i] = v.item[i];
	}
	item[0] = v.item[0];

	return *this;
}

template<typename E>
template <typename T>
ndArrayTypeWrapper<E>& ndArrayTypeWrapper<E>::operator=(const ndArray<T>& v)
{
	assert(dim == v._shape.size() - 1);

	for (size_t i = dim; i != 0; --i) {
		assert(this->array->_shape[i] == v._shape[i]);
	}

	for (size_t i = v._shape[dim] - 1; i != 0; --i) {
		item[i] = v.item[i];
	}
	item[0] = v.item[0];

	return *this;
}

template<typename E>
ndArrayTypeWrapper<E>& ndArrayTypeWrapper<E>::operator=(const ndArrayTypeWrapper<E>& v)
{
	assert(dim == v.dim);
	for (size_t i = dim; i != 0; --i) {
		assert(this->array->_shape[i] == v.array->_shape[i]);
	}

	for (size_t i = this->array->_shape[dim] - 1; i != 0; --i) {
		item[i] = v.item[i];
	}
	item[0] = v.item[0];

	return *this;
}

#endif

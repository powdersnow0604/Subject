#include "ndArray_shape.h"


namespace na {

	std::allocator<size_t> __ndArray_shape_allocator;
	const size_t __static_ptr_size = 5;

	////////////////////////////////////////////// shape member function //////////////////////////////////////////////////////////////

	__ndArray_shape::__ndArray_shape(const std::vector<size_t>& shp): _size(shp.size() + 1), capacity(0), static_ptr{1,0,0,0,0}
	{
		if (_size <= __static_ptr_size) ptr = static_ptr;
		else {
			ptr = __ndArray_shape_allocator.allocate(_size);
			capacity = _size;
			ptr[0] = 1;
		}

		size_t i = 1;
		for (auto iter = shp.crbegin(); i < _size; ++i, ++iter) {
			ptr[i] = ptr[i - 1] * (*iter);
		}
	}

	__ndArray_shape::__ndArray_shape(std::initializer_list<size_t> shp) : _size(shp.size() + 1), capacity(0), static_ptr{1,0,0,0,0}
	{
		if (_size <= __static_ptr_size) ptr = static_ptr;
		else {
			ptr = __ndArray_shape_allocator.allocate(_size);
			capacity = _size;
			ptr[0] = 1;
		}

		size_t i = 1;
		for (auto iter = std::crbegin(shp); i < _size; ++i, ++iter) {
			ptr[i] = ptr[i - 1] * (*iter);
		}
	}

	__ndArray_shape::__ndArray_shape(const __ndArray_shape& other) :_size(other._size), capacity(0), static_ptr{ 1,0,0,0,0 }
	{
		if (_size <= __static_ptr_size) ptr = static_ptr;
		else {
			capacity = _size;
			ptr = __ndArray_shape_allocator.allocate(capacity);
			ptr[0] = 1;
		}

		for (size_t i = _size - 1; i != 0; --i) {
			ptr[i] = other.ptr[i];
		}
	}

	__ndArray_shape::__ndArray_shape(const __ndArray_shape& other, int mode, size_t arg) : capacity(0), static_ptr{ 1,0,0,0,0 }
	{
		//decrease dim
		if (mode == 0) {
			if (arg < _size && arg > 0) {
				_size = arg;
			}
			else _size = other._size;

			if (_size <= __static_ptr_size) ptr = static_ptr;
			else {
				capacity = _size;
				ptr = __ndArray_shape_allocator.allocate(capacity);
				ptr[0] = 1;
			}

			for (size_t i = _size - 1; i != 0; --i) {
				ptr[i] = other.ptr[i];
			}
		}
		//erase dim
		else {
			assert(arg < _size && arg > 0);
			_size = other._size - 1;
			size_t target_dim_size = other.ptr[arg] / other.ptr[arg - 1];

			if (_size <= __static_ptr_size) ptr = static_ptr;
			else {
				capacity = _size;
				ptr = __ndArray_shape_allocator.allocate(capacity);
				ptr[0] = 1;
			}

			size_t i;
			for (i = arg - 1; i != 0; --i) {
				ptr[i] = other.ptr[i];
			}
			for (i = arg; i < other._size - 1; ++i) {
				ptr[i] = other.ptr[i + 1] / target_dim_size;
			}
		}
	}

	__ndArray_shape::__ndArray_shape(const __ndArray_shape_view& shp) :_size(shp.size()), capacity(0), static_ptr{ 1,0,0,0,0 }
	{
		if (_size <= __static_ptr_size) ptr = static_ptr;
		else {
			capacity = _size;
			ptr = __ndArray_shape_allocator.allocate(capacity);
			ptr[0] = 1;
		}

		for (size_t i = _size - 1; i != 0; --i) {
			ptr[i] = shp[i];
		}
	}

	__ndArray_shape::__ndArray_shape(__ndArray_shape&& other) noexcept :_size(other._size), capacity(0), static_ptr{ 1,0,0,0,0 }
	{
		if (_size <= __static_ptr_size) {
			ptr = static_ptr;
			for (size_t i = _size - 1; i != 0; --i) {
				ptr[i] = other.ptr[i];
			}
		}
		else {
			ptr = other.ptr;
			capacity = other.capacity;
			other.ptr = nullptr;
			other._size = 0;
		}
	}

	__ndArray_shape::~__ndArray_shape() noexcept
	{
		if (_size > __static_ptr_size) __ndArray_shape_allocator.deallocate(ptr, _size);
	}

	__ndArray_shape& __ndArray_shape::init(const std::vector<size_t>& shp)
	{
		//if (_size != 0) return *this;
		if (_size > __static_ptr_size) __ndArray_shape_allocator.deallocate(ptr, _size);

		_size = shp.size() + 1;
		if (_size <= __static_ptr_size) {
			ptr = static_ptr;
			capacity = 0;
		}
		else {
			ptr = __ndArray_shape_allocator.allocate(_size);
			capacity = _size;
			ptr[0] = 1;
		}

		size_t i = 1;
		for (auto iter = shp.crbegin(); i < _size; ++i, ++iter) {
			ptr[i] = ptr[i - 1] * (*iter);
		}

		return *this;
	}

	__ndArray_shape& __ndArray_shape::init(std::initializer_list<size_t> shp)
	{
		//if (_size != 0) return *this;
		if (_size > __static_ptr_size) __ndArray_shape_allocator.deallocate(ptr, _size);

		_size = shp.size() + 1;
		if (_size <= __static_ptr_size) {
			ptr = static_ptr;
			capacity = 0;
		}
		else {
			ptr = __ndArray_shape_allocator.allocate(_size);
			capacity = _size;
			ptr[0] = 1;
		}


		size_t i = 1;
		for (auto iter = std::crbegin(shp); i < _size; ++i, ++iter) {
			ptr[i] = ptr[i - 1] * (*iter);
		}

		return *this;
	}
	
	__ndArray_shape& __ndArray_shape::reshape(const std::vector<size_t>& shp)
	{
		size_t sum = 1;
		for (auto& elem : shp) {
			sum *= elem;
		}

		assert(sum == ptr[_size - 1]);

		if ((shp.size() + 1 <= _size) || (shp.size() + 1 <= __static_ptr_size) || (shp.size() + 1 <= capacity)) {
			_size = shp.size() + 1;
			size_t i = 1;
			for (auto iter = shp.crbegin(); iter != shp.crend(); ++iter, ++i) {
				ptr[i] = *iter * ptr[i - 1];
			}
		}
		else {
			__ndArray_shape_allocator.deallocate(ptr, _size);

			_size = capacity = shp.size() + 1;
			ptr = __ndArray_shape_allocator.allocate(_size);
			ptr[0] = 1;
			size_t i = 1;
			for (auto iter = shp.crbegin(); iter != shp.crend(); ++iter, ++i) {
				ptr[i] = *iter * ptr[i - 1];
			}

		}

		return *this;
	}

	__ndArray_shape& __ndArray_shape::reshape(std::initializer_list<size_t> shp)
	{
		size_t sum = 1;
		for (auto& elem : shp) {
			sum *= elem;
		}

		assert(sum == ptr[_size - 1]);

		if ((shp.size() + 1 <= _size) || (shp.size() + 1 <= __static_ptr_size) || (shp.size() + 1 <= capacity)) {
			_size = shp.size() + 1;
			size_t i = 1;
			for (auto iter = std::crbegin(shp); iter != std::crend(shp); ++iter, ++i) {
				ptr[i] = *iter * ptr[i - 1];
			}
		}
		else {
			__ndArray_shape_allocator.deallocate(ptr, _size);

			_size = capacity = shp.size() + 1;
			ptr = __ndArray_shape_allocator.allocate(_size);
			ptr[0] = 1;
			size_t i = 1;
			for (auto iter = std::crbegin(shp); iter != std::crend(shp); ++iter, ++i) {
				ptr[i] = *iter * ptr[i - 1];
			}

		}

		return *this;
	}

	__ndArray_shape& __ndArray_shape::decrease_dim(size_t new_size)
	{
		if (new_size < _size && new_size > 0) {
			_size = new_size;
		}

		return *this;
	}

	__ndArray_shape __ndArray_shape::dim_decreased(size_t new_size) const
	{
		if (new_size < _size && new_size > 0) {
			__ndArray_shape res;
			res._size = new_size;

			if (res._size <= __static_ptr_size) res.ptr = res.static_ptr;
			else {
				res.capacity = res._size;
				res.ptr = __ndArray_shape_allocator.allocate(res.capacity);
				res.ptr[0] = 1;
			}

			for (size_t i = res._size - 1; i != 0; --i) {
				res.ptr[i] = ptr[i];
			}

			return res;
		}

		return *this;
	}

	__ndArray_shape& __ndArray_shape::erase_dim(size_t dim)
	{
		assert(dim < _size && dim > 0);

		size_t target_dim_size = ptr[dim] / ptr[dim - 1];
		--_size;

		for (size_t i = dim; i < _size; ++i) {
			ptr[i] = ptr[i + 1] / target_dim_size;
		}

		return *this;
	}

	__ndArray_shape __ndArray_shape::dim_erased(size_t dim) const
	{
		assert(dim < _size && dim > 0);

		size_t target_dim_size = ptr[dim] / ptr[dim - 1];

		__ndArray_shape res;
		res._size = _size - 1;

		if (res._size <= __static_ptr_size) res.ptr = res.static_ptr;
		else {
			res.capacity = res._size;
			res.ptr = __ndArray_shape_allocator.allocate(res.capacity);
			res.ptr[0] = 1;
		}

		size_t i;
		for (i = dim - 1; i != 0; --i) {
			res.ptr[i] = ptr[i];
		}
		for (i = dim; i < res._size; ++i) {
			res.ptr[i] = ptr[i + 1] / target_dim_size;
		}

		return res;
	}

	bool __ndArray_shape::operator==(const __ndArray_shape& other) const
	{
		if (_size != other._size) return false;
		for (size_t i = _size - 1; i != 0; --i) {
			if (other.ptr[i] != ptr[i]) return false;
		}

		return true;
	}

	__ndArray_shape& __ndArray_shape::operator=(const __ndArray_shape& other)
	{
		if ((other._size <= _size) || (other._size <= __static_ptr_size) || (other._size <= capacity)) {
			if (_size == 0) ptr = static_ptr;
			_size = other._size;
			for (size_t i = _size - 1;; --i) {
				ptr[i] = other.ptr[i];
				if (i == 0) break;
			}
			return *this;
		}
		else {
			if (_size > __static_ptr_size) __ndArray_shape_allocator.deallocate(ptr, _size);
			ptr = __ndArray_shape_allocator.allocate(other._size);
			_size = other._size;
			for (size_t i = _size - 1;; --i) {
				ptr[i] = other.ptr[i];
				if (i == 0) break;
			}
			return *this;
		}
	}

	///////////////////////////////////////////////////////  view member function ///////////////////////////////////////////////////////

	void __ndArray_shape_view::init(const __ndArray_shape& shp)
	{
		ptr = shp.ptr;
		_size = shp._size;
	}

	__ndArray_shape_view& __ndArray_shape_view::operator=(const __ndArray_shape_view& other)
	{
		ptr = other.ptr;
		_size = other._size;
		return *this;
	}

	
	///////////////////////////////////////////////////////  functions ///////////////////////////////////////////////////////
	std::ostream& operator << (std::ostream& out, const __ndArray_shape& shp)
	{
		out << "[";
		for (size_t i = 0; i < shp.size()-1; ++i) {
			out << shp[i] << " ";
		}
		out << shp[shp.size() - 1] << "]";

		return out;
	}

	bool operator==(const __ndArray_shape& arg1, const __ndArray_shape_view& arg2)
	{
		if (arg1.size() != arg2.size()) return false;
		for (size_t i = arg1.size() - 1; i != 0; --i) {
			if (arg1[i] != arg2[i]) return false;
		}

		return true;
	}

	bool operator==(const __ndArray_shape_view& arg1, const __ndArray_shape& arg2)
	{
		if (arg1.size() != arg2.size()) return false;
		for (size_t i = arg1.size() - 1; i != 0; --i) {
			if (arg1[i] != arg2[i]) return false;
		}

		return true;
	}
}
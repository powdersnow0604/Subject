#ifndef __NDARRAY_SHAPE__
#define __NDARRAY_SHAPE__

#include <vector>
#include <memory>
#include <cassert>
#include <iostream>


namespace na {

	/////////////////////////////////////////////////////////////////////		class		///////////////////////////////////////////////////////////////////
	class __ndArray_shape
	{
		size_t* ptr;
		size_t static_ptr[5];
		size_t _size;
		size_t capacity;
	public:
		friend class __ndArray_shape_view;
		__ndArray_shape() : ptr(nullptr), _size(0), capacity(0), static_ptr{1,0,0,0,0} {}
		__ndArray_shape(const std::vector<size_t>& shp);
		__ndArray_shape(std::initializer_list<size_t> shp);
		__ndArray_shape(const __ndArray_shape& other);
		__ndArray_shape(const __ndArray_shape& other, int mode, size_t arg);
		__ndArray_shape(const __ndArray_shape_view& shp);
		__ndArray_shape(__ndArray_shape&& other) noexcept;
		~__ndArray_shape() noexcept;
		__ndArray_shape& init(const std::vector<size_t>& shp);
		__ndArray_shape& init(std::initializer_list<size_t> shp);
		size_t size() const { return _size; }
		size_t back() const { return ptr[_size - 1]; }
		__ndArray_shape& reshape(const std::vector<size_t>& shp);
		__ndArray_shape& reshape(std::initializer_list<size_t> shp);
		__ndArray_shape& decrease_dim(size_t new_size);
		__ndArray_shape dim_decreased(size_t new_size) const;
		__ndArray_shape& erase_dim(size_t dim);
		__ndArray_shape dim_erased(size_t dim) const;
		bool operator== (const __ndArray_shape& other) const;
		__ndArray_shape& operator=(const __ndArray_shape& other);
		size_t& operator[](size_t i) { assert(i < _size);  return ptr[i]; }
		size_t operator[](size_t i) const { assert(i < _size); return ptr[i]; }
	};


	class __ndArray_shape_view
	{
		size_t* ptr;
		size_t _size;
	public:
		__ndArray_shape_view() : ptr(nullptr), _size(0) {}
		__ndArray_shape_view(const __ndArray_shape& shp) :ptr(shp.ptr), _size(shp._size) {}
		void init(const __ndArray_shape& shp);
		__ndArray_shape_view& operator=(const __ndArray_shape_view& other);
		size_t size() const { return _size; }
		size_t operator[](size_t i) const { assert(i < _size); return ptr[i]; }
	};

	/////////////////////////////////////////////////////////////////////		functions		///////////////////////////////////////////////////////////////////
	std::ostream& operator << (std::ostream& out, const __ndArray_shape& arr);

	bool operator==(const __ndArray_shape& arg1, const __ndArray_shape_view& arg2);

	bool operator==(const __ndArray_shape_view& arg1, const __ndArray_shape& arg2);
}

#endif
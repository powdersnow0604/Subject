#ifndef __NDARRAY_ALLOCATOR_HPP__
#define __NDARRAY_ALLOCATOR_HPP__

namespace na {
	class __ndArray_cache {
		//{ 64 256 1024 unlimited } * 8 bytes
		void* cached[4];
	public:
		__ndArray_cache(): cached{0,0,0,0} {}
		void* allocate(size_t size);
		void deallocate(void* ptr, size_t size);
	};
}


#endif 
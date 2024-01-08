#include <memory>
#include "ndArray_allocator.h"

#define ALIGNED_MALLOC(size, align) _aligned_malloc((size), (align))


namespace na {

	////////////////////////////  instance  ////////////////////////////

	__ndArray_allocator __ndArray_allocator_instance;

	namespace linalg {
		__ndArray_allocator __ndArray_allocator_linalg;
	}

	////////////////////////////////////////////////////////////////////

	void* __ndArray_allocator::allocate(size_t size)
	{
		if (size < cache.limit) {
			mtx.lock();
			void* candidate = cache.allocate(size);
			mtx.unlock();

			if (candidate) return candidate;
		}
		
		return malloc(size);
	}

	void* __ndArray_allocator::zero_initialized_allocate(size_t _Count, size_t _Size)
	{
		return calloc(_Count, _Size);
	}

	void __ndArray_allocator::deallocate(void* ptr, size_t size)
	{
		if (size >= cache.limit) {
			free(ptr);
			return;
		}

		mtx.lock();
		cache.deallocate(ptr, size);
		mtx.unlock();
	}

	void* __ndArray_cache_sep_by_class::allocate(size_t size)
	{
		if (cnt != 0) {
			size_t cls = size >> interval_exponential;
			if (cached_size[cls] == size) {
				cached_size[cls] = 0;
				to_strk = 1;
				return cached[cls];
			}
			else if (cached_size[cls + class_num] == size) {
				cached_size[cls + class_num] = 0;
				to_strk = 1;
				return cached[cls + class_num];
			}
			else {
				if (--timer == 0) {
					for (uint8_t i = to_strk; i != 0; --i) {
						uint8_t candidate = dist(gen);
						if (cached_size[candidate] != 0) {
							free(cached[candidate]);
							cached_size[candidate] = 0;
							--cnt;
						}
					}
					timer = timer_limit;
				}
			}
		}

		return nullptr;
		
	}

	void __ndArray_cache_sep_by_class::deallocate(void* ptr, size_t size)
	{
		size_t cls = size >> interval_exponential;
		if (cached_size[cls] == 0) {
			cached_size[cls] = size;
			cached[cls] = ptr;
			++cnt;
		}
		else if (cached_size[cls + class_num] == 0) {
			cached_size[cls + class_num] = size;
			cached[cls + class_num] = ptr;
			++cnt;
		}
		else {
			cached_size[cls] = size;
			free(cached[cls]);
			cached[cls] = ptr;
		}
	}
}
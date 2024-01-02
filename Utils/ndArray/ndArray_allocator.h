#ifndef __NDARRAY_ALLOCATOR_HPP__
#define __NDARRAY_ALLOCATOR_HPP__

#include <cinttypes>
#include <random>
#include <mutex>

namespace na {

	class __ndArray_cache_sep_by_class {
	public:
		static constexpr uint32_t limit = 1 << 17;
		static constexpr uint16_t interval = 1 << 12;
		static constexpr uint16_t interval_exponential = 12;
		static constexpr uint8_t capacity = 64;
		static constexpr uint8_t class_num = 32;
		static constexpr uint8_t timer_limit = 3;
	private:
		void* cached[capacity];
		size_t cached_size[capacity];
		int8_t timer;
		std::mt19937 gen;
		std::uniform_int_distribution<int> dist;
		uint8_t cnt;
		uint8_t to_strk;
	public:
		__ndArray_cache_sep_by_class() : cached{ 0, }, cached_size{ 0, }, timer(timer_limit), gen(std::random_device()()), dist(0, capacity - 1), cnt(0), to_strk(1) {}
		//assuming size is not greater than limit
		void* allocate(size_t size);
		void deallocate(void* ptr, size_t size);
	};

	class __ndArray_allocator {
		std::mutex mtx;
		__ndArray_cache_sep_by_class cache;
	public:
		void* allocate(size_t size);
		void* zero_initialized_allocate(size_t _Count, size_t _Size);
		void deallocate(void* ptr, size_t size);
	};

	class __ndArray_allocator_without_optimization {
	public:
		void* allocate(size_t size) { return malloc(size); }
		void* zero_initialized_allocate(size_t _Count, size_t _Size) { return calloc(_Count, _Size); }
		void deallocate(void* ptr, size_t size) { free(ptr); }
	};

}


#endif 
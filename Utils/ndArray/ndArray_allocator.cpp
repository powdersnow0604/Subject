#include <memory>
#include "ndArray_allocator.h"
#include <limits>

namespace na {
	constexpr size_t __na_cache_cap_1 = std::numeric_limits<size_t>::max() << 9;
	constexpr size_t __na_cache_cap_2 = std::numeric_limits<size_t>::max() << 11;
	constexpr size_t __na_cache_cap_3 = std::numeric_limits<size_t>::max() << 13;


	void* __ndArray_cache::allocate(size_t size) {
		void* candidate = nullptr;

		if (size & __na_cache_cap_2) {
			if (size & __na_cache_cap_3) {
				if (cached[3] != nullptr) {
					candidate = cached[3];
					cached[3] = nullptr;
				}
			}
			else {
				if (cached[2] != nullptr) {
					candidate = cached[2];
					cached[2] = nullptr;
				}
			}
		}
		else {
			if (size & __na_cache_cap_1) {
				if (cached[1] != nullptr) {
					candidate = cached[1];
					cached[1] = nullptr;
				}
			}
			else {
				if (cached[0] != nullptr) {
					candidate = cached[0];
					cached[0] = nullptr;
				}
			}
		}

		return nullptr;
	}
}
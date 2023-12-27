#ifndef __NDARRAY_SUPPORTERS_FOR_STDVECTOR__
#define __NDARRAY_SUPPORTERS_FOR_STDVECTOR__

namespace na {
#pragma region supporters_for_vector
	template<typename >
	struct __supporter_is_vector : std::false_type {};

	template<typename T>
	struct __supporter_is_vector<std::vector<T>> : std::true_type {};

	template< class T >
	inline constexpr bool __supporter_is_vector_v = __supporter_is_vector<T>::value;


	template<typename T>
	struct __supporter_vector_element_type {
		using value_type = T;
	};

	template<typename T>
	struct __supporter_vector_element_type <std::vector<T>> {
		using value_type = typename __supporter_vector_element_type<T>::value_type;
	};

	template<typename T>
	using __supporter_vector_element_type_v = typename __supporter_vector_element_type<T>::value_type;


	template<typename T>
	struct __supporter_dim {
		static constexpr size_t value = 0;
	};

	template<typename T>
	struct __supporter_dim<std::vector<T>> {
		static constexpr size_t value = __supporter_dim<T>::value + 1;
	};

	template<typename T>
	constexpr size_t __supporter_dim_v = __supporter_dim<T>::value;


	template <typename T>
	bool __supporter_is_rect_vector(const std::vector<T>& vec) {

		if constexpr (__supporter_dim<std::vector<T>>::value == 1) {
			return true;
		}
		else if constexpr (__supporter_dim<std::vector<T>>::value == 2) {
			const size_t size = vec[0].size();
			for (size_t i = 1; i < vec.size(); ++i) {
				if (size != vec[i].size()) return false;
			}
			return true;
		}
		else {
			const size_t size = vec[0].size();
			for (size_t i = 1; i < vec.size(); ++i) {
				if (!__supporter_is_rect_vector(vec[i])) return false;
				if (size != vec[i].size()) return false;
			}
			return true;
		}
	}


	template <typename T>
	void __supporter_calc_shape(const std::vector<T>& vec, std::vector<size_t>& shp) {
		if constexpr (__supporter_dim<std::vector<T>>::value == 1) {
			shp.push_back(vec.size());
			return;
		}
		else {
			shp.push_back(vec.size());
			__supporter_calc_shape(vec[0], shp);
		}
	}
#pragma endregion
}

#endif 


#ifndef __SUPPORTVECTOR__
#define __SUPPORTVECTOR__

#include <vector>
#include <functional>
#include <iostream>
#include <type_traits>
#include <cmath>

using std::vector;
using std::function;


namespace SupportVector {
	//declaration
	static size_t print_vector_helper = 0;


	//template meta programmiing
	template<typename >
	struct is_vector : std::false_type {};

	template<typename T>
	struct is_vector<std::vector<T>> : std::true_type {};

	template< class T >
	inline constexpr bool is_vector_v = is_vector<T>::value;


	template<typename T>
	struct vector_element_type {
		using value_type = T;
	};

	template<typename T>
	struct vector_element_type <std::vector<T>> {
		using value_type = vector_element_type<T>::value_type;
	};

	template<typename T>
	using vector_element_type_v = vector_element_type<T>::value_type;


	template<typename T>
	struct dim {
		static const size_t value = 0;
	};

	template<typename T>
	struct dim<std::vector<T>> {
		static const size_t value = dim<T>::value + 1;
	};

	template<typename T>
	size_t dim_v = dim<T>::value;


	template<size_t N, typename T>
	struct VectorNd_helper {
		using type = std::vector<typename VectorNd_helper<N - 1, T>::type>;
	};

	template<typename T>
	struct VectorNd_helper<0, T> {
		using type = T;
	};

	template<size_t N, typename T>
	using VectorNd = VectorNd_helper<N, T>::type;

	//definition
	#pragma region support vectorize operation
	//for vector wise multiplication
	template<typename T>
	vector<T> operator+(const vector<T>& vec1, const vector<T>& vec2)
	{
		if (vec1.size() != vec2.size()) return {};

		vector<T> res;
		res.reserve(vec1.size());

		for (size_t i = 0; i < vec1.size(); ++i) {
			res.push_back(vec1[i] + vec2[i]);
		}

		return res;
	}

	template<typename T>
	vector<T> operator-(const vector<T>& vec1, const vector<T>& vec2)
	{
		if (vec1.size() != vec2.size()) return {};

		vector<T> res;
		res.reserve(vec1.size());

		for (size_t i = 0; i < vec1.size(); ++i) {
			res.push_back(vec1[i] - vec2[i]);
		}

		return res;
	}

	template<typename T>
	vector<T> operator*(const vector<T>& vec1, const vector<T>& vec2)
	{
		if (vec1.size() != vec2.size()) return {};

		vector<T> res;
		res.reserve(vec1.size());

		for (size_t i = 0; i < vec1.size(); ++i) {
			res.push_back(vec1[i] * vec2[i]);
		}

		return res;
	}

	template<typename T>
	vector<T> operator/(const vector<T>& vec1, const vector<T>& vec2)
	{
		if (vec1.size() != vec2.size()) return {};

		vector<T> res;
		res.reserve(vec1.size());

		for (size_t i = 0; i < vec1.size(); ++i) {
			res.push_back(vec1[i] / vec2[i]);
		}

		return res;
	}

	template <typename T>
	vector<T>& operator+=(vector<T>& vec1, const vector<T>& vec2)
	{
		if (vec1.size() != vec2.size()) return vec1;

		auto iter_vec2 = vec2.begin();
		for (auto& i : vec1) {
			i += *iter_vec2++;
		}

		return vec1;
	}

	template <typename T>
	vector<T>& operator-=(vector<T>& vec1, const vector<T>& vec2)
	{
		if (vec1.size() != vec2.size()) return vec1;

		auto iter_vec2 = vec2.begin();
		for (auto& i : vec1) {
			i -= *iter_vec2++;
		}

		return vec1;
	}

	template <typename T>
	vector<T>& operator*=(vector<T>& vec1, const vector<T>& vec2)
	{
		if (vec1.size() != vec2.size()) return vec1;

		auto iter_vec2 = vec2.begin();
		for (auto& i : vec1) {
			i *= *iter_vec2++;
		}

		return vec1;
	}

	template <typename T>
	vector<T>& operator/=(vector<T>& vec1, const vector<T>& vec2)
	{
		if (vec1.size() != vec2.size()) return vec1;

		auto iter_vec2 = vec2.begin();
		for (auto& i : vec1) {
			i /= *iter_vec2++;
		}

		return vec1;
	}

	
	//for scalar multiplication version 1
	template <typename T>
	vector<T>& operator+=(vector<T>& vec1, vector_element_type_v<vector<T>>   scalar)
	{
		for (auto& i : vec1) {
			i += scalar;
		}

		return vec1;
	}

	template <typename T>
	vector<T>& operator-=(vector<T>& vec1, vector_element_type_v<vector<T>>   scalar)
	{
		for (auto& i : vec1) {
			i -= scalar;
		}

		return vec1;
	}

	template <typename T>
	vector<T>& operator*=(vector<T>& vec1, vector_element_type_v<vector<T>>   scalar)
	{
		for (auto& i : vec1) {
			i *= scalar;
		}

		return vec1;
	}

	template <typename T>
	vector<T>& operator/=(vector<T>& vec1, vector_element_type_v<vector<T>>   scalar)
	{
		for (auto& i : vec1) {
			i /= scalar;
		}

		return vec1;
	}

	template <typename T>
	vector<T> operator+(const vector<T>& vec1, vector_element_type_v<vector<T>> scalar)
	{
		vector<T> res = vec1;
		for (auto& i : res) {
			i += scalar;
		}
		return res;
	}

	template <typename T>
	vector<T> operator-(const vector<T>& vec1, vector_element_type_v<vector<T>>  scalar)
	{
		vector<T> res = vec1;
		for (auto& i : res) {
			i -= scalar;
		}
		return res;
	}

	template <typename T>
	vector<T> operator*(const vector<T>& vec1, vector_element_type_v<vector<T>>  scalar)
	{
		vector<T> res = vec1;
		for (auto& i : res) {
			i *= scalar;
		}
		return res;
	}

	template<typename T>
	vector<T> operator/(const vector<T>& vec1, vector_element_type_v<vector<T>>  scalar)
	{
		vector<T> res = vec1;
		for (auto& i : res) {
			i /= scalar;
		}
		return res;
	}


	//for scalar multiplication version 2
	template <typename T, std::enable_if_t<is_vector_v<T>, bool> = true>
	vector<T>& operator+=(vector<T>& vec1, T scalar)
	{
		for (auto& i : vec1) {
			i += scalar;
		}

		return vec1;
	}

	template <typename T, std::enable_if_t<is_vector_v<T>, bool> = true>
	vector<T>& operator-=(vector<T>& vec1, T scalar)
	{
		for (auto& i : vec1) {
			i -= scalar;
		}

		return vec1;
	}

	template <typename T, std::enable_if_t<is_vector_v<T>, bool> = true>
	vector<T>& operator*=(vector<T>& vec1, T scalar)
	{
		for (auto& i : vec1) {
			i *= scalar;
		}

		return vec1;
	}

	template <typename T, std::enable_if_t<is_vector_v<T>, bool> = true>
	vector<T>& operator/=(vector<T>& vec1, T scalar)
	{
		for (auto& i : vec1) {
			i /= scalar;
		}

		return vec1;
	}

	template <typename T, std::enable_if_t<is_vector_v<T>, bool> = true>
	vector<T> operator+(const vector<T>& vec1, T scalar)
	{
		vector<T> res = vec1;
		for (auto& i : res) {
			i += scalar;
		}
		return res;
	}

	template <typename T, std::enable_if_t<is_vector_v<T>, bool> = true>
	vector<T> operator-(const vector<T>& vec1, T scalar)
	{
		vector<T> res = vec1;
		for (auto& i : res) {
			i -= scalar;
		}
		return res;
	}

	template <typename T, std::enable_if_t<is_vector_v<T>, bool> = true>
	vector<T> operator*(const vector<T>& vec1, T scalar)
	{
		vector<T> res = vec1;
		for (auto& i : res) {
			i *= scalar;
		}
		return res;
	}

	template<typename T, std::enable_if_t<is_vector_v<T>, bool> = true>
	vector<T> operator/(const vector<T>& vec1, T scalar)
	{
		vector<T> res = vec1;
		for (auto& i : res) {
			i /= scalar;
		}
		return res;
	}
	
	#pragma endregion


	template<typename type, size_t size>
	std::vector<type> zeros() {
		return std::vector<type>(size, type());
	}

	template<typename type, size_t size, size_t... nums, std::enable_if_t<is_vector_v<VectorNd<sizeof...(nums), type>>, bool> = true>
	VectorNd<sizeof...(nums) + 1llu, type> zeros() {
		return VectorNd<sizeof...(nums) + 1llu, type>{size, zeros<type, nums...>()};
	}


	template<typename type, type init, size_t size>
	std::vector<type> nums() {
		return std::vector<type>(size, init);
	}

	template<typename type, type init, size_t size, size_t... sizes, std::enable_if_t<is_vector_v<VectorNd<sizeof...(sizes), type>>, bool> = true>
	VectorNd<sizeof...(sizes) + 1llu, type> nums() {
		return VectorNd<sizeof...(sizes) + 1llu, type>{size, nums<type, init, sizes...>()};
	}


	template<typename T, std::enable_if_t<!is_vector_v<T>, bool> = true>
	void mask(vector<T>& vec)
	{
		for (auto& i : vec) {
			i = T();
		}
	}

	template<typename T, std::enable_if_t<is_vector_v<T>, bool> = true>
	void mask(vector<T>& vec)
	{
		for (auto& i : vec) {
			mask(i);
		}
	}


	template<typename T, std::enable_if_t<!is_vector_v<T>, bool> = true>
	void vSet(vector<T>& vec, T arg)
	{
		for (auto& i : vec) {
			i = arg;
		}
	}

	template<typename T, std::enable_if_t<is_vector_v<T>, bool> = true>
	void vSet(vector<T>& vec, vector_element_type_v<T> arg)
	{
		for (auto& i : vec) {
			vSet(i, arg);
		}
	}


	template <typename T, std::enable_if_t<!is_vector_v<T>, bool> = true>
	T vSum(const vector<T>& vec) {
		T sum = T();
		for (auto& i : vec) {
			sum += i;
		}
		return sum;
	}

	template <typename T, std::enable_if_t<is_vector_v<T>, bool> = true>
	T vSum(const vector<T>& vec) {
		T sum = vec[0];
		mask(sum);

		for (auto& i : vec) {
			sum += i;
		}
		return sum;
	}

	template <typename T, std::enable_if_t<!is_vector_v<T>, bool> = true>
	T vProduct(const vector<T>& vec) {
		T pduct = static_cast<T>(1);
		for (auto& i : vec) {
			pduct *= i;
		}
		return pduct;
	}

	template <typename T, std::enable_if_t<is_vector_v<T>, bool> = true>
	T vProduct(const vector<T>& vec) {
		T pduct = vec[0];
		vSet(pduct,static_cast<vector_element_type_v<T>>(1));

		for (auto& i : vec) {
			pduct *= i;
		}
		return pduct;
	}

	template <typename T>
	T max(const vector<T>& vec, function<bool(T, T)> comp = [](T a, T b) {return a > b; })
	{
		T maxval = vec[0];
		for (size_t i = 1; i<vec.size(); ++i) {
			if (!comp(maxval, vec[i])) maxval = vec[i];
		}

		return maxval;
	}

	template <typename T>
	T min(const vector<T>& vec, function<bool(T, T)> comp = [](T a, T b) {return a < b; })
	{
		T minval = vec[0];
		for (size_t i = 1; i < vec.size(); ++i) {
			if (!comp(minval, vec[i])) minval = vec[i];
		}

		return minval;
	}

	template <typename T>
	size_t argmax(const vector<T>& vec, function<bool(T, T)> comp = [](T a, T b) {return a > b; })
	{
		size_t maxarg = 0;
		for (size_t i = 1; i < vec.size(); ++i) {
			if (!comp(vec[maxarg], vec[i])) maxarg = i;
		}

		return maxarg;
	}

	template <typename T>
	size_t argmin(const vector<T>& vec, function<bool(T, T)> comp = [](T a, T b) {return a < b; })
	{
		size_t minarg = 0;
		for (size_t i = 1; i < vec.size(); ++i) {
			if (!comp(vec[minarg], vec[i])) minarg = i;
		}

		return minarg;
	}

	template <typename T>
	vector<T> loc(const vector<T>& vec, size_t start, size_t end, size_t interval)
	{
		if (start < 0 || end < 0 || interval == 0 || start > vec.size() - 1 || end > vec.size() - 1 || 
			(interval > 0 && end < start) || (interval < 0 && end > start)) return vec;

		vector<T> res;
		res.reserve((end - start) / interval);

		if (start < end) {
			for (; start < end; start += interval) {
				res.push_back(vec[start]);
			}
		}
		else if (end < start)
		{
			for (; start > end; start += interval) {
				res.push_back(vec[start]);
			}
		}

		return res;
	}

	template <typename T>
	vector<T> range(T start, T end, T interval)
	{
		if (interval == 0 || (interval > 0 && end < start) || (interval < 0 && end > start))
			return {};

		vector<T> res;

		if (start < end) {
			for (; start < end; start += interval) {
				res.push_back(start);
			}
		}
		else if (start > end) {
			for (; start > end; start += interval) {
				res.push_back(start);
			}
		}

		return res;
	}


	template<typename T, std::enable_if_t<!is_vector_v<T>, bool> = true>
	std::ostream& operator << (std::ostream& out, const vector<T>& vec)
	{
		size_t vdim = dim_v<vector<T>>;
		if (vdim == print_vector_helper) print_vector_helper = 0;
		else if (vdim > print_vector_helper) print_vector_helper = vdim;

		std::cout << "[";
		for (size_t i = 0; i < vec.size(); ++i) {
			std::cout << vec[i];
			if (i != vec.size() - 1) { std::cout << ", "; }
		}
		std::cout << "]";

		if (vdim == print_vector_helper) {
			std::cout << std::endl;
			print_vector_helper = 0;
		}

		return out;
	}

	
	template<typename T, std::enable_if_t<is_vector_v<T>,bool> = true>
	std::ostream& operator << (std::ostream& out, const vector<T>& vec)
	{
		size_t vdim = dim_v<vector<T>>;

		if (vdim == print_vector_helper) print_vector_helper = 0;
		else if (vdim > print_vector_helper) print_vector_helper = vdim;

		std::cout << "[";
		for (size_t i = 0; i < vec.size(); ++i) {
			if (i != 0) {
				for(size_t j = 0; j < print_vector_helper - vdim + 1; ++j)
				std::cout << " ";
			}

			std::cout << vec[i];

			if (i != vec.size() - 1) {
				std::cout << "," << std::endl;
				for (size_t j = 0; j < vdim / 3; ++j) std::cout << std::endl;
			}
			
		}
		std::cout << "]";

		if (vdim == print_vector_helper) { 
			std::cout << std::endl;
			print_vector_helper = 0; 
		}

		return out;
	}
	
	
	template<typename T, std::enable_if_t<std::is_scalar_v<T>, bool> = true>
	T sqrt(T arg)
	{
		return static_cast<T>(std::sqrt(arg));
	}

	template<typename T, std::enable_if_t<is_vector_v<vector<T>>, bool> = true>
	vector<T> sqrt(const vector<T>& vec) {
		vector<T> res = vec;
		for (auto& i : res) {
			i = sqrt(i);
		}
		return res;
	}


	template<typename T>
	T mean(const vector<T>& vec) {
 		return vSum(vec) / static_cast<vector_element_type_v<vector<T>>>(vec.size());
	}

	
	template<typename T>
	T stdev(const vector<T>& vec) {
		T m = mean(vec);
		
		return SupportVector::sqrt(mean(vec * vec) - mean(vec) * mean(vec));

		/*
		auto s = vSum((vec - m) * (vec - m));

		return SupportVector::sqrt(s / static_cast<vector_element_type_v<vector<T>>>(vec.size()));
		*/

		/*
		T s = vec[0];
		mask(s);
		for (auto& i : vec) {
			s += (i - m) * (i - m);
		}*/
	}

	template<typename T>
	T stdev(const vector<T>& vec, const T m) {

		return SupportVector::sqrt(mean(vec * vec) - mean(vec) * mean(vec));

		/*
		auto s = vSum((vec - m) * (vec - m));

		return SupportVector::sqrt(s / static_cast<vector_element_type_v<vector<T>>>(vec.size()));
		*/

		/*
		T s = vec[0];
		mask(s);
		for (auto& i : vec) {
			s += (i - m) * (i - m);
		}*/
	}

};

#endif

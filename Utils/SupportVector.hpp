#ifndef __SUPPORTVECTOR__
#define __SUPPORTVECTOR__

#include <vector>
#include <functional>


using std::vector;
using std::function;


namespace SupportVector {
	#pragma region support vectorize operation
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
	vector<T> operator+(const vector<T>& vec1, T scalar)
	{
		vector<T> res = vec1;
		for (auto& i : res) {
			i += scalar;
		}
		return res;
	}

	template <typename T>
	vector<T> operator-(const vector<T>& vec1, T scalar)
	{
		vector<T> res = vec1;
		for (auto& i : res) {
			i -= scalar;
		}
		return res;
	}

	template <typename T>
	vector<T> operator*(const vector<T>& vec1, T scalar)
	{
		vector<T> res = vec1;
		for (auto& i : res) {
			i *= scalar;
		}
		return res;
	}

	template<typename T>
	vector<T> operator/(const vector<T>& vec1, T scalar)
	{
		vector<T> res = vec1;
		for (auto& i : res) {
			i /= scalar;
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

	template <typename T>
	vector<T>& operator+=(vector<T>& vec1, T scalar)
	{
		for (auto& i : vec1) {
			vec1 += scalar;
		}

		return vec1;
	}

	template <typename T>
	vector<T>& operator-=(vector<T>& vec1, T scalar)
	{
		for (auto& i : vec1) {
			vec1 -= scalar;
		}

		return vec1;
	}

	template <typename T>
	vector<T>& operator*=(vector<T>& vec1, T scalar)
	{
		for (auto& i : vec1) {
			vec1 *= scalar;
		}

		return vec1;
	}

	template <typename T>
	vector<T>& operator/=(vector<T>& vec1, T scalar)
	{
		for (auto& i : vec1) {
			vec1 /= scalar;
		}

		return vec1;
	}
	#pragma endregion

	template <typename T>
	T vSum(const vector<T>& vec) {
		T sum = T();
		for (auto& i : vec) {
			sum += i;
		}
		return sum;
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
};

#endif

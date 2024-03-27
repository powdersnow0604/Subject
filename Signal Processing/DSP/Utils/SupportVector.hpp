#ifndef __SUPPORTVECTOR__
#define __SUPPORTVECTOR__

#include <vector>

using std::vector;

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
};

#endif

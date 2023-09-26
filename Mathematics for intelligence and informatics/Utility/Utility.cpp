#include "Utility.h"
#include <algorithm>
#include <cmath>

namespace BasicAi {
	namespace Utility {

		DataModel StandardScaler::transform(const DataModel& dm)
		{
			std::vector<std::vector<double>> scaled_input(dm.input->size(), std::vector<double>(dm[0].size()));
			auto mean_dm = mean(dm);
			auto stdev_dm = stdev(dm, mean_dm);

			for (size_t i = 0; i < dm.size; ++i) {
				for (size_t j = 0; j < dm[0].size(); ++j) {
					scaled_input[i][j] = (dm[i][j] - mean_dm[j]) / stdev_dm[j];
				}
			}

			return { scaled_input, *dm.target };
		}

		std::vector<double> BasicAi::Utility::mean(const DataModel& dm)
		{
			std::vector<double> sum(dm[0].size(), 0.);

			for (auto& i : *dm.input) {
				auto iter = i.cbegin();
				std::transform(sum.begin(), sum.end(), sum.begin(), [&iter](double& n) {return n + *iter++; });
			}

			std::transform(sum.begin(), sum.end(), sum.begin(), [&dm](double& n) {return n / dm.input->size(); });

			return sum;
		}

		std::vector<double> BasicAi::Utility::stdev(const DataModel& dm, const std::vector<double>& mean)
		{
			std::vector<double> sum(dm[0].size(), 0);
			std::vector<double> stdev(dm[0].size(), 0);

			for (auto& i : *dm.input) {
				auto iter = i.cbegin();
				auto iter_mean = mean.cbegin();
				std::transform(sum.begin(), sum.end(), sum.begin(), [&iter, &iter_mean](const double& n)
					//{return n + std::pow(*iter++ - *iter_mean++, 2); });
					{return n + (*iter++ - *iter_mean++) * (*iter - *iter_mean); });
			}

			std::transform(sum.begin(), sum.end(), stdev.begin(), [&dm](const double& n) {return std::sqrt(n / dm.size); });

			return stdev;
		}

		double BasicAi::Utility::EuclidianDistance(const std::vector<double>& point1, const std::vector<double>& point2)
		{
			if (point1.size() != point2.size()) return 999999999999999.;

			double sum = 0.;
			for (auto i = point1.begin(), j = point2.begin(); i != point1.end(); ++i, ++j)
			{
				sum += (*i - *j) * (*i - *j);
			}

			return std::sqrt(sum);
		}

		std::vector<double> BasicAi::Utility::absDiff(const std::vector<double>& point1, const std::vector<double>& point2)
		{
			if (point1.size() != point2.size()) return {};
			std::vector<double> res;
			res.reserve(point1.size());

			for (size_t i = 0; i < point1.size(); ++i) {
				res.push_back(std::abs(point1[i] - point2[i]));
			}

			return res;
		}

		#pragma region support vectorize operation
		vector<double> operator+(const vector<double>& vec1, const vector<double>& vec2)
		{
			if (vec1.size() != vec2.size()) return {};

			vector<double> res;
			res.reserve(vec1.size());

			for (size_t i = 0; i < vec1.size(); ++i) {
				res.push_back(vec1[i] + vec2[i]);
			}

			return res;
		}

		vector<double> operator-(const vector<double>& vec1, const vector<double>& vec2)
		{
			if (vec1.size() != vec2.size()) return {};

			vector<double> res;
			res.reserve(vec1.size());

			for (size_t i = 0; i < vec1.size(); ++i) {
				res.push_back(vec1[i] - vec2[i]);
			}

			return res;
		}

		vector<double> operator*(const vector<double>& vec1, const vector<double>& vec2)
		{
			if (vec1.size() != vec2.size()) return {};

			vector<double> res;
			res.reserve(vec1.size());

			for (size_t i = 0; i < vec1.size(); ++i) {
				res.push_back(vec1[i] * vec2[i]);
			}

			return res;
		}

		vector<double> operator/(const vector<double>& vec1, const vector<double>& vec2)
		{
			if (vec1.size() != vec2.size()) return {};

			vector<double> res;
			res.reserve(vec1.size());

			for (size_t i = 0; i < vec1.size(); ++i) {
				res.push_back(vec1[i] / vec2[i]);
			}

			return res;
		}

		vector<double> operator+(const vector<double>& vec1, double scalar)
		{
			vector<double> res = vec1;
			for (auto& i : res) {
				i += scalar;
			}
			return res;
		}

		vector<double> operator-(const vector<double>& vec1, double scalar)
		{
			vector<double> res = vec1;
			for (auto& i : res) {
				i -= scalar;
			}
			return res;
		}

		vector<double> operator*(const vector<double>& vec1, double scalar)
		{
			vector<double> res = vec1;
			for (auto& i : res) {
				i *= scalar;
			}
			return res;
		}

		vector<double> operator/(const vector<double>& vec1, double scalar)
		{
			vector<double> res = vec1;
			if (scalar != 0) {
				for (auto& i : res) {
					i /= scalar;
				}
			}
			return res;
		}

		vector<double>& operator+=(vector<double>& vec1, const vector<double>& vec2)
		{
			if (vec1.size() != vec2.size()) return vec1;
			
			auto iter_vec2 = vec2.begin();
			for (auto& i : vec1) {
				i += *iter_vec2++;
			}

			return vec1;
		}

		vector<double>& operator-=(vector<double>& vec1, const vector<double>& vec2)
		{
			if (vec1.size() != vec2.size()) return vec1;

			auto iter_vec2 = vec2.begin();
			for (auto& i : vec1) {
				i -= *iter_vec2++;
			}

			return vec1;
		}

		vector<double>& operator*=(vector<double>& vec1, const vector<double>& vec2)
		{
			if (vec1.size() != vec2.size()) return vec1;

			auto iter_vec2 = vec2.begin();
			for (auto& i : vec1) {
				i *= *iter_vec2++;
			}

			return vec1;
		}

		vector<double>& operator/=(vector<double>& vec1, const vector<double>& vec2)
		{
			if (vec1.size() != vec2.size()) return vec1;

			auto iter_vec2 = vec2.begin();
			for (auto& i : vec1) {
				i /= *iter_vec2++;
			}

			return vec1;
		}
		
		vector<double>& operator+=(vector<double>& vec1, double scalar)
		{
			for (auto& i : vec1) {
				vec1 += scalar;
			}

			return vec1;
		}

		vector<double>& operator-=(vector<double>& vec1, double scalar)
		{
			for (auto& i : vec1) {
				vec1 -= scalar;
			}

			return vec1;
		}

		vector<double>& operator*=(vector<double>& vec1, double scalar)
		{
			for (auto& i : vec1) {
				vec1 *= scalar;
			}

			return vec1;
		}

		vector<double>& operator/=(vector<double>& vec1, double scalar)
		{
			if (scalar != 0) {
				for (auto& i : vec1) {
					vec1 /= scalar;
				}
			}
			return vec1;
		}
		#pragma endregion
	}
}
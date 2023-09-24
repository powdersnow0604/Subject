#include "ADC.h"

#include <cmath>

namespace DSP {
	namespace ADC {
		double default_analog_signal(double x, double freq, double A, double theta)
		{
			return A * std::sin(2 * M_PI * freq * x + theta);
		}

		std::vector<double> range(double start, double end, double interval)
		{
			if (start < end && interval < 0 || start > end && interval > 0) return {};

			std::vector<double> res;
			res.reserve(static_cast<size_t>((end - start) / interval));

			for (double i = start; i < end; i += interval) {
				res.push_back(i);
			}

			return res;
		}

		smpling_result_pack sampling(double s_freq, double start, double end, double analog_max_freq, double analog_A, double analog_theta,
			 std::function<double(double, double, double, double)> func)
		{
			const double s_period = 1 / s_freq;
			std::vector<std::vector<double>> res;
			res.reserve(static_cast<size_t>((end - start) / s_period));

			std::vector<double> rng = range(start, end, s_period);

			for (auto& i : rng) {
				res.push_back({ i, func(i, analog_max_freq, analog_A, analog_theta) });
			}

			return {res, s_freq, s_period};
		}

		quantization_result_pack quantization(const std::vector<std::vector<double>>& points, unsigned int qbit, double maxval, bool only_positive)
		{
			double step_size;
			std::vector<std::vector<double>> res = points;

			if (only_positive) step_size = maxval / (std::pow(2, qbit) - 1);
			else step_size = maxval / (std::pow(2, qbit-1) - 1);

			for (auto& i : res) {
				i[1] = static_cast<double>(std::llround(i[1] / step_size));
			}

			return { res, qbit };
		}

		std::vector<std::vector<double>> inv_quantization(const std::vector<std::vector<double>>& points, unsigned int qbit, double maxval, bool only_positive)
		{
			double step_size;
			std::vector<std::vector<double>> res = points;

			if (only_positive) step_size = maxval / (std::pow(2, qbit) - 1);
			else step_size = maxval / (std::pow(2, qbit - 1) - 1);

			for (auto& i : res) {
				i[1] = i[1] * step_size;
			}

			return res;
		}


	}
}
#ifndef __ADC__
#define __ADC__

#include <vector>
#include <functional>

namespace DSP {
	constexpr double M_PI = 3.14159265358979323846;

	namespace ADC {
		double default_analog_signal(double x, double freq = 1., double A = 1., double theta = 0.);

		std::vector<double> range(double start, double end, double interval);

		typedef struct {
			std::vector<std::vector<double>> points;
			const double s_freq;
			const double s_period;
		}smpling_result_pack;

		smpling_result_pack sampling(double s_freq, double start, double end, 
			double analog_max_freq = 1., double analog_A = 1., double analog_theta = 0.,
			std::function<double(double, double, double, double)> func = default_analog_signal);

		typedef struct {
			std::vector<std::vector<double>> points;
			const unsigned int qbit;
		}quantization_result_pack;
		
		quantization_result_pack quantization(const std::vector<std::vector<double>>& points, unsigned int qbit, double maxval, bool only_positive);

		std::vector<std::vector<double>> inv_quantization(const std::vector<std::vector<double>>& points, unsigned int qbit, double maxval, bool only_positive);

	}
}

#endif
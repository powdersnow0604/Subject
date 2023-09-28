#include "ADC.h"
#include <cmath>


namespace DSP {
	namespace ADC {

		void ADC::set(double s_freq_, unsigned int qbit_)
		{
			s_freq = s_freq_;
			s_period = 1 / s_freq_;
			qbit = qbit_;
		}

		signal ADC::sampling(double start, double end, std::function<double(double)> func)
		{
			signal res;

			vector<double> rng = range(start, end, s_period);
			res.size = rng.size();

			for (auto& i : rng) {
				res.A.push_back(func(i));
				res.T.push_back(i);
			}

			return res;
		}

		signal ADC::quantization(const signal& points)
		{
			double step_size;
			signal res = points;

			//calc min and max;
			maxval = points[0]; minval = points[0];
			for (auto& i : points.A) {
				if (maxval < i) maxval = i;
				else if (minval > i) minval = i;
			}
			
			step_size = (maxval - minval)/ (std::pow(2, qbit) - 1);

			for (auto& i : res.A) {
				i = static_cast<double>(std::llround((i-minval) / step_size));
			}

			return res;
		}

		signal ADC::inv_quantization(const signal& points)
		{
			double step_size;
			signal res = points;

			step_size = (maxval - minval) / (std::pow(2, qbit) - 1);

			for (auto& i : res.A) {
				i = i * step_size + minval;
			}

			return res;
		}

		signal ADC::error(const signal& sample, const signal& inv_quant)
		{
			if (sample.size != inv_quant.size) return {};
			signal res;
			res.size = sample.size;
			res.A.reserve(res.size);
			res.T = inv_quant.T;

			for (size_t i = 0; i < res.size; ++i) {
				res.A.push_back(sample[i] - inv_quant[i]);
			}

			return res;
		}
	}
}
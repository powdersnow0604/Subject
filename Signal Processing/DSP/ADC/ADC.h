#ifndef __ADC__
#define __ADC__

#include "Signals.h"
#include "Utility.h"
#include <vector>
#include <functional>

using namespace DSP::Signals;
using namespace DSP::Utility;


namespace DSP {
	namespace ADC {
		template <double freq = 1., double A = 1., double theta = 0.> 
		double default_analog_signal(double x)
		{
			return A * std::sin(2 * M_PI * freq * x + theta);
		}

		class ADC {
			double s_freq;
			double s_period;
			unsigned int qbit;
			double maxval;
			double minval;
		public:
			void set(double s_freq_, unsigned int qbit_);

			signal sampling(double start, double end, std::function<double(double)> func = default_analog_signal<1.,1.,0.>);

			signal quantization(const signal& points);

			signal inv_quantization(const signal& points);

			signal error(const signal& sample, const signal& inv_quant);

			ADC(): s_freq(1), s_period(1), qbit(1), maxval(0.), minval(0.) {}
			ADC(double s_freq_, unsigned int qbit_):s_freq(s_freq_), s_period(1/s_freq_), qbit(qbit_), maxval(0.), minval(0.) {}
		};
	}
}

#endif
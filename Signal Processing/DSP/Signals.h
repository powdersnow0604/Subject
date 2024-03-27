#ifndef __SIGNALS__
#define __SIGNALS__

#include <functional>
#include <map>
#include <vector>


using std::map;
using std::vector;
using Vector2D = vector<vector<double>>;
using Vector = vector<double>;

namespace DSP {
	namespace Signals {
		typedef struct signal_ {
			Vector A;
			Vector T;
			size_t size;
			double& operator[](size_t i);
			double& operator()(size_t i);
			double operator[](size_t i) const;
			double operator()(size_t i) const;

			signal_(): size(0) {}
		}signal;

		Vector2D signal2vector2d(const signal& sig);

		signal delta(int start, int end, double delay = 0., double A = 1.);

		signal step(int start, int end, double delay = 0., double A = 1.);

		signal exponential(int start, int end, double x, double A = 1.);

		signal makeSignal(int start, int end, std::function<double(double)> func);

		Vector range(double start, double end, double interval);
	}
}

#endif
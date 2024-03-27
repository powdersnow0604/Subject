#include "Signals.h"
#include <cmath>

namespace DSP {
	namespace Signals {

		double& signal_::operator[](size_t i)
		{
			return A[i];
		}

		double& signal_::operator()(size_t i)
		{
			return T[i];
		}

		double signal_::operator[](size_t i) const {
			return A[i];
		}

		double signal_::operator()(size_t i) const
		{
			return T[i];
		}

		Vector2D signal2vector2d(const signal& sig) 
		{
			Vector2D res;
			res.reserve(sig.size);

			for (size_t i = 0; i < sig.size; ++i) {
				res.push_back({ sig(i), sig[i] });
			}
	
			return res;
		}

		signal delta(int start, int end, double delay, double A)
		{
			signal res;

			vector<double> rng = range(start, end, 1);

			res.size = rng.size();
			res.A.reserve(res.size);
			res.T.reserve(res.size);

			for (auto& i : rng) {
				res.A.push_back(i == delay ? A : 0 );
				res.T.push_back(i);
			}

			return res;
		}

		signal step(int start, int end, double delay, double A)
		{
			signal res;

			vector<double> rng = range(start, end, 1);

			res.size = rng.size();
			res.A.reserve(res.size);
			res.T.reserve(res.size);

			for (auto& i : rng) {
				res.A.push_back(i >= delay ? A : 0);
				res.T.push_back(i);
			}

			return res;
		}

		signal exponential(int start, int end, double x, double A)
		{
			signal res;
			vector<double> rng = range(start, end, 1);

			res.size = rng.size();
			res.A.reserve(res.size);
			res.T.reserve(res.size);

			for (auto& i : rng) {
				res.A.push_back(std::pow(x, i) * A);
				res.T.push_back(i);
			}

			return res;
		}

		signal makeSignal(int start, int end, std::function<double(double)> func)
		{
			signal res;
			vector<double> rng = range(start, end, 1);

			res.size = rng.size();
			res.A.reserve(res.size);
			res.T.reserve(res.size);

			for (auto& i : rng) {
				res.A.push_back(func(i));
				res.T.push_back(i);
			}

			return res;
		}

		Vector range(double start, double end, double interval)
		{
			if (start < end && interval < 0 || start > end && interval > 0) return {};

			Vector res;
			res.reserve(static_cast<size_t>((end - start) / interval));

			for (double i = start; i < end; i += interval) {
				res.push_back(i);
			}

			return res;
		}

	}
}

	
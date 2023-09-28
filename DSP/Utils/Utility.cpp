#include "Utility.h"

namespace DSP {
	namespace Utility {
		
		signal diff(const signal& sig)
		{
			signal res;
			res.size = sig.size - 1;

			for (size_t i = 0; i < res.size; ++i) {
				res.A.push_back(sig[i + 1] - sig[i]);
				res.T.push_back(sig(i+1));
			}

			return res;
		}

		signal cummSum(const signal& sig)
		{
			signal res;
			res.T = sig.T;
			res.size = sig.size;

			double sum = 0.;
			for (auto& i : sig.A) {
				sum += i;
				res.A.push_back(sum);
			}

			return res;
		}
	}
}
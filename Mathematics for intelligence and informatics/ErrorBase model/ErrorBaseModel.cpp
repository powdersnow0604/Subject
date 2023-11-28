#include "ErrorBaseModel.h"
#include <iostream>


namespace BasicAi {
	namespace ErrorBaseModel {
		
		void Perceptron::fit(const DataModel& Dm)
		{
			weights = random_vector(Dm[0].size()+1, -0.2, 0.2);
			w0 = weights.back(); weights.resize(Dm[0].size());
			errors.reserve(epoch);
			vector<size_t> shuff = range(0ull, Dm.size, 1ull);
			double error, update;

			for (size_t i = epoch; i != 0; --i) {
				error = 0;
				shuffle(shuff);

				for (size_t j = 0; j < Dm.size; ++j) {
					update = eta * (Dm(shuff[j]) - predict({ Dm[shuff[j]] })[0]);
					weights += Dm[shuff[j]] * update;
					w0 += update;
					error += update != 0.;
				}
				errors.push_back(error);
			}
		}


		Vector Perceptron::predict(const InputModel& In)
		{
			Vector res; res.reserve(In.size);
			for (size_t i = 0; i < In.size; ++i) {
				res.push_back(vSum(weights * In[i]) + w0);
			}

			return res;
		}


		double Perceptron::score(const DataModel& Dm)
		{
			auto res = predict(Dm.getInput());
			size_t cnt = 0;

			for (size_t i = 0; i < Dm.size; ++i) {
				if (res[i] == Dm(i)) ++cnt;
			}

			return (double)cnt / Dm.size;
		}
		
	}
}
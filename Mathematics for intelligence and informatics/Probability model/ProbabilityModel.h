#ifndef __PROBABILITYMODEL_H__
#define __PROBABILITYMODEL_H__

#include "pch.h"

using namespace BasicAi::DataModels;

namespace BasicAi {
	namespace Probability {

		class NaiveBayes
		{
			map<double,vector<map<double, double>>> dist;
			map<double, double> prior;
			map<Vector, double> norm_term;
			Vector p_res_value;
		public:
			void fit(const DataModel& Dm);
			Vector predict(const InputModel& In);
			Vector2D predictProbs(const InputModel& In);
			void printDist(void);
			Vector getPredictValue();
		};
	}
}
#endif
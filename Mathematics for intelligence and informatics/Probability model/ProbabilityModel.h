#ifndef __PROBABILITYMODEL_H__
#define __PROBABILITYMODEL_H__

#include "Model_base.h"
#include "pch.h"

using namespace BasicAi::DataModels;

namespace BasicAi {
	namespace Probability {

		class NaiveBayes
		{
			map<double,vector<map<double, double>>> dist; // prior[level] ·Î ³ª´²¾ß Á¤È®ÇÑ È®·ü
			map<double, double> prior; // size ·Î ³ª´²¾ß Á¤È®ÇÑ È®·ü
			map<Vector, double> norm_term; // size ·Î ³ª´²¾ß Á¤È®ÇÑ È®·ü
			Vector p_res_value;
			size_t size;
		public:
			void fit(const DataModel& Dm);
			Vector predict(const InputModel& In, bool update = false);
			Vector2D predictProbs(const InputModel& In, bool update = false);
			void printDist(void);
			Vector getPredictValue();
			double score(const DataModel& Dm);
			NaiveBayes(): size(0) {}
		};
	}
}
#endif
#ifndef __PROBABILITYMODEL_H__
#define __PROBABILITYMODEL_H__

#include "Model_base.h"
#include "pch.h"

using namespace BasicAi::DataModels;

namespace BasicAi {
	namespace Probability {

		class NaiveBayes
		{
			map<double,vector<map<double, size_t>>> dist; // prior[level] ·Î ³ª´²¾ß Á¤È®ÇÑ È®·ü
			map<double, size_t> prior; // size ·Î ³ª´²¾ß Á¤È®ÇÑ È®·ü
			map<Vector, size_t> norm_term; // size ·Î ³ª´²¾ß Á¤È®ÇÑ È®·ü
			Vector2D p_res_value;
			size_t size;
		public:
			void fit(const DataModel& Dm);
			Vector predict(const InputModel& In, bool update = false);
			Vector2D predictProbs(const InputModel& In, bool update = false);
			void printDist(void);
			Vector2D getPredictValue();
			double score(const DataModel& Dm);
			NaiveBayes(): size(0) {}
		};


		class GaussianNB
		{
			map<double,Vector> mean;
			map<double, Vector> stdev;
			map<double, size_t> prior; // sample_num À¸·Î ³ª´²¾ß Á¤È®ÇÑ È®·ü
			Vector2D p_res_value;
			map<double, Vector> sum_of_value;
			size_t sample_num;

			double Gaussian(double mean, double stdev, double x);

		public:
			void fit(const DataModel& Dm);
			Vector predict(const InputModel& In, bool update = false);
			Vector2D predict_proba(const InputModel& In, bool update = false);
			void printStatus(void);
			Vector2D getPredictValue();
			double score(const DataModel& Dm);
			GaussianNB() : sample_num(0) {}
		};
	}
}
#endif
#ifndef __SIMILARITY_BASE__
#define __SIMILARITY_BASE__


#include "Model_base.h"

namespace BasicAi {
	namespace Similarity {

		class KNN : BasicAi::model {
			BasicAi::DataModels::DataModel data;
			unsigned int K;
		public:
			void fit(const BasicAi::DataModels::DataModel&);
			BasicAi::DataModels::TargetModel predict(const BasicAi::DataModels::InputModel&);
			double score(const BasicAi::DataModels::DataModel&);

			KNN(unsigned int k = 3) : K(k) {}
		};

		class KMean  {
			unsigned int K;

		public:

			typedef struct point_{
				double distance;
				double cls;
				point_(double dis_, double cls_) : distance(dis_), cls(cls_) {}
			}point;

			typedef struct result_pack {
				BasicAi::DataModels::DataModel dm;
				std::vector<std::vector<double>> centroid;
				result_pack(BasicAi::DataModels::DataModel& dm_, decltype(centroid) cent_): dm(dm_), centroid(cent_){}
			}KMean_res_pack;

			KMean_res_pack predict(const BasicAi::DataModels::InputModel&, unsigned int max_iter = 100, double threshold = 1e-4);
			KMean(unsigned int k = 3) : K(k) {}
		};

	}
}
#endif

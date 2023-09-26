#ifndef __SIMILARITY_BASE__
#define __SIMILARITY_BASE__

#include "Model_base.h"
#include <map>

using namespace BasicAi::DataModels;

namespace BasicAi {
	namespace Similarity {


		class KNN : model {
			DataModel data;
			unsigned int K;
		public:
			void fit(const DataModel&);
			TargetModel predict(const InputModel&);
			double score(const DataModel&);

			KNN(unsigned int k = 3) : K(k) {}
		};


		enum KMean_initial_modes
		{
			KMEAN_RANDOM = 0,
			KMEAN_FIT = 1
		};

		class KMean {
			unsigned int K;

		public:

			typedef struct calc_cent_ {
				vector<double> features;
				double cnt;
				calc_cent_(vector<double> f_, double c_) : features(f_), cnt(c_) {}
				calc_cent_() : cnt(0) {}
			}calc_cent;

			typedef struct result_pack {
				DataModel dm;
				vector<std::vector<double>> centroid;
				result_pack(DataModel& dm_, decltype(centroid) cent_) : dm(dm_), centroid(cent_) {}
			}KMean_res_pack;


			KMean(unsigned int k = 3) : K(k) {}

			KMean_res_pack predict(const BasicAi::DataModels::InputModel&, unsigned int max_iter = 100, double threshold = 1e-4, int init_mode = KMEAN_RANDOM);

			void print(const KMean_res_pack& res, bool print_d_features_flag = true, bool print_centroid_flag = true);

			void changeK(unsigned int K_);

			//by class and centroid, key of class is class, key of centroid is concat(67,class)
			map<double, vector<vector<double>>> classify(const KMean_res_pack& pack);
		};

	}
}
#endif

#ifndef __SIMILARITY_BASE__
#define __SIMILARITY_BASE__

#include "Model_base.h"
#include <random>
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


		enum class KMEAN_INIT
		{
			KMEAN_RANDOM = 0,
			KMEAN_KMEANPP = 1
		};

		class KMeans {
			unsigned int K;
			unsigned int max_iter;
			double threshold;
			KMEAN_INIT init;
			unsigned char n_init;
			int random_seed;

			double costFunction(const DataModel& dm, const Vector2D& Centroids);
			Vector2D KMeanspp(const InputModel& In, std::mt19937& mt);

		public:
			Vector2D cluster_centers;
			double inertia;
			Vector labels;

			typedef struct calc_cent_ {
				vector<double> features;
				double cnt;
				calc_cent_(vector<double> f_, double c_) : features(f_), cnt(c_) {}
				calc_cent_() : cnt(0) {}
			}calc_cent;

			typedef struct paramPack_ {
				unsigned int K;
				unsigned int max_iter;
				double threshold;
				KMEAN_INIT init;
				unsigned char n_init;
				int random_seed;

				paramPack_(unsigned int k = 3, unsigned int max_iter_ = 100, double threshold_ = 1e-4, unsigned int n_init_ = 10,
					KMEAN_INIT init_mode = KMEAN_INIT::KMEAN_RANDOM, int random_seed_ = -1) :
					K(k), max_iter(max_iter_), threshold(threshold_), init(init_mode), n_init(n_init_),
					random_seed(random_seed_) {}

			}paramPack;


			KMeans(unsigned int k = 3, unsigned int max_iter_ = 100, double threshold_ = 1e-4, unsigned int n_init_ = 10,
				KMEAN_INIT init_mode = KMEAN_INIT::KMEAN_RANDOM, int random_seed_ = -1) :
				K(k), max_iter(max_iter_), threshold(threshold_), init(init_mode), n_init(n_init_),
			random_seed(random_seed_), inertia(-1.){}

			void fit(const InputModel&);

			Vector predict(const InputModel&);

			void print(const InputModel& In, bool print_d_features_flag = true, bool print_centroid_flag = true);

			void setParam(paramPack pack);

			paramPack getParam();

			//by class and centroid, key of class is class, key of centroid is concat(67,class)
			map<double, vector<vector<double>>> classify(const InputModel& In);

			static Vector2D elbowMethod(const InputModel& In, unsigned int max_K, unsigned int max_iter_ = 100, 
				double threshold_ = 1e-4, unsigned int n_init_ = 1, int random_seed_ = -1);
		};

	}
}
#endif

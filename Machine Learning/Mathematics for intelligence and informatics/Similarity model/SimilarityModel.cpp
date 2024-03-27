#include "SimilarityModel.h"
#include "Utility.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <limits>

using namespace BasicAi::DataModels;
using namespace BasicAi::Utility;

namespace BasicAi {
	namespace Similarity {
		//KNN
		void KNN::fit(const DataModel& sample)
		{
			data = sample;
		}

		TargetModel KNN::predict(const BasicAi::DataModels::InputModel& Im)
		{
			std::vector<double> result;
			std::vector<std::vector<double>> distance;
			result.reserve(Im.size);

			std::map<double, size_t> m;

			for (auto& i : *Im.input) {
				distance.reserve(data.size);

				for (size_t j = 0; j < data.size; ++j) {
					distance.push_back({ data(j), EuclidianDistance(i, data[j]) });
				}

				std::sort(distance.begin(), distance.end(), [](const std::vector<double> opr1, const std::vector<double> opr2)
					{return opr1[1] < opr2[1]; });

				for (size_t j = 0; j < K; ++j) {
					m[distance[j][0]] += 1;
				}


				auto max_class = Utility::max(m.cbegin(), m.cend(),
					std::function<bool(const decltype(m.cbegin())&, const decltype(m.cbegin())&)>
					([](const auto& opr1, const auto& opr2) -> bool {return opr1->second > opr2->second; }));

				result.push_back(max_class->first);

				distance.clear();
				m.clear();
			}

			return result;
		}

		double KNN::score(const DataModel& Dm)
		{
			double cnt = 0.;

			auto pred = predict(*Dm.input);

			for (size_t i = 0; i < Dm.size; ++i) {
				if (pred[i] == Dm(i)) ++cnt;
			}

			return cnt / Dm.size;
		}


		//kMean
		void KMeans::fit (const InputModel& In)
		{
			unsigned int max_cnt, init_cnt = 0;
			//constexpr double rand_upper_bound = std::numeric_limits<double>::max(), rand_lower_bound = std::numeric_limits<double>::lowest();
			size_t min_arg;

			std::mt19937 gen;
			//std::uniform_real_distribution<double> udist(rand_upper_bound, rand_lower_bound);

			Vector2D centroid(K,Vector(In[0].size()));
			Vector2D prev_centroid;
			std::map<double, calc_cent> calc_helper;

			DataModel res(*In.input, std::vector<double>(In.size));
			const Vector zeros(res[0].size());
			Vector inertias; inertias.reserve(n_init);
			vector<Vector2D> save_centroids; save_centroids.reserve(n_init);
			Vector2D save_labels; save_labels.reserve(n_init);
			
			//input data의 max, min 찾기
			Vector input_max(res[0]), input_min(res[0]);

			for (auto& i : *res.input) {
				for (size_t j = 0; j < res[0].size(); ++j) {
					if (i[j] > input_max[j]) input_max[j] = i[j];
					else if (i[j] < input_min[j]) input_min[j] = i[j];
				}
			}

			std::uniform_real_distribution<double> udist(max(input_max), min(input_min));

			//random 시드 설정
			if(random_seed == -1){
				std::random_device rd;
				gen.seed(rd());
			}
			else {
				gen.seed(random_seed);
			}
			

			while (init_cnt++ < n_init) {
				//centroid 초기화
				if (init == KMEAN_INIT::KMEAN_RANDOM) {
					//임의의 discriptive features 생성
					for (auto& i : centroid) {
						for (size_t j = 0; j < i.size(); ++j) {
							i[j] = udist(gen);
						}
					}
				}
				if (init == KMEAN_INIT::KMEAN_KMEANPP) {
					centroid = KMeanspp(In, gen);
				}

				max_cnt = 0;
				while (max_cnt++ < max_iter) {
					//초기화
					for (double i = 0; i < K; ++i) {
						calc_helper[i] = { zeros, 0 };
					}

					//각 features에 대해
					for (size_t i = 0; i < res.size; ++i) {
						//각 centroid에 대해
						size_t min = 0;
						double d = EuclidianDistance(res[i], centroid[0]);
						for (size_t j = 1; j < K; ++j) {
							double temp = EuclidianDistance(res[i], centroid[j]);
							if (d > temp) {
								min = j;
								d = temp;
							}
						}

						res(i) = (double)min;
					}

					//new centroid 계산
					prev_centroid = centroid;

					for (size_t i = 0; i < res.size; ++i) {
						calc_helper[res(i)].features += res[i];
						calc_helper[res(i)].cnt += 1;
					}

					auto calc_iter = calc_helper.begin();
					std::for_each(centroid.begin(), centroid.end(), [&calc_iter, &input_max, &input_min, this](auto& vec)
						{
							if ((calc_iter->second).cnt != 0) {
								vec = (calc_iter->second).features / (calc_iter->second).cnt;
							}
							
							else {
								for (size_t i = 0; i < vec.size(); ++i) {
									vec[i] = input_min[i] + (input_max[i] - input_min[i]) * calc_iter->first / (this->K - 1);
								}
							}

							++calc_iter;
						});


					for (size_t i = 0; i < centroid.size(); ++i) {
						double moved = EuclidianDistance(centroid[i], prev_centroid[i]);
						if (moved > threshold) goto continue_iter;
					}

					break;

				continue_iter:
					continue;

				}

				inertias.push_back(costFunction(res, centroid));
				save_centroids.push_back(centroid);
				save_labels.push_back(*res.target);
			}

			min_arg = argmin(inertias);
			cluster_centers = save_centroids[min_arg];
			labels = save_labels[min_arg];
			inertia = inertias[min_arg];
		}

		Vector KMeans::predict(const InputModel& In)
		{
			Vector res; res.reserve(In.size);

			for (auto& point : *In.input) {
				size_t min_arg = 0;
				double min_distance = EuclidianDistance(cluster_centers[0], point);
				for (size_t i = 1; i < cluster_centers.size(); ++i) {
					double temp = EuclidianDistance(cluster_centers[i], point);
					if (min_distance > temp) {
						min_arg = i;
						min_distance = temp;
					}
				}

				res.push_back(static_cast<double>(min_arg));
			}

			return res;
		}

		void KMeans::setParam(paramPack pack)
		{
			K = pack.K;
			max_iter = pack.max_iter;
			threshold = pack.threshold;
			n_init = pack.n_init;
			init = pack.init;
			random_seed = pack.random_seed;
		}

		KMeans::paramPack KMeans::getParam()
		{
			return { K, max_iter, threshold, n_init, init, random_seed };
		}

		void KMeans::print(const InputModel& In, bool print_d_features_flag, bool print_centroid_flag)
		{
			using std::cout;
			using std::endl;

			if (print_d_features_flag) {
				cout << "<    descriptive features    >" << endl;
				cout << "[feature]  [class]" << endl;
				cout << "-------------------" << endl;
				for (size_t i = 0; i < In.size; ++i) {
					cout << "[ ";
					for (auto& j : In[i]) {
						cout << j << " ";
					}
					cout << "]: " << labels[i] << endl;
				}

				cout << endl;
			}


			if (print_centroid_flag) {
				cout << "<    centroid    >" << endl;
				cout << "[centroid number]  [position]" << endl;
				cout << "-----------------------------" << endl;
				for (size_t i = 0; i < cluster_centers.size(); ++i) {
					cout << "centroid " << i << ": [ ";
					for (auto& j : cluster_centers[i]) {
						cout << j << " ";
					}
					cout << "]" << endl;
				}

				cout << endl;
			}
		}

		map<double, vector<vector<double>>> KMeans::classify(const InputModel & In)
		{
			map<double, vector<vector<double>>> res;

			for (size_t i = 0; i < In.size; ++i) {
				res[labels[i]].push_back(In[i]);
			}

			int index = 0;
			for (auto& i : cluster_centers) {
				res[static_cast<double>(pow(10, (int)std::log10(K)+1) + index++)].push_back(i);
			}
			
			return res;
		}

		double KMeans::costFunction(const DataModel& dm, const Vector2D& Centroids)
		{
			double sum = 0.;
			for (size_t i = 0; i < dm.size; ++i) {
				sum += std::pow(EuclidianDistance(dm[i], Centroids[static_cast<size_t>(dm(i))]), 2);
			}

			return sum;
		}

		Vector2D KMeans::elbowMethod(const InputModel& In, unsigned int max_K, unsigned int max_iter_,
			double threshold_, unsigned int n_init_, int random_seed_)
		{
			if (max_K < 2) return{};

			vector<unsigned int> k_range = range(2u, max_K+1, 1u);
			Vector2D res; res.reserve(k_range.size());

			KMeans model(2, max_iter_, threshold_, n_init_, KMEAN_INIT::KMEAN_RANDOM, random_seed_ == -1 ? std::random_device()() : random_seed_);
			auto params = model.getParam();

			for (auto& i : k_range) {
				params.K = i;
				model.setParam(params);
				model.fit(In);
				res.push_back({ static_cast<double>(i), model.inertia });
			}

			return res;
		}

		Vector2D KMeans::KMeanspp(const InputModel& In, std::mt19937& mt)
		{
			double d_sum, rnd;
			Vector2D centroids; centroids.reserve(K);
			Vector probs(In.size+1);

			std::uniform_int_distribution<int> ui_dist(0, (int)In.size - 1);
			std::uniform_real_distribution<double> ur_dist(0, 1);

			centroids.push_back(In[ui_dist(mt)]);

			while (centroids.size() != K) {
				for (size_t i = 0; i < In.size; ++i) {
					double d = EuclidianDistance(In[i], centroids[0]);
					for (size_t j = 1; j < centroids.size(); ++j) {
						double temp = EuclidianDistance(In[i], centroids[j]);
						if (d > temp) d = temp;
					}

					probs[i+1] = d * d;
				}

				d_sum = vSum(probs);

				probs /= d_sum;

				rnd = ur_dist(mt);
				for (size_t i = 0; i < In.size; ++i) {
					if (probs[i] <= rnd && rnd < probs[i + 1]) {
						centroids.push_back(In[i]);
						break;
					}
				}
			}

			return centroids;
		}

	}
}

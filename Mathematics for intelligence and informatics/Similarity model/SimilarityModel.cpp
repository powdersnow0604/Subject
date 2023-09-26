#include "SimilarityModel.h"
#include "Utility.h"
#include <algorithm>
#include <random>
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
		KMean::KMean_res_pack BasicAi::Similarity::KMean::predict
		(const InputModel& In, unsigned int max_iter, double threshold, int init_mode)
		{
			std::vector<std::vector<double>> centroid(K);
			std::vector<std::vector<double>> prev_centroid(K);
			std::map<double, calc_cent> calc_new_centroid;
			DataModel res(*In.input, std::vector<double>(In.size));
			double rand_upper_bound = std::numeric_limits<double>::max(), rand_lower_bound = std::numeric_limits<double>::min();

			const vector<double> zeros(res[0].size());
			
			
			//input data의 max, min 찾기
			std::vector<double> input_max(res[0]), input_min(res[0]);

			for (auto& i : *res.input) {
				for (size_t j = 0; j < res[0].size(); ++j) {
					if (i[j] > input_max[j]) input_max[j] = i[j];
					else if (i[j] < input_min[j]) input_min[j] = i[j];
				}
			}

			if (init_mode == KMEAN_FIT) {
				rand_upper_bound = input_max[0];
				rand_lower_bound = input_min[0];
				for (size_t i = 1; i < res[0].size(); ++i) {
					if (rand_upper_bound < input_max[i]) rand_upper_bound = input_max[i];
					if (rand_lower_bound > input_min[i]) rand_lower_bound = input_min[i];
				}
			}

			//random 범위 설정
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<double> udist(rand_lower_bound, rand_upper_bound);

			//임의의 discriptive features 생성
			for (auto& i : centroid) {
				i.reserve(res[0].size());
				for (size_t j = 0; j < res[0].size(); ++j) {
					i.push_back(udist(gen));
				}
			}

			while (0 < max_iter--) {
				//초기화
				for (double i = 0; i < K; ++i) {
					calc_new_centroid[i] = {zeros, 0};
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
					calc_new_centroid[res(i)].features += res[i];
					calc_new_centroid[res(i)].cnt += 1;
				}

				auto calc_iter = calc_new_centroid.begin();
				std::for_each(centroid.begin(), centroid.end(), [&calc_iter, &input_max, &input_min, this](auto& vec) 
					{
						if ((calc_iter->second).cnt != 0) {
							vec = (calc_iter->second).features / (calc_iter->second).cnt;
						}
						else {
							for (size_t i = 0; i < vec.size(); ++i) {
								vec[i] = input_min[i] + (input_max[i] - input_min[i]) * (calc_iter->first + 1) / this->K;
							}
						}

						++calc_iter;
					});


				for (size_t i = 0; i < centroid.size(); ++i) {
					auto absdiff = absDiff(centroid[i], prev_centroid[i]);
					if (vector_sum(absdiff) > threshold) goto continue_iter;
				}

				break;

			continue_iter:
				continue;

			}


			return { res, centroid };
		}

		void KMean::changeK(unsigned int K_) { K = K_; }

		void KMean::print(const KMean_res_pack& res, bool print_d_features_flag, bool print_centroid_flag)
		{
			using std::cout;
			using std::endl;

			if (print_d_features_flag) {
				cout << "<    descriptive features    >" << endl;
				cout << "[feature]  [class]" << endl;
				cout << "-------------------" << endl;
				for (size_t i = 0; i < res.dm.size; ++i) {
					cout << "[" << res.dm[i][0] << " " << res.dm[i][1] << "]: " << res.dm(i) << endl;
				}

				cout << endl;
			}


			if (print_centroid_flag) {
				cout << "<    centroid    >" << endl;
				cout << "[centroid number]  [position]" << endl;
				cout << "-----------------------------" << endl;
				for (size_t i = 0; i < res.centroid.size(); ++i) {
					cout << "centroid " << i << ": [ ";
					for (auto& j : res.centroid[i]) {
						cout << j << " ";
					}
					cout << "]" << endl;
				}

				cout << endl;
			}
		}

		map<double, vector<vector<double>>> KMean::classify(const KMean_res_pack& pack)
		{
			map<double, vector<vector<double>>> res;

			for (size_t i = 0; i < pack.dm.size; ++i) {
				res[pack.dm(i)].push_back(pack.dm[i]);
			}

			int index = 0;
			for (auto& i : pack.centroid) {
				res[(double)std::stoi(std::to_string(67) + std::to_string(index++))].push_back(i);
			}
			
			return res;
		}
	}
}

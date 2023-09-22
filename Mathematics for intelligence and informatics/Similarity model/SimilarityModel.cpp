#include "SimilarityModel.h"
#include "Utility.h"
#include <algorithm>
#include <map>
#include <random>
#include <cmath>
#include <iostream>

using namespace BasicAi::DataModels;
using namespace BasicAi::Utility;

void BasicAi::Similarity::KNN::fit(const DataModel& sample)
{
	data = sample;
}


BasicAi::DataModels::TargetModel BasicAi::Similarity::KNN::predict(const BasicAi::DataModels::InputModel& Im)
{
	std::vector<double> result;
	std::vector<std::vector<double>> distance;
	result.reserve(Im.input.size());

	std::map<double, size_t> m;

	for (auto& i : Im.input) {
		distance.reserve(data.target.size());

		for (size_t j = 0; j < data.input.size(); ++j) {
			distance.push_back({ data.target[j], EuclidianDistance(i, data.input[j])});
		}

		std::sort(distance.begin(), distance.end(), [](const std::vector<double> opr1, const std::vector<double> opr2)
			{return opr1[1] < opr2[1]; });

		for (size_t i = 0; i < K; ++i) {
			m[distance[i][0]] += 1;
		}

		auto max = m.cbegin();
		for (auto i = ++(m.cbegin()); i != m.end(); ++i) {
			if (max->second < i->second) max = i;
		}

		result.push_back(max->first);

		distance.clear();
		m.clear();
	}

	return result;
}


double BasicAi::Similarity::KNN::score(const BasicAi::DataModels::DataModel& Dm) 
{
	double cnt = 0.;

	auto pred = predict(Dm.input);

	for (size_t i = 0; i < Dm.target.size(); ++i) {
		if (pred.target[i] == Dm.target[i]) ++cnt;
	}

	return cnt / Dm.target.size();
}


BasicAi::Similarity::KMean::KMean_res_pack BasicAi::Similarity::KMean::predict
(const BasicAi::DataModels::InputModel& In, unsigned int max_iter, double threshold)
{
	unsigned int iter_cnt = 0;
	DataModel res;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> udist(0, 10000);
	std::vector<std::vector<double>> centroid(K);
	std::vector<std::vector<double>> prev_centroid(K);
	std::vector<point> distance;
	std::map<double, std::vector<double>> calc_new_centroid;


	res.input = In.input;
	res.target.resize(res.input.size());

	
	//임의의 discriptive features 생성
	for (auto& i : centroid) {
		i.reserve(res.input[0].size());
		for (size_t j = 0; j < res.input[0].size(); ++j) {
			i.push_back(udist(gen));
		}
	}
	

	while (iter_cnt++ < max_iter) {		
		//각 features에 대해
		for (size_t i = 0; i < res.input.size(); ++i) {
			distance.reserve(centroid.size());

			//각 centroid에 대해
			for (size_t j = 0; j < K; ++j) {
				distance.push_back({EuclidianDistance(res.input[i], centroid[j]), (double)j});
			}

			auto min = distance.begin();
			for (auto iter = distance.begin()+1; iter != distance.end(); ++iter) {
				if (iter->distance < min->distance) min = iter;
			}

			res.target[i] = min->cls;

			distance.clear();
		}
		

		
		//new centroid 계산
		prev_centroid = centroid;

		for (size_t i = 0; i < K; ++i) {
			calc_new_centroid[i].resize(res.input[0].size() + 1);
		}

		for (size_t i = 0; i < res.target.size(); ++i) {
			for (size_t j = 0; j < res.input[0].size(); ++j) {
				calc_new_centroid[res.target[i]][j] += res.input[i][j];
			}
			calc_new_centroid[res.target[i]][res.input[0].size()] += 1;
		}

		
		auto calc_iter = calc_new_centroid.begin();
		std::for_each(centroid.begin(), centroid.end(), [&calc_new_centroid, &calc_iter] (auto& vec)
			{
				for (size_t i = 0; i < vec.size(); ++i) {
					vec[i] = (calc_iter->second)[i];
				}

				if ((calc_iter->second)[vec.size()] != 0) {
					for (auto& i : vec) {
						i /= (calc_iter->second)[vec.size()];
					}
				}

				++calc_iter;
			});

		
		double sum;
		for (size_t i = 0; i < centroid.size(); ++i) {
			sum = 0;
			auto absdiff = absDiff(centroid[i], prev_centroid[i]);
			for (auto& j : absdiff) {
				sum += j;
			}
			if (sum > threshold) goto continue_iter;
		}

		break;

	continue_iter:
		calc_new_centroid.clear();
		continue;
		
	}

	std::cout << "iter cnt = " << iter_cnt << std::endl;
	return KMean_res_pack( res, centroid );
}


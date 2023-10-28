#include "ProbabilityModel.h"
#include <iostream>
#include <set>
#include <conio.h>

namespace BasicAi {
	namespace Probability {
		static void insert(Vector& vec, double arg)
		{
			auto iter = vec.begin();
			for (; iter != vec.end(); ++iter) {
				if (*iter == arg) return;
				if (*iter > arg) break;
			}

			vec.insert(iter, arg);
		}

		void NaiveBayes::fit(const DataModel& Dm)
		{
			size_t noc = 1;
			size_t noc_temp;
			//vector<std::set<double>> dset;
			Vector2D dset;
			Vector2D temp_d;
			dset.resize(Dm[0].size());
			
			for (size_t i = 0; i < Dm.size; ++i) {
				if (dist[Dm(i)].size() == 0) dist[Dm(i)].resize(Dm[0].size());
				for (size_t j = 0; j<Dm[0].size(); ++j) {
					++dist[Dm(i)][j][Dm[i][j]];
					//dset[j].insert(Dm[i][j]);
					insert(dset[j], Dm[i][j]);
				}

				++norm_term[Dm[i]];
				++prior[Dm(i)];
			}
			
			for (size_t i = 0; i < dset.size(); ++i) {
				noc *= dset[i].size();
			}
			noc_temp = noc;

			temp_d.resize(noc, Vector(dset.size()));

			for (size_t i = 0; i < dset.size(); ++i) {
				noc_temp /= dset[i].size();
				for (size_t j = 0; j < noc; ++j) {
					temp_d[j][i] = dset[i][(j / noc_temp) % dset[i].size()];
				}
			}


			for (size_t i = 0; i < temp_d.size(); ++i) {
				if (norm_term[temp_d[i]] == 0.) continue;
				norm_term[temp_d[i]] /= Dm.size; // / norm_term[temp_d[i]];
			}


			/*
			for (auto& [level, v] : dist) {
				for (auto& d : v) {
					for (auto& [df, value] : d) {
						value /= prior[level];
					}
				}
			}*/

			for (auto& [level, v] : dist) {
				for (size_t i = 0; i< v.size(); ++i) {
					for (auto& df_list : dset[i]) {
						v[i][df_list] /= prior[level];
					}
				}
			}



			for (auto& [k, v] : prior) {
				v /= Dm.size;
			}
			
		}

		Vector NaiveBayes::predict(const InputModel& In)
		{
			double max_class;
			double max_value;

			Vector res; res.reserve(In.size);
			p_res_value.clear();

			for (size_t i = 0; i < In.size; ++i) { //for each samples

				max_class = -1;
				max_value = -1;
				for (auto& [level, df] : dist) { // for class
		
					double product = 1;
					for (size_t j = 0; j < In[i].size(); ++j) { //for descriptive features
						
						if (df[j].find(In[i][j]) == df[j].end()) {
							product = 0;
							break;
						}
						product *= df[j][In[i][j]];
					}
					
					product *= prior[level];

					if (max_value < product) {
						max_class = level;
						max_value = product;
					}
				}

				p_res_value.push_back(max_value);
				res.push_back(max_class);
			}

			return res;
		}

		Vector2D NaiveBayes::predictProbs(const InputModel& In)
		{
			Vector2D res(In.size);

			for (size_t i = 0; i < In.size; ++i) { //for each samples

				res[i].reserve(prior.size());
				double sum = 0;

				for (auto& [level, df] : dist) { // for class

					double product = 1;
					for (size_t j = 0; j < In[0].size(); ++j) { //for descriptive features
						if (df[j].find(In[i][j]) == df[j].end()) {
							product = 0;
							break;
						}
						product *= df[j][In[i][j]];
					}

					product *= prior[level]; //* norm_term[In[i]];
					sum += product;
					res[i].push_back(product);
				}

				res[i] /= sum;
			}

			return res;
		}

		void NaiveBayes::printDist(void)
		{
			for (auto& [level, df] : dist) {
				std::cout << "class: " << level << " prob: " << prior[level] << std::endl;
				for (auto& vec : df) {
					for (auto& [k, v] : vec) {
						std::cout << "df = [ " << k << ", " << v << " }" << std::endl;
					}

					std::cout << std::endl;
				}
			}
		}

		Vector NaiveBayes::getPredictValue()
		{
			return p_res_value;
		}
	}
}
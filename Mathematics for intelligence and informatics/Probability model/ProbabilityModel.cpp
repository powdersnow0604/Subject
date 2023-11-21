#include "ProbabilityModel.h"
#include <iostream>
#include <set>
#include <conio.h>
#include <cmath>

namespace BasicAi {
	namespace Probability {
		using namespace BasicAi;

		//Naive Bayes
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
			dist.clear();
			prior.clear();
			norm_term.clear();
			p_res_value.clear();

			size_t noc = 1;
			size_t noc_temp;
			Vector2D dset;
			Vector2D temp_d;
			dset.resize(Dm[0].size());

			for (size_t i = 0; i < Dm.size; ++i) {
				if (dist[Dm(i)].size() == 0) dist[Dm(i)].resize(Dm[0].size());
				for (size_t j = 0; j < Dm[0].size(); ++j) {
					++dist[Dm(i)][j][Dm[i][j]];
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
				//if (norm_term[temp_d[i]] == 0.) continue;
				norm_term[temp_d[i]]; // /= Dm.size;
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
				for (size_t i = 0; i < v.size(); ++i) {
					for (auto& df_list : dset[i]) {
						v[i][df_list]; ///= prior[level];
					}
				}
			}


			/*
			for (auto& [k, v] : prior) {
				v /= Dm.size;
			}
			*/

			size = Dm.size;
			
		}

		Vector NaiveBayes::predict(const InputModel& In, bool update)
		{
			double max_class;
			double max_value;

			Vector res; res.reserve(In.size);
			p_res_value.clear();
			p_res_value.resize(In.size);

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
						product *= (double)df[j][In[i][j]] / prior[level]; 
					}
					
					product *= (double)prior[level] / size; 

					p_res_value[i].push_back(product);

					if (max_value < product) {
						max_class = level;
						max_value = product;
					}
				}

				//update model
				if (update) {
					++size;
					++prior[max_class];
					for (size_t j = 0; j < In[i].size(); ++j) {
						++dist[max_class][j][In[i][j]];
					}
					++norm_term[In[i]];
				}

				res.push_back(max_class);
			}

			return res;
		}

		Vector2D NaiveBayes::predictProbs(const InputModel& In, bool update)
		{
			double max_class;
			double max_value;
			Vector2D res(In.size);

			for (size_t i = 0; i < In.size; ++i) { //for each samples

				res[i].reserve(prior.size());
				double sum = 0;
				max_class = -1;
				max_value = -1;

				for (auto& [level, df] : dist) { // for class

					double product = 1;
					for (size_t j = 0; j < In[0].size(); ++j) { //for descriptive features
						if (df[j].find(In[i][j]) == df[j].end()) {
							product = 0;
							break;
						}
						product *= (double)df[j][In[i][j]] / prior[level]; 
					}

					product *= (double)prior[level] / size; 
					sum += product;
					res[i].push_back(product);

					if (max_value < product) {
						max_class = level;
						max_value = product;
					}
				}

				res[i] /= sum;

				//update model
				if (update) {
					++size;
					++prior[max_class];
					for (size_t j = 0; j < In[i].size(); ++j) {
						++dist[max_class][j][In[i][j]];
					}
					++norm_term[In[i]];
				}
			}

			return res;
		}

		void NaiveBayes::printDist(void)
		{
			for (auto& [level, df] : dist) {
				std::cout << "class: " << level << " prob: " << prior[level] << std::endl;
				for (auto& vec : df) {
					for (auto& [k, v] : vec) {
						std::cout << "df = [ " << k << ", " << v << " ]" << std::endl;
					}

					std::cout << std::endl;
				}
			}
		}

		Vector2D NaiveBayes::getPredictValue()
		{
			return p_res_value;
		}

		double NaiveBayes::score(const DataModel& Dm)
		{
			size_t i;
			size_t cnt = 0;
			Vector res = predict(Dm.input);

			for (i = 0; i < Dm.size; ++i) {
				if (Dm(i) == res[i]) ++cnt;
			}

			return (double)cnt / Dm.size;
		}


		//GaussianNB
		double GaussianNB::Gaussian(double mean, double stdev, double x)
		{
			return std::exp(-(x - mean) * (x - mean) / (2 * stdev * stdev)) / (stdev * std::sqrt(2 * 3.14159265359));
		}


		void GaussianNB::fit(const DataModel& Dm)
		{
			mean.clear();
			stdev.clear();
			prior.clear();
			sum_of_value.clear();

			sample_num = Dm.size;
			map<double, Vector2D> split_by_class;

			for (size_t i = 0; i < sample_num; ++i) {
				split_by_class[Dm(i)].push_back(Dm[i]);
				++(prior[Dm(i)]);
			}

			for (auto& [k, v] : split_by_class) {
				mean[k] = SupportVector::mean(v);
				stdev[k] = SupportVector::stdev(v, mean[k]);
				sum_of_value[k] = SupportVector::vSum(v);
			}
		}


		Vector GaussianNB::predict(const InputModel& In, bool update)
		{
			double sum;
			double max_value;
			double max_class;

			Vector res; res.reserve(In.size);
			p_res_value.clear();  p_res_value.resize(In.size);

			for (size_t k = 0; k < In.size; ++k) { //for each sample
				
				max_value = -1000000;
				max_class = -1;

				p_res_value[k].reserve(In[0].size());

				for (auto& [cls, value] : sum_of_value) { //for each class
					sum = 0;

					for (size_t i = 0; i < In[0].size(); ++i) {
						sum += std::log((Gaussian(mean.at(cls)[i], stdev.at(cls)[i], In[k][i])));
					}
					sum += std::log((double)prior.at(cls) / sample_num);

					if (sum > max_value) {
						max_value = sum;
						max_class = cls;
					}

					p_res_value[k].push_back(sum);
				}

				if (update) {
					for (size_t i = 0; i < In[0].size(); ++i) {
						++sample_num;
						double& var = stdev[max_class][i];
						double& m = mean[max_class][i];
						size_t& p = prior[max_class];
						double& sov = sum_of_value[max_class][i];

						var *= var;
						var +=  2 * m * sum_of_value[max_class][i] - p * m * m;

						m = (p * m + In[k][i]) / (p + 1);
						++p;
						sov += In[k][i];

						var += In[k][i] - In[k][i] + m * sov * 2. + p * m * m;
					}
				}

				res.push_back(max_class);
			}

			return res;
		}


		Vector2D GaussianNB::predict_proba(const InputModel& In, bool update)
		{
			double product, total_sum;
			double max_value;
			double max_class;
			double temp;

			Vector2D res; res.resize(In.size);
			p_res_value.clear();  p_res_value.resize(In.size);

			for (size_t k = 0; k < In.size; ++k) { //for each sample
				total_sum = 0;
				max_value = -1000000;
				max_class = -1;

				p_res_value[k].reserve(In[0].size());
				res[k].reserve(In[0].size());

				for (auto& [cls, value] : sum_of_value) { //for each class
					product = 1;

					for (size_t i = 0; i < In[0].size(); ++i) {
						product *= Gaussian(mean.at(cls)[i], stdev.at(cls)[i], In[k][i]);
					}
					product *= (double)prior.at(cls) / sample_num;
					temp = std::log(product);

					if (temp > max_value) {
						max_value = temp;
						max_class = cls;
					}

					p_res_value[k].push_back(temp);
					res[k].push_back(product);
					total_sum += product;
				}

				if (update) {
					for (size_t i = 0; i < In[0].size(); ++i) {
						++sample_num;
						double& var = stdev[max_class][i];
						double& m = mean[max_class][i];
						size_t& p = prior[max_class];
						double& sov = sum_of_value[max_class][i];

						var *= var;
						var += 2 * m * sum_of_value[max_class][i] - p * m * m;

						m = (p * m + In[k][i]) / (p + 1);
						++p;
						sov += In[k][i];

						var += In[k][i] - In[k][i] + m * sov * 2. + p * m * m;
					}
				}

				res[k] /= total_sum;
			}

			return res;
		}


		void GaussianNB::printStatus(void)
		{
			std::cout << "[mean]" << std::endl;
			for (auto& [k, v] : mean) {
				std::cout << "class " << k << ": " << std::endl << v << std::endl;
			}

			std::cout << "[stdev]" << std::endl;
			for (auto& [k, v] : stdev) {
				std::cout << "class " << k << ": " << std::endl << v << std::endl;
			}
		}


		Vector2D GaussianNB::getPredictValue()
		{
			return p_res_value;
		}


		double GaussianNB::score(const DataModel& Dm)
		{
			size_t i;
			size_t cnt = 0;
			Vector res = predict(Dm.input);

			for (i = 0; i < Dm.size; ++i) {
				if (Dm(i) == res[i]) ++cnt;
			}

			return (double)cnt / Dm.size;
		}
	}
}
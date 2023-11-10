#include "InformationModel.h"
#include "cmath"
#include <iostream>
#include <random>
#include <algorithm>

namespace BasicAi {
	namespace InformationModel {

		//Decision Tree
		int DecisionTree::makeTree(const vector<size_t>& desf, const Vector2D& input, const Vector& target, node& node_, size_t depth)
		{
			//degenerate case1: train data is empty
			if (input.size() == 0 || target.size() == 0) return 1;

			//degenerate case2: all instances have the same level
			else if (is_homogeneous(target)) {
				node_.pred = target[0];
				node_.is_leafnode = true;
				max_depth = max_depth > depth ? max_depth : depth;
				return 0;
			}

			//calculate majority label
			auto res = count<double>(target);
			size_t max_num = 0;
			double max_f = 0;

			for (auto& [k, v] : res) {
				if (max_num < v) {
					max_num = v;
					max_f = k;
				}
			}

			node_.pred = max_f;

			//degenerate case3: descriptive features are empty
			if (desf.size() == 0) {
				node_.is_leafnode = true;
				max_depth = max_depth > depth ? max_depth : depth;
				return 0;
			}

			//degenerate case4: reached max level
			if (depth == max_depth) {
				node_.is_leafnode = true;
				max_depth = max_depth > depth ? max_depth : depth;
				return 0;
			}

			size_t arg_max = argmax(IG(desf, input, target));
			size_t best_d = desf[arg_max];
			node_.query_feature = best_d;
			
			map<double, DataModel> levels;
			for (size_t i = 0; i < input.size(); ++i) {
				DataModel& dm = levels[input[i][best_d]];
				dm.input->push_back(input[i]);
				dm.target->push_back(target[i]);
			}

			vector<size_t> new_desf = concatenate({ loc(desf, 0, arg_max, 1), loc(desf, arg_max + 1, desf.size(),1) });

			for (auto& [k, v] : levels) {
				if (makeTree(new_desf, v.getInput(), v.getTarget(), node_.childs[k], depth+1)) {
					node_.childs[k].pred = max_f;
					node_.is_leafnode = true;
					max_depth = max_depth > depth + 1 ? max_depth : depth + 1;
				}
			}

			return 0;
		}


		double DecisionTree::entropy(const Vector& data)
		{
			map<double, size_t> levels = count(data);
			double sum = 0, p;

			for (auto& [k, v] : levels) {
				p = (double)v / data.size();
				sum += p * std::log2(p);
			}

			return -sum;
		}


		double DecisionTree::rem(size_t desf, const Vector2D& input, const Vector& target)
		{
			map<double, double> levels;
			map<double, Vector> data;
			double sum = 0;

			for (size_t i = 0; i < input.size(); ++i) {
				++levels[input[i][desf]];
				data[input[i][desf]].push_back(target[i]);
			}

			for (auto& [k, v] : data) {
				sum += (levels[k] / input.size()) * entropy(v);
			}

			return sum;
		}


		Vector DecisionTree::IG(const vector<size_t>& desf, const Vector2D& input, const Vector& target)
		{
			double H = entropy(target);
			Vector ig; ig.reserve(desf.size());

			for (size_t i : desf) {
				ig.push_back(H - rem(i, input, target));
			}

			return ig;
		}


		void DecisionTree::fit(const DataModel& Dm)
		{
			if (Dm.size == 0) return;

			vector<size_t> desf = range<size_t>(0, Dm[0].size(), 1);
			const Vector2D& input = Dm.getInput();
			const Vector& target = Dm.getTarget();

			(void)makeTree(desf, input, target, root, 0);
		}


		double DecisionTree::predictHelper(const Vector& In, const node& node_)
		{
			if (node_.childs.size() == 0) return node_.pred;

			return predictHelper(In, node_.childs.at(In[node_.query_feature]));
		}


		Vector DecisionTree::predict(const InputModel& In)
		{
			Vector res; res.reserve(In.size);

			for (size_t i = 0; i < In.size; ++i) {
				res.push_back(predictHelper(In[i], root));
			}

			return res;
		}


		double DecisionTree::score(const DataModel& Dm)
		{
			auto res = predict(Dm.getInput());
			double cnt = 0;

			for (size_t i = 0; i < Dm.size; ++i) {
				if (res[i] == Dm(i)) ++cnt;
			}

			return cnt / Dm.size;
		}


		void DecisionTree::showTreeHelper(const node& node_, size_t cnt) const
		{
			if (node_.childs.size() == 0) {
				std::cout << "[leaf node]: pred = " << node_.pred << std::endl;
			}
			else {
				std::cout << "[non-leaf node]: queried feature = " << node_.query_feature << std::endl;
				for (auto& [k, v] : node_.childs) {
					for (size_t i = 0; i < cnt*2; ++i) std::cout << " ";
					std::cout << "\b|- " << "value = " << k << " -> ";
					showTreeHelper(v, cnt + 1);
				}
			}
		}


		void DecisionTree::showTree() const
		{
			std::cout << "- ";
			showTreeHelper(root, 0);
		}


		void DecisionTree::pruningHelper(node& node_, const DataModel& Dm, double threshold)
		{
			if (node_.is_leafnode == true) return;

			for (auto& [k, v] : node_.childs) {
				pruningHelper(v, Dm, threshold);
			}

			double err_T = 1 - score(Dm);
			size_t div = getLeavesNum(node_) - 1;
			node_.is_leafnode = true;
			double err_prune = 1 - score(Dm);

			double eval = (err_prune - err_T) / div;

			if (threshold > eval) node_.childs.clear();
			else node_.is_leafnode = false;
		}


		void DecisionTree::pruning(const DataModel& Dm, double threshold)
		{
			pruningHelper(root, Dm, threshold);
		}


		size_t DecisionTree::getNodeNum(const node& node_)
		{
			size_t sum = 1;
			for (auto& [k, v] : node_.childs) {
				sum += getNodeNum(v);
			}

			return sum;
		}


		size_t DecisionTree::getLeavesNum(const node& node_)
		{
			if (node_.childs.size() == 0) return 1;

			size_t sum = 0;
			for (auto& [k, v] : node_.childs) {
				sum += getNodeNum(v);
			}

			return sum;
		}


		//Ensemble
		void EnsembleModel::fit(const DataModel& Dm)
		{
			if (type == EnsembleType::BOOSTING) {
				fitBoosting(Dm);
			}
			else if (type == EnsembleType::BAGGING){
				fitBagging(Dm);
			}
		}


		void EnsembleModel::fitBoosting(const DataModel& Dm)
		{
			vector<double> weight(Dm.size, 1. / Dm.size);
			const Vector2D& input = Dm.getInput();
			const Vector& target = Dm.getTarget();
			vector<bool> misclassify; misclassify.resize(Dm.size);
			const vector<size_t>& for_select_k = range<size_t>(0, weight.size(), 1);

			size_t i, j, subset_size = static_cast<size_t>(std::sqrt(Dm.size));
			std::mt19937 gen{std::random_device()()};


			forest.resize(model_num);
			confidence_factor.reserve(model_num);

			for (i = 0; i < model_num; ++i) {
				const vector<size_t>& ind = select_k_randomly_by_weight(for_select_k, weight, subset_size);
				const Vector2D& df = loc_by_list(input, ind);
				const Vector& tf = loc_by_list(target, ind);

				forest[i].fit({ df,tf });
				
				const Vector& pred = forest[i].predict(input);

				double epsilon = 0;

				for (j = 0; j < Dm.size; ++j) {
					if (pred[j] != target[j]) {
						epsilon += weight[j];
						misclassify[j] = true;
					}
					misclassify[j] = false;
				}

				for (j = 0; j < Dm.size; ++j) {
					if(misclassify[j]) weight[j] = weight[j] / (2. * epsilon);
					else weight[j] = weight[j] / (2. * (1. - epsilon));
				}

				confidence_factor[i] = std::log((1. - epsilon) / epsilon) / 2.;
				
			}
		}


		void EnsembleModel::fitBagging(const DataModel& Dm)
		{
			std::mt19937 gen{ std::random_device()() };
			std::uniform_int_distribution<size_t> dist(0, Dm.size - 1);
			const Vector2D& input = Dm.getInput();
			const Vector& target = Dm.getTarget();
			const vector<size_t>& for_selecting_df = range<size_t>(0, Dm[0].size(), 1);
			size_t df_num = Dm[0].size() * 3 >> 2;
			Vector2D subset_input;
			Vector subset_target;

			size_t i, j, rand_temp;

			forest.resize(model_num);

			for (i = 0; i < model_num; ++i) {
				subset_input.clear(); subset_input.reserve(Dm.size);
				subset_target.clear(); subset_target.reserve(Dm.size);

				vector<size_t> selected_df = select_k_randomly(for_selecting_df, df_num);
				for (j = Dm.size; j != 0; --j) {
					rand_temp = dist(gen);
					subset_input.push_back(loc_by_list(input[rand_temp], selected_df));
					subset_target.push_back(target[rand_temp]);
				}

				forest[i].fit({ subset_input, subset_target });
			}

		}


		Vector EnsembleModel::predict(const InputModel& In)
		{
			if (type == EnsembleType::BOOSTING) {
				return predictBoosting(In);
			}
			else if (type == EnsembleType::BAGGING) {
				return {};
			}
		}


		Vector EnsembleModel::predictBoosting(const InputModel& In)
		{
			vector<map<double, double>> pred; pred.resize(In.size);
			vector<double> res; res.reserve(In.size);
			size_t i, j;

			for (i = 0; i < model_num; ++i) {
				auto res = forest[i].predict(In);
				for (j = 0; j < In.size; ++j) {
					pred[j][res[j]] += confidence_factor[i];
				}
			}

			for (i = 0; i < In.size; ++i) {
				res.push_back(max(pred[i]).first);
			}

			return res;
			
		}


		Vector EnsembleModel::predictBagging(const InputModel& In)
		{

		}


		double EnsembleModel::score(const DataModel& Dm)
		{
			Vector pred = predict(Dm.getInput());

			size_t cnt = 0;

			for (size_t i = 0; i < Dm.size; ++i) {
				std::cout << pred[i] << " ";
				if (pred[i] == Dm(i)) ++cnt;
			}
			std::cout << std::endl;

			return (double)cnt / Dm.size;
		}


		void EnsembleModel::showTree(size_t i)
		{
			if (i >= model_num) return;

			forest[i].showTree();
		}


		void EnsembleModel::showTreeAll()
		{
			for (size_t i = 0; i < model_num; ++i) {
				std::cout << "[model" << i << "]" << std::endl;
				forest[i].showTree();
				std::cout << std::endl;
			}
		}
	}
}
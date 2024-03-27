#ifndef __INFORMATIONMODEL_H__
#define __INFORMATIONMODEL_H__

#include "DataModel.h"
#include "pch.h"

namespace BasicAi {
	namespace InformationModel {
		using namespace BasicAi::DataModels;

		class DecisionTree  {
			struct node {
				size_t query_feature;
				double pred;  //final prediction for leaf node, majority class for non-leaf node
				bool is_leafnode; //for pruning
				map<double, node> childs;
				node(): query_feature(0), pred(0), is_leafnode(false) {}
			}root;

			size_t tree_depth;
			size_t max_depth;

			int makeTree(const vector<size_t>& desf, const Vector2D& input, const Vector& target, node& node_, size_t depth);
			double entropy(const Vector& data);
			double rem(size_t desf, const Vector2D& input, const Vector& target);
			Vector IG(const vector<size_t>& desf, const Vector2D& input, const Vector& target);
			void showTreeHelper(const node& node_, size_t cnt) const;
			double predictHelper(const Vector& In, const node& node_);
			void pruningHelper(node& node_, const DataModel& Dm, double threshold);
			size_t getNodeNum(const node& node_);
			size_t getLeavesNum(const node& node_);

		public:
			DecisionTree(size_t max_depth_ = 10): tree_depth(0), max_depth(max_depth_) {}
			void fit(const DataModel& Dm);
			Vector predict(const InputModel& In);
			double score(const DataModel& Dm);
			void showTree() const;
			void pruning(const DataModel& Dm, double threshold = 1e-4);
		};

		enum class EnsembleType {
			BOOSTING,
			BAGGING
		};

		class EnsembleModel {
			vector<DecisionTree> forest;
			Vector confidence_factor;
			EnsembleType type;
			size_t model_num;

			void fitBoosting(const DataModel& Dm);
			Vector predictBoosting(const InputModel& In);
			void fitBagging(const DataModel& Dm);
			Vector predictBagging(const InputModel& In);

		public:
			EnsembleModel(size_t model_num_ = 10, EnsembleType type_ = EnsembleType::BAGGING):model_num(model_num_), type(type_) {}
			void fit(const DataModel& Dm);
			Vector predict(const InputModel& In);
			double score(const DataModel& Dm);
			void showTree(size_t i);
			void showTreeAll();
		};
	}
}

#endif
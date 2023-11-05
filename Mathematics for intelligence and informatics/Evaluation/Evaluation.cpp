#include "Evaluation.h"

#include <map>

namespace BasicAi {
	namespace Evaluation {

		static std::map<double, Vector2D> splitByClass(const DataModel& Dm)
		{
			std::map<double, Vector2D> input_map;

			for (size_t i = 0; i < Dm.size; ++i) {
				input_map[Dm(i)].push_back(Dm[i]);
			}

			return input_map;
		}

		splitResult test_train_split(const DataModel& Dm, double test_ratio, double class_std)
		{
			if (test_ratio > 1. || test_ratio < 0.) return {};

			auto res = splitByClass(Dm);
			size_t test_num;
			Vector test_target;
			Vector train_target;
			Vector2D test_input;
			Vector2D train_input;

			for (auto& [k, v] : res) {
				test_num = static_cast<size_t>(v.size() * test_ratio + 0.5);
				shuffle(v);
				concatenate(test_input, { loc(v, 0, test_num, 1) });
				concatenate(train_input, { loc(v, test_num, v.size(), 1)});
				concatenate(test_target, { Vector(test_num, k) });
				concatenate(train_target, { Vector(v.size() - test_num, k)});
			}

			return {test_target, train_target, test_input, train_input};
		}

		splitResult_v test_train_val_split(const DataModel& Dm, double test_ratio, double val_ratio, double class_std)
		{
			if (test_ratio > 1. || test_ratio < 0. || val_ratio > 1. || val_ratio < 0. || (test_ratio + val_ratio) > 1.) return {};

			auto res = splitByClass(Dm);
			size_t test_num;
			size_t val_num;

			Vector test_target;
			Vector train_target;
			Vector val_target;
			Vector2D test_input;
			Vector2D train_input;
			Vector2D val_input;

			for (auto& [k, v] : res) {
				test_num = static_cast<size_t>(v.size() * test_ratio + 0.5);
				val_num = static_cast<size_t>(v.size() * val_ratio + 0.5);
				shuffle(v);
				concatenate(test_input, { loc(v, 0, test_num, 1) });
				concatenate(val_input, { loc(v, test_num, test_num + val_num, 1) });
				concatenate(train_input, { loc(v, test_num + val_num, v.size(), 1)});
				concatenate(test_target, { Vector(test_num, k) });
				concatenate(val_target, { Vector(val_num, k) });
				concatenate(train_target, { Vector(v.size() - test_num - val_num, k)});
			}

			return { test_target, train_target, val_target, test_input, train_input, val_input };
		}

	}
}
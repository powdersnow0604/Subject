#include "Evaluation.h"
#include <iostream>
#include <map>

namespace BasicAi {
	namespace Evaluation {

		static size_t bsearch(const std::vector < std::pair<double, Vector2D>>& vec, size_t s, size_t e, double key, size_t null_value)
		{
			if (s == e) return vec[s].first == key ? s : null_value;

			size_t mid = (s + e) / 2;

			if (vec[mid].first == key) return mid;
			else if (vec[mid].first > key) return bsearch(vec, 0, mid - 1, key, null_value);
			else return bsearch(vec, mid +1, e, key, null_value);
		}

		static std::vector < std::pair<double, Vector2D>> splitByClass(const DataModel& Dm)
		{
			std::vector < std::pair<double, Vector2D>> res;
			size_t s;

			res.push_back({ Dm(0), {Dm[0]} });

			for (size_t i = 1; i < Dm.size; ++i) {
				if ((s = bsearch(res, 0, res.size() - 1, Dm(i), res.size())) == res.size()) {
					auto iter = res.begin();
					for (; iter != res.end(); ++iter) {
						if (iter->first > Dm(i)) break;
					}
					res.insert(iter, { Dm(i), {Dm[i]} });
				}
				else {
					res[s].second.push_back(Dm[i]);
				}

			}


			return res;
		}

		splitResult test_train_split(const DataModel& Dm, double test_ratio, double class_std)
		{
			if (test_ratio > 1. || test_ratio < 0.) return {};

			auto res = splitByClass(Dm);
			size_t test_num;
			double temp;
			double remainder = 0;
			Vector test_target;
			Vector train_target;
			Vector2D test_input;
			Vector2D train_input;

			shuffle(res);

			for (auto& [k, v] : res) {
				temp = v.size() * test_ratio + remainder + 1e-15;  // 1e-15 는 truncation error 보정용
				test_num = static_cast<size_t>(temp);
				remainder = temp - test_num;

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
			double temp;
			double rem_test = 0;
			double rem_val = 0;

			Vector test_target;
			Vector train_target;
			Vector val_target;
			Vector2D test_input;
			Vector2D train_input;
			Vector2D val_input;

			shuffle(res);

			for (auto& [k, v] : res) {
				temp = v.size() * test_ratio + rem_test + 1e-15;  // 1e-15 는 truncation error 보정용
				test_num = static_cast<size_t>(temp);
				rem_test = temp - test_num;

				temp = v.size() * val_ratio + rem_val + 1e-15;
				val_num = static_cast<size_t>(temp);
				rem_val = temp - val_num;

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
#ifndef __EVALUATION_H__
#define __EVALUATION_H__

#include "DataModel.h"
#include "pch.h"
#include <random>
#include <array>
#include <iostream>

using namespace BasicAi::DataModels;

namespace BasicAi {
	namespace Evaluation {
		using std::array;

		template <typename T>
		concept Evalreq = requires(T t, DataModel dm) {
			t.score(dm);
			t.fit(dm);
		};

		struct splitResult {
			Vector test_target;
			Vector train_target;
			Vector2D test_input;
			Vector2D train_input;
			splitResult(Vector test_t_, Vector train_t_, Vector2D test_i_, Vector2D train_i_):
				test_target(test_t_), train_target(train_t_), test_input(test_i_), train_input(train_i_) {}
			splitResult() {}
		};

		struct splitResult_v {
			Vector test_target;
			Vector train_target;
			Vector val_target;
			Vector2D test_input;
			Vector2D train_input;
			Vector2D val_input;
			splitResult_v(Vector test_t_, Vector train_t_, Vector val_t_, Vector2D test_i_, Vector2D train_i_, Vector2D val_i_) :
				test_target(test_t_), train_target(train_t_), test_input(test_i_), train_input(train_i_), val_target(val_t_), val_input(val_i_) {}
			splitResult_v() {}
		};

		splitResult test_train_split(const DataModel& Dm, double test_ratio);

		// ratio를 곱한 것의 버림 적용
		splitResult_v test_train_val_split(const DataModel& Dm, double test_ratio, double val_ratio);

		array<array<size_t, 2>, 2> confusionMatrix(const TargetModel& Tm, const TargetModel& pred, double positive);

		void roc_curve(const TargetModel& Tar, const Vector& proba, double positive);

		double roc_auc_score(const TargetModel& Tar, const Vector& proba, double positive, bool print_roc);


		template <Evalreq T>
		double ThreeWayHoldOut(T& model, const splitResult_v& Sr, size_t epoch)
		{
			Vector2D train_score; train_score.reserve(epoch);
			Vector2D val_score; val_score.reserve(epoch);
			double score;

			for (size_t i = 0; i < epoch; ++i) {
				model.fit({ Sr.train_input, Sr.train_target });
				score = model.score({ Sr.train_input, Sr.train_target });
				train_score.push_back({ (double)i,score });
				score = model.score({ Sr.val_input, Sr.val_target });
				val_score.push_back({ (double)i, score });
			}

			plot(train_score, "3-way hold out", Scalar(0, 0, 255));
			plot(val_score, "3-way hold out", Scalar(255, 0, 0));
			show("3-way hold out");

			return model.score({ Sr.test_input, Sr.test_target });
		}

		template <Evalreq T>
		double K_FoldCV(T& model, const DataModel& Dm, double ratio = 0.1, bool print = false)
		{
			if (ratio <= 0 && ratio >= 1) return -1;

			const Vector2D& input = *Dm.input.get();
			const Vector& target = *Dm.target.get();
			
			size_t fold_size = static_cast<size_t>(Dm.size * ratio);
			size_t curr_fold_size;
			size_t cnt = 0;
			vector<double> score;


			for (size_t i = 0; i < Dm.size; i += fold_size) {
				curr_fold_size = fold_size > (Dm.size - i) ? (Dm.size - i) : fold_size;
				const Vector2D& fold_test_input = loc(input, i, i + curr_fold_size, 1);
				const Vector2D& fold_train_input = concatenate({ loc(input, 0, i, 1), loc(input, i + curr_fold_size, Dm.size, 1) });
				const Vector& fold_test_target = loc(target, i, i + curr_fold_size, 1);
				const Vector& fold_train_target = concatenate({ loc(target, 0, i, 1), loc(target, i + curr_fold_size, Dm.size, 1) });

				//calc score
				model.fit({ fold_train_input, fold_train_target });
				score.push_back(model.score({ fold_test_input, fold_test_target }));
				++cnt;
			}

			double mean_ = mean(score);
			double stdev_ = stdev(score, mean_);

			if (print) {
				std::cout << "mean = " << mean_ << std::endl;
				std::cout << "variance = " << stdev_ * stdev_ << std::endl;
			}
		

			return mean_;
		}

		template<Evalreq T>
		double LooCV(T& model, const DataModel& Dm, double ratio = 0.1)
		{
			if (ratio <= 0 && ratio >= 1) return -1;

			Vector2D input = *Dm.input.get();
			Vector target = *Dm.target.get();
			Vector2D fold_train_input, fold_test_input;
			Vector fold_train_target, fold_test_target;
			size_t cnt = 0;
			double score = 0;

			for (size_t i = 0; i < Dm.size; ++i) {
				fold_test_input = { input[i] };
				fold_train_input = concatenate({ loc(input, 0, i, 1), loc(input, i + 1, Dm.size, 1) });
				fold_test_target = { target[i] };
				fold_train_target = concatenate({ loc(target, 0, i, 1), loc(target, i + 1, Dm.size, 1) });

				model.fit({ fold_train_input, fold_train_target });
				score += model.score({ fold_test_input, fold_test_target });
				++cnt;
			}

			return score / cnt;
		}

	}
}
#endif


#include "DataModel.h"
#include <random>
#include <variant>


namespace BasicAi {
	namespace DataModels {
		//Input model
		InputModel::InputModel(std::initializer_list<std::vector<double>>&& list) {
			size = list.size();
			input = std::make_shared< std::vector<std::vector<double>>>(list);
		}

		InputModel::InputModel(const std::vector<std::vector<double>>& list)
		{
			size = list.size();
			input = std::make_shared< std::vector<std::vector<double>>>(list);
		}

		std::vector<double>& InputModel::operator[](size_t i)
		{
			return (*input)[i];
		}


		//Target model
		TargetModel::TargetModel(std::initializer_list<double>&& list) {
			size = list.size();
			target = std::make_shared<std::vector<double>>(list);
		}

		TargetModel::TargetModel(const std::vector<double>& list) {
			size = list.size();
			target = std::make_shared<std::vector<double>>(list);
		}

		double& TargetModel::operator[](size_t i)
		{
			return (*target)[i];
		}


		//Data model
		DataModel::DataModel(TargetModel& t, InputModel& i) : size(0)
		{
			if (t.size == i.size) {
				size = t.size;
				target = t.target;
				input = i.input;
			}
		}

		DataModel::DataModel(const std::vector<std::vector<double>>& _input, const std::vector<double>& _target) : size(0)
		{
			if (_input.size() == _target.size()) {
				size = _input.size();
				target = std::make_shared<std::vector<double>>(_target);
				input = std::make_shared< std::vector<std::vector<double>>>(_input);
			}
		}

		DataModel& DataModel::operator= (const DataModel& other)
		{
			target = other.target;
			input = other.input;
			size = other.size;

			return *this;
		}

		std::vector<double>& DataModel::operator[](size_t i) const
		{
			return (*input)[i];
		}

		double& DataModel::operator()(size_t i) const
		{
			return (*target)[i];
		}


		//function
		InputModel randomInput(double arg1, double arg2, size_t size, int dim, RANDOM_ENGINE engine_type)
		{
			vector<vector<double>> res;

			std::random_device rd;
			std::mt19937 gen(rd());
			std::variant<std::uniform_real_distribution<double>, std::normal_distribution<double>> engine;

			switch (static_cast<int>(engine_type))
			{
			case 1: engine = std::normal_distribution<double>(arg1, arg2); break;
			case 0: engine = std::uniform_real_distribution<double>(arg1, arg2); break;
			}

			res.resize(size);
			for (auto& vec : res) {
				std::visit([&vec, &gen, dim ](auto&& arg) {
					vec.reserve(dim);
					for (size_t i = 0; i < dim; ++i) vec.push_back(arg(gen)); }, engine);
			}

			return res;
		}
	}
}
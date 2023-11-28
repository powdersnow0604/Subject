#include "DataModel.h"
#include <random>
#include <variant>
#include <fstream>
#include <iostream>

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

		InputModel::InputModel(const std::shared_ptr<Vector2D>& list)
		{
			input = list;
			size = input->size();
		}

		InputModel::InputModel(): size(0)
		{
			input = std::make_shared< std::vector<std::vector<double>>>(Vector2D());
		}


		const std::vector<double>& InputModel::operator[](size_t i) const
		{
			return (*input)[i];
		}

		std::vector<double>& InputModel::operator[](size_t i) 
		{
			return (*input)[i];
		}

		const Vector2D& InputModel::get() const
		{
			return *(input.get());
		}

		Vector2D& InputModel::get()
		{
			return *(input.get());
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

		TargetModel::TargetModel(const std::shared_ptr<Vector>& list)
		{
			target = list;
			size = target->size();
		}

		TargetModel::TargetModel() : size(0)
		{
			target = std::make_shared<std::vector<double>>(Vector());
		}

		const double& TargetModel::operator[](size_t i) const
		{
			return (*target)[i];
		}

		double& TargetModel::operator[](size_t i) 
		{
			return (*target)[i];
		}

		const Vector& TargetModel::get() const
		{
			return *(target.get());
		}

		Vector& TargetModel::get()
		{
			return *(target.get());
		}


		//Data model
		DataModel::DataModel(InputModel& i, TargetModel& t) : size(0)
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

		DataModel::DataModel() : size(0)
		{
			target = std::make_shared<std::vector<double>>(Vector());
			input = std::make_shared< std::vector<std::vector<double>>>(Vector2D());
		}

		DataModel& DataModel::operator= (const DataModel& other)
		{
			target = other.target;
			input = other.input;
			size = other.size;

			return *this;
		}

		const std::vector<double>& DataModel::operator[](size_t i) const
		{
			return (*input)[i];
		}

		const double& DataModel::operator()(size_t i) const
		{
			return (*target)[i];
		}

		std::vector<double>& DataModel::operator[](size_t i)
		{
			return (*input)[i];
		}

		double& DataModel::operator()(size_t i)
		{
			return (*target)[i];
		}

		const Vector2D& DataModel::getInput() const
		{
			return *(input.get());
		}
		const Vector& DataModel::getTarget() const
		{
			return *(target.get());
		}

		Vector2D& DataModel::getInput()
		{
			return *(input.get());
		}

		Vector& DataModel::getTarget()
		{
			return *(target.get());
		}

		bool DataModel::is_string_digit(std::string str)
		{
			for (size_t i = 0; i < str.size(); ++i) {
				if (!isdigit(str[i])) return false;
			}
			return true;
		}

		vector<map<std::string, double>> DataModel::read_csv(const char* path)
		{
			if (size != 0) {
				size = 0;
				input->clear();
				target->clear();
			}

			std::ifstream fin(path);
			if (!fin.is_open()) {
				std::perror(NULL);
				return {};
			}

			char line[128];
			char* x;
			char* sptr = line;
			std::vector<bool> is_digit_v;
			size_t num = 0, index = 0;
			std::vector<std::map<std::string, double>> table;

			fin.getline(line, 128);
			sptr = line;
			(void)strtok_s(NULL, ",", &sptr);
			while (x = strtok_s(NULL, ",", &sptr)) {
				++num;
			}

			table.resize(num);


			fin.getline(line, 128);
			input->push_back({});
			sptr = line;
			(void)strtok_s(NULL, ",", &sptr);
			for (size_t i = 0; i < num; ++i) {
				x = strtok_s(NULL, ",", &sptr);
				is_digit_v.push_back(is_string_digit(x));

				if (i == num - 1) {
					if (is_digit_v[i]) {
						target->push_back(atof(x));
					}
					else {
						if (table[i].find(x) != table[i].end()) {
							target->push_back(table[i][x]);
						}
						else {
							table[i][x] = (double)(table[i].size());
							target->push_back(table[i][x]);
						}
					}
				}
				else {
					if (is_digit_v[i]) {
						input->operator[](size).push_back(atof(x));
					}
					else {
						if (table[i].find(x) != table[i].end()) {
							input->operator[](size).push_back(table[i][x]);
						}
						else {
							table[i][x] = (double)(table[i].size());
							input->operator[](size).push_back(table[i][x]);
						}
					}
				}
			}

			++size;


			do {
				fin.getline(line, 128);
				if (strlen(line) == 0) break;
				input->push_back({});
				sptr = line;
				(void)strtok_s(NULL, ",", &sptr);
				for (size_t i = 0; i < num; ++i) {
					x = strtok_s(NULL, ",", &sptr);
					if (i == num - 1) {
						if (is_digit_v[i]) {
							target->push_back(atof(x));
						}
						else {
							if (table[i].find(x) != table[i].end()) {
								target->push_back(table[i][x]);
							}
							else {
								table[i][x] = (double)(table[i].size());
								target->push_back(table[i][x]);
							}
						}
					}
					else {
						if (is_digit_v[i]) {
							input->operator[](size).push_back(atof(x));
						}
						else {
							if (table[i].find(std::string(x)) != table[i].end()) {
								input->operator[](size).push_back(table[i][x]);
							}
							else {
								table[i][x] = (double)(table[i].size());
								input->operator[](size).push_back(table[i][x]);
							}
						}
					}
				}

				++size;
			} while (1);

			fin.close();

			return table;
		}

		void DataModel::print() const
		{
			std::cout << "data:" << std::endl;
			for (size_t i = 0; i < size; ++i) {
				for (size_t j = 0; j < input->operator[](i).size(); ++j) {
					std::cout << input->operator[](i)[j] << " ";
				}
				std::cout << target->operator[](i) << std::endl;
			}
			std::cout << std::endl;
		}

		DataModel DataModel::copy() const
		{
			DataModel res;
			res.size = size;
			res.input = std::make_shared<Vector2D>(res.getInput());
			res.target = std::make_shared<Vector>(res.getTarget());

			return res;
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


		Vector2D transformInput2Value(const vector<vector<std::string>>& input, const vector<map<std::string, double>>& table)
		{
			Vector2D res(input.size());

			for (size_t i = 0; i < input.size(); ++i) {
				for (size_t j = 0; j < input[0].size(); ++j) {
					if (table[j].size() == 0) {
						res[i].push_back(atof(input[i][j].c_str()));
					}
					else {
						res[i].push_back(table[j].at(input[i][j]));
					}
				}
			}

			return res;
		}


		void printMappingTable(const vector<map<std::string, double>>& table)
		{
			for (size_t i = 0; i < table.size() - 1; ++i) {
				std::cout << "descriptive feature " << i << ": " << std::endl;
				if (table[i].size() == 0) {
					std::cout << "-its number itself\n" << std::endl;
					continue;
				}
				for (auto& [k, v] : table[i]) {
					std::cout << '-' << k << ": " << v << std::endl;
				}
				std::cout << std::endl;
			}
			
			std::cout << "target feature:" << std::endl;
			if (table.back().size() == 0) {
				std::cout << "-its number itself\n" << std::endl;
				return;
			}
			for (auto& [k, v] : table.back()) {
				std::cout << '-' << k << ": " << v << std::endl;
			}

			std::cout << std::endl;
		}
	}
}
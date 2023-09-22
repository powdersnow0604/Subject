#include "DataModel.h"


BasicAi::DataModels::InputModel::InputModel(std::initializer_list<std::vector<double>> && list) {
	for (auto& i : list) {
		input.push_back(i);
	}
}

BasicAi::DataModels::InputModel::InputModel(const std::vector<std::vector<double>>& list)
{
	input = list;
}

BasicAi::DataModels::InputModel::InputModel(std::vector<std::vector<double>>&& list)
{
	std::swap(input, list);
}

BasicAi::DataModels::TargetModel::TargetModel(std::initializer_list<double>&& list) {
	for (auto& i : list) {
		target.push_back(i);
	}
}

BasicAi::DataModels::TargetModel::TargetModel(const std::vector<double>& list) {
	target = list;
}

BasicAi::DataModels::TargetModel::TargetModel(std::vector<double>&& list) {
	std::swap(target, list);
}

BasicAi::DataModels::DataModel::DataModel(TargetModel& t, InputModel& i)
{
	if (t.target.size() == i.input.size()) {
		std::swap(target, t.target);
		std::swap(input, i.input);
	}
}

BasicAi::DataModels::DataModel::DataModel(std::vector<std::vector<double>>& _input, std::vector<double>& _target)
{
	std::swap(target, _target);
	std::swap(input, _input);
}

BasicAi::DataModels::DataModel BasicAi::DataModels::DataModel::operator= (const DataModel& other)
{
	target = other.target;
	input = other.input;

	return *this;
}

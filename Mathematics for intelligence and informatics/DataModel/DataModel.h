#ifndef __DATAMODEL__
#define __DATAMODEL__

#include <vector>

namespace BasicAi {
	namespace DataModels {


		class InputModel {
		public:
			std::vector<std::vector<double>> input;

			InputModel(std::initializer_list<std::vector<double>>&& list);	//�̵� ������
			InputModel(const std::vector<std::vector<double>>& list);	//���� ������
			InputModel(std::vector<std::vector<double>>&& list);	//�̵� ������
		};

		class TargetModel {
		public:
			std::vector<double> target;

			TargetModel(std::initializer_list<double>&& list);	//�̵� ������
			TargetModel(const std::vector<double>& list);	//���� ������
			TargetModel(std::vector<double>&& list);	//�̵� ������
		};

		class DataModel {
		public:
			std::vector<std::vector<double>> input;
			std::vector<double> target;

			DataModel(TargetModel& t, InputModel& i); //�̵� ������
			DataModel(std::vector<std::vector<double>>& _input, std::vector<double>& _target); //�̵� ������
			DataModel() {}

			DataModel operator= (const DataModel& other);
		};
	}
}

#endif


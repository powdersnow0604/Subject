#ifndef __DATAMODEL__
#define __DATAMODEL__

#include <vector>

namespace BasicAi {
	namespace DataModels {


		class InputModel {
		public:
			std::vector<std::vector<double>> input;

			InputModel(std::initializer_list<std::vector<double>>&& list);	//이동 생성자
			InputModel(const std::vector<std::vector<double>>& list);	//복사 생성자
			InputModel(std::vector<std::vector<double>>&& list);	//이동 생성자
		};

		class TargetModel {
		public:
			std::vector<double> target;

			TargetModel(std::initializer_list<double>&& list);	//이동 생성자
			TargetModel(const std::vector<double>& list);	//복사 생성자
			TargetModel(std::vector<double>&& list);	//이동 생성자
		};

		class DataModel {
		public:
			std::vector<std::vector<double>> input;
			std::vector<double> target;

			DataModel(TargetModel& t, InputModel& i); //이동 생성자
			DataModel(std::vector<std::vector<double>>& _input, std::vector<double>& _target); //이동 생성자
			DataModel() {}

			DataModel operator= (const DataModel& other);
		};
	}
}

#endif


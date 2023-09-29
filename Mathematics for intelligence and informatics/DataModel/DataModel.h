#ifndef __DATAMODEL__
#define __DATAMODEL__

#include <vector>
#include <map>
#include <memory>

using std::vector;
using std::map;
using Vector2D = vector<vector<double>>;
using Vector = vector<double>;

namespace BasicAi {
	namespace DataModels {

		//class
		class InputModel {
		public:
			std::shared_ptr< std::vector<std::vector<double>>> input;
			size_t size;

			InputModel(std::initializer_list<std::vector<double>>&& list);	
			InputModel(const std::vector<std::vector<double>>& list);	

			std::vector<double>& operator[](size_t i) const;
		};

		class TargetModel {
		public:
			std::shared_ptr<std::vector<double>> target;
			size_t size;

			TargetModel(std::initializer_list<double>&& list);	
			TargetModel(const std::vector<double>& list);	

			double& operator[](size_t) const;
		};

		class DataModel {
		public:
			std::shared_ptr< std::vector<std::vector<double>>> input;
			std::shared_ptr<std::vector<double>> target;
			size_t size;

			DataModel(TargetModel& t, InputModel& i); 
			DataModel(const std::vector<std::vector<double>>& _input, const std::vector<double>& _target);
			DataModel(): size(0) {}

			DataModel& operator= (const DataModel& other);
			std::vector<double>& operator[](size_t) const;
			double& operator()(size_t) const;
		};


		//function
		enum class RANDOM_ENGINE
		{
			RE_UNIFORM = 0,
			RE_GAUSSION = 1
		};

		InputModel randomInput(double arg1, double arg2, size_t size, int dim, RANDOM_ENGINE engine_type = RANDOM_ENGINE::RE_GAUSSION);
	}
}

#endif


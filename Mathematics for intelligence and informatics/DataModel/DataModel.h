#ifndef __DATAMODEL__
#define __DATAMODEL__

#include <vector>
#include <map>
#include <memory>
#include <string>

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
			InputModel(const std::shared_ptr<Vector2D>& list);
			InputModel();

			const std::vector<double>& operator[](size_t i) const;
			std::vector<double>& operator[](size_t i);
			const Vector2D& get() const;
			Vector2D& get();
		};

		class TargetModel {
		public:
			std::shared_ptr<std::vector<double>> target;
			size_t size;

			TargetModel(std::initializer_list<double>&& list);	
			TargetModel(const std::vector<double>& list);	
			TargetModel(const std::shared_ptr<Vector>& list);
			TargetModel();

			const double& operator[](size_t) const;
			double& operator[](size_t);
			const Vector& get() const;
			Vector& get();
		};

		class DataModel {

			bool is_string_digit(std::string str);
		public:
			std::shared_ptr< std::vector<std::vector<double>>> input;
			std::shared_ptr<std::vector<double>> target;
			size_t size;

			DataModel(InputModel& i, TargetModel& t);
			DataModel(const std::vector<std::vector<double>>& _input, const std::vector<double>& _target);
			DataModel(const DataModel& other) : size(other.size), input(other.input), target(other.target) {};
			DataModel();

			DataModel& operator= (const DataModel& other);
			std::vector<double>& operator[](size_t);
			double& operator()(size_t);
			const std::vector<double>& operator[](size_t) const;
			const double& operator()(size_t) const;
			const Vector2D& getInput() const;
			const Vector& getTarget() const;
			Vector2D& getInput();
			Vector& getTarget();
			vector<map<std::string, double>> read_csv(const char* path);
			void print() const;
			DataModel copy() const;
		};


		//function
		enum class RANDOM_ENGINE
		{
			RE_UNIFORM = 0,
			RE_GAUSSION = 1
		};

		InputModel randomInput(double arg1, double arg2, size_t size, int dim, RANDOM_ENGINE engine_type = RANDOM_ENGINE::RE_GAUSSION);

		Vector2D transformInput2Value(const vector<vector<std::string>>& input, const vector<map<std::string, double>>& table);

		void printMappingTable(const vector<map<std::string, double>>& table);
	}
}

#endif


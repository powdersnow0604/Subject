#ifndef __DATASETS_H__
#define __DATASETS_H__

#include "ndArray.hpp"


namespace DeepLearning {
	namespace Datasets {

		struct dataset {
			na::ndArray<float> train_input;
			na::ndArray<float> test_input;
			na::ndArray<float> train_target;
			na::ndArray<float> test_target;
			dataset(na::ndArray<float> _train_input, na::ndArray<float> _test_input, na::ndArray<float> _train_target, na::ndArray<float> _test_target):
				train_input(_train_input), test_input(_test_input), train_target(_train_target), test_target(_test_target) {}
		};

		class Mnist {
		public:
			static dataset load_data();
		};
	}
}



#endif


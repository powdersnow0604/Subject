#ifndef __MODEL_BASE__
#define __MODEL_BASE__ 

#include <vector>
#include "DataModel.h"

namespace BasicAi {
	class model {
	public:
		virtual void fit(const BasicAi::DataModels::DataModel&) = 0;
		virtual BasicAi::DataModels::TargetModel predict(const BasicAi::DataModels::InputModel&) = 0;
		virtual double score(const BasicAi::DataModels::DataModel&) = 0;
	};
}

#endif

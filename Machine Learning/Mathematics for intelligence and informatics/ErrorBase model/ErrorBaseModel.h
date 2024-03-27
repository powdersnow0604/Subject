#ifndef __ERRORBASEMODEL_H__
#define __ERRORBASEMODEL_H__


#include "DataModel.h"
#include "pch.h"


namespace BasicAi {
	namespace ErrorBaseModel {
		using namespace BasicAi::DataModels;

		class Perceptron {
			double eta;
			size_t epoch;
			Vector weights;
			double w0;
			Vector errors;

		public:
			Perceptron(double _eta = 1e-2, size_t _epoch = 10) : eta(_eta), epoch(_epoch), w0(0) {}
			void fit(const DataModel& Dm);
			Vector predict(const InputModel& In);
			double score(const DataModel& Dm);
		};
	}
}



#endif
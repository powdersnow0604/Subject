#include "data_loader.h"

namespace DataLoader {

	DataModel load_Fraud(void)
	{
		TargetModel target = { 1,0,0,1,0,1,0,0,0,1,0,1,1,0,0,0,0,0,0,0 };
		InputModel input = { { 0,0,0}, { 1,0,0}, { 1,0,0},{ 1,1,1},{ 2,0,0},{ 2,0,0},{ 0,0,0},{ 2,0,0 },{0,0,1}, {3,0,0},
			{0,2,0}, {0,0,0 },{0,0,1}, {1,0,0}, {2,0,0}, {0,0,0}, {2,2,1}, {2,0,2}, {2,0,0}, {1,0,0} };

		return { input, target };
	}

	
	DataModel load_spam_mail(void)
	{
		TargetModel target = { 0,0,0,1,1,1 }; 	
		InputModel input = { {1,0,1}, {1,1,0}, {1,1,0}, {0,1,1}, {0,0,0}, {0,0,0} };

		return { input, target };
	}

	
	DataModel load_guess_who_game(void)
	{
		TargetModel target = { 0,1,2,3 };
		InputModel input = { {1,0,1}, {1,0,0}, {0,1,0}, {0,0,0} };

		return { input, target };
	}

}
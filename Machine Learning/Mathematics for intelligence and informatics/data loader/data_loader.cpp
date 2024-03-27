#include "data_loader.h"

namespace DataLoader {

	DataModel load_Meningitis(void)
	{
		TargetModel target = { 0,0,0,0,1,0,0,1,0,1};
		InputModel input = { { 1,1,0}, { 0,1,0}, { 1,0,1},{ 1,0,1},{ 0,1,0},{ 1,0,1},{ 1,0,1},{ 1,0,1},{ 0,1,0},{ 1,0,1} };

		return { input, target };
	}


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


	DataModel load_rental_price(void)
	{
		TargetModel target = { 320, 380, 400, 390, 385, 410, 480, 600, 570, 620 };
		InputModel input = { {500,4,8}, {550, 7, 50}, {620, 9, 7}, {630,5,24}, {665,8,100},{700,4,8},{770,10,7}, {880,12,50}, {920,14,8},{1000,9,24} };

		return { input, target };
	}

}
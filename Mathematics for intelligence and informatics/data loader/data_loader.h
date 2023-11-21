#ifndef __DATA_LOADER_H__
#define __DATA_LOADER_H__

#include "DataModel.h"

namespace DataLoader{
	using namespace BasicAi::DataModels;

	/*		for Naive Bayes		*/
	//  [Meningitis data]
	//des1: Headache
	// 0: false, 1: true
	//des2: Fever
	// 0: false, 1: true
	//des3: Vomiting
	// 0: false, 1: true
	//target: Meningitis
	// 0: false, 1: true
	DataModel load_Meningitis(void);

	/*		for Naive Bayes		*/	
	//  [Fraud data]
	//des1: Credit History
	// 0: current, 1: paid, 2 arrears, 3: none
	//des2: Guarantor/Coapplicant
	// 0: none, 1: guarantor, 2: coapplicant
	//des3: Acconmodation
	// 0: own. 1: rent, 2: free
	//target: Fraud
	//0: false, 1: true
	DataModel load_Fraud(void);

	/*		for Decision tree		*/
	//  [Spam/Ham mail]
	//des1: Suspicious Words
	// 0: false, 1: true
	//des2: Unknown Sender
	// 0: false, 2: true
	//des3: Contaons Images
	// 0: false, 1: true
	//taret: class
	// 0: spam, 1: ham
	DataModel load_spam_mail(void);

	//  [Guess who game]
	//des1: Mam
	// 0: No, 1: Yes
	//des2: Long Hain
	// 0: No, 1: Yes
	//des3: Glasses
	// 0: No, 1: Yes
	//taret: Name
	// 0: Brain, 1: John, 2: Aphra, 3: Aoife
	DataModel load_guess_who_game(void);

}

#endif

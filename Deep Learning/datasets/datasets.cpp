#include "datasets.h"
#include <stdio.h>


namespace DeepLearning {
	namespace Datasets {

		dataset Mnist::load_data()
		{
			/*
			size_t dim_x, dim_y, i, cnt;
			int temp;
			na::ndArray<float> train_input;
			na::ndArray<float> test_input;
			na::ndArray<float> train_target;
			na::ndArray<float> test_target;

			FILE* file;
			fopen_s(&file, "C:\\Users\\User\\Desktop\\C derived\\data\\Mnist\\Mnist.txt", "r");

			if (file != NULL) {
				fscanf_s(file, "%zu %zu", &dim_y, &dim_x);
				train_input.alloc({ dim_y, dim_x });
				cnt = 0;

				for (i = dim_y * dim_x; i != 0; --i) {
					fscanf_s(file, "%d", &temp);

					train_input.at(cnt++) = (float)temp;
				}

				fscanf_s(file, "%zu %zu", &dim_y, &dim_x);
				train_target.alloc({ dim_y, dim_x });
				cnt = 0;

				for (i = dim_y * dim_x; i != 0; --i) {
					fscanf_s(file, "%d", &temp);
					train_target.at(cnt++) = (float)temp;
				}
				fclose(file);
			}
			

			fopen_s(&file, "C:\\Users\\User\\Desktop\\C derived\\data\\Mnist\\Mnist_test.txt", "r");
			if (file != NULL) {
				fscanf_s(file, "%zu %zu", &dim_y, &dim_x);
				test_input.alloc({ dim_y, dim_x });
				cnt = 0;

				for (i = dim_y * dim_x; i != 0; --i) {
					fscanf_s(file, "%d", &temp);
					test_input.at(cnt++) = (float)temp;
				}

				fscanf_s(file, "%zu %zu", &dim_y, &dim_x);
				test_target.alloc({ dim_y, dim_x });
				cnt = 0;

				for (i = dim_y * dim_x; i != 0; --i) {
					fscanf_s(file, "%d", &temp);
					test_target.at(cnt++) = (float)temp;
				}
				fclose(file);
			}
			

			FILE* fin;

			fopen_s(&fin, "C:\\Users\\User\\Desktop\\C derived\\data\\Mnist\\Mnist_train_input.data", "wb");
			if (fin != NULL) {
				fwrite(train_input.data(), sizeof(float), 60000ull * 784, fin);
				fclose(fin);
			}

			fopen_s(&fin, "C:\\Users\\User\\Desktop\\C derived\\data\\Mnist\\Mnist_test_input.data", "wb");
			if (fin != NULL) {
				fwrite(test_input.data(), sizeof(float), 10000ull * 784, fin);
				fclose(fin);
			}

			fopen_s(&fin, "C:\\Users\\User\\Desktop\\C derived\\data\\Mnist\\Mnist_train_target.data", "wb");
			if (fin != NULL) {
				fwrite(train_target.data(), sizeof(float), 60000ull * 1, fin);
				fclose(fin);
			}

			fopen_s(&fin, "C:\\Users\\User\\Desktop\\C derived\\data\\Mnist\\Mnist_test_target.data", "wb");
			if (fin != NULL) {
				fwrite(test_target.data(), sizeof(float), 10000ull * 1, fin);
				fclose(fin);
			}

			return { train_input, test_input, train_target, test_target };
			*/


			size_t train_size = 6000; // [0, 60000]
			size_t test_size = 1000; // [0, 10000]
			
			na::ndArray<float> train_input; train_input.alloc({ train_size, 784 });
			na::ndArray<float> test_input; test_input.alloc({ test_size, 784 });
			na::ndArray<float> train_target; train_target.alloc({ train_size, 1 });
			na::ndArray<float> test_target; test_target.alloc({ test_size, 1 });

			FILE* fin;

			fopen_s(&fin, "C:\\Users\\User\\Desktop\\C derived\\data\\Mnist\\Mnist_train_input.data", "rb");
			if (fin != NULL) {
				fread((float*)(train_input.data()), sizeof(float), train_size * 784, fin);
				fclose(fin);
			}

			fopen_s(&fin, "C:\\Users\\User\\Desktop\\C derived\\data\\Mnist\\Mnist_test_input.data", "rb");
			if (fin != NULL) {
				fread((float*)(test_input.data()), sizeof(float), test_size * 784, fin);
				fclose(fin);
			}

			fopen_s(&fin, "C:\\Users\\User\\Desktop\\C derived\\data\\Mnist\\Mnist_train_target.data", "rb");
			if (fin != NULL) {
				fread((float*)(train_target.data()), sizeof(float), train_size * 1, fin);
				fclose(fin);
			}

			fopen_s(&fin, "C:\\Users\\User\\Desktop\\C derived\\data\\Mnist\\Mnist_test_target.data", "rb");
			if (fin != NULL) {
				fread((float*)(test_target.data()), sizeof(float), test_size * 1, fin);
				fclose(fin);
			}

			return { train_input, test_input, train_target, test_target };
			
		}
	}
}
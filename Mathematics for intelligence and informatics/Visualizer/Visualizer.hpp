#ifndef __VISUALIZER__
#define __VISUALIZER__

#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/core/types_c.h"
#include <vector>
#include <sstream>
#include <time.h>
#include <iostream>


namespace Visualizer {
	template <typename T>
	static std::string toStringWithPrecision(const T& value, const int n = 2)
	{
		std::ostringstream out;
		out.precision(n);
		out << std::fixed << value;
		return out.str();
	}

	template <typename type>
	static std::vector<unsigned char> vec_flatten(const std::vector<std::vector<type>>& array)
	{
		std::vector<unsigned char> flattened_array;

		for (auto& element : array)
		{
			flattened_array.insert(flattened_array.end(), element.begin(), element.end());

		}
		return flattened_array;
	}

	template <typename type>
	static void  scaler_for_show_2d(std::vector<std::vector<type>>& array, std::vector<double> in_range, std::vector<double> out_range)
	{
		for (auto& i : array) {
			i[1] = (i[1] - in_range[2]) * (out_range[3] - out_range[2]) / (in_range[3] - in_range[2]) + out_range[2];
			i[0] = (i[0] - in_range[0]) * (out_range[1] - out_range[0]) / (in_range[1] - in_range[0]) + out_range[0];
		}
	}

	template <typename type>
	static std::vector<double> pointArrRange(const std::vector<std::vector<type>>& pointArr)
	{
		//vector(4) xmin, xmax, ymin, ymax
		std::vector<double> range = { pointArr[0][0], pointArr[0][0], pointArr[0][1], pointArr[0][1] };
		for (auto& i : pointArr) {
			if (range[0] > i[0]) range[0] = i[0];
			else if (range[1] < i[0]) range[1] = i[0];

			if (range[2] > i[1]) range[2] = i[1];
			else if (range[3] < i[1]) range[3] = i[1];
		}

		return range;
	}

	template <typename type>
	void show(const std::vector<std::vector<type>>& pointArr, const std::string& WinName, int plot_type = 0)
	{
		//init
		double size_mul = 1.4;
		int tick_num = 5, tick_length = 10;
		int plot_height = 400, plot_width = 400;
		int mat_height = (int)(plot_height * size_mul), mat_width = (int)(plot_width * size_mul);
		int margin_height = (mat_height - plot_height) / 2, margin_width = (mat_width - plot_width) / 2;

		const std::string& title = WinName;
		std::vector<std::vector<type>> copy_vector = pointArr;
		srand(time(NULL));
		cv::Scalar color(rand() % 256, rand() % 256, rand() % 256);
		cv::Mat print_mat(mat_height, mat_width, CV_8UC3, cv::Scalar(255, 255, 255));

		std::vector<double> range_pointArr = pointArrRange(pointArr); //vector(4) xmin, xmax, ymin, ymax
		std::vector<double> range_out = { 0, (double)(plot_width - 1), 0, (double)(plot_height - 1) };
		

		//graph format

		////////////////////////////////////////////////
		//											  //
		//				margin_y*1					  //
		//											  //
		//      //////////////////////////////////    // 
		//		//		plot_width +			//	  //
		//margin//		margin_x*0.5			//marg//  plot_height * size_mul
		//_x*1	//								//in_x//
		//		//								//*0.5//
		//		//	plot_height					//	  //
		//		//	+ margin_y*0.5				//	  //
		//		//								//	  //
		//		//								//	  //
		//		//								//	  //
		//		//								//	  //
		//		//////////////////////////////////    //
		//				margin_y*0.5				  //
		////////////////////////////////////////////////
		//         plot_width * size_mul


		cv::Point tl(margin_width, margin_height);
		cv::Point br(mat_width - margin_width / 2 , margin_height * 1.5 + plot_height);
		cv::rectangle(print_mat, cv::Rect(tl, br), cv::Scalar(0, 0, 0)); //?

		for (int i = 0; i < tick_num; ++i) {
			int x = i * plot_width / (tick_num - 1);
			int y = i * plot_height / (tick_num - 1);
			cv::line(print_mat, cv::Point(margin_width + x, margin_height * 1.5 + plot_height - 1),
				cv::Point(margin_width + x, margin_height * 1.5 + plot_height - 1 + tick_length), cv::Scalar(0, 0, 0));

			cv::line(print_mat, cv::Point(margin_width, margin_height * 1.5 + plot_height - 1 - y),
				cv::Point(margin_width - tick_length, margin_height * 1.5 + plot_height - 1 - y), cv::Scalar(0, 0, 0));

			std::string text_x = toStringWithPrecision(range_pointArr[0] + i * (range_pointArr[1] - range_pointArr[0]) / (tick_num - 1));
			auto text_size_x = cv::getTextSize(text_x, 0, 0.4, 1, 0);
			cv::putText(print_mat, text_x, cv::Point(margin_width + x - text_size_x.width / 2,
				margin_height * 1.5 + plot_height - 1 + tick_length + text_size_x.height * 1.3),
				0, 0.4, cv::Scalar(0, 0, 0));

			std::string text_y = toStringWithPrecision(range_pointArr[2] + i * (range_pointArr[3] - range_pointArr[2]) / (tick_num - 1));
			auto text_size_y = cv::getTextSize(text_y, 0, 0.4, 1, 0);
			cv::putText(print_mat, text_y, cv::Point(margin_width - tick_length - text_size_y.width * 1.1,
				margin_height * 1.5 + plot_height - 1 - y + text_size_y.height / 2),
				0, 0.4, cv::Scalar(0, 0, 0));
		}

		auto title_size = cv::getTextSize(WinName, 0, 1, 3, 0);
		cv::putText(print_mat, WinName, cv::Point((mat_width - title_size.width) / 2, (margin_height + title_size.height) / 2),
			0, 1, cv::Scalar(0, 0, 0), 3);

		//plotting points
		scaler_for_show_2d(copy_vector, range_pointArr, range_out);

		if (plot_type) {
			for (auto& i : copy_vector) {
				cv::circle(print_mat, cv::Point(margin_width + i[0], margin_height * 1.5 + (plot_height - 1) - i[1]), 3, color, -1);
			}
		}
		else {
			std::vector<cv::Point> points;
			points.reserve(copy_vector.size());

			for (auto& i : copy_vector) {
				points.push_back(cv::Point(margin_width + i[0], margin_height * 1.5 + (plot_height - 1) - i[1]));
			}

			cv::polylines(print_mat, points, false, color);
		}

		//show
		cv::imshow(WinName, print_mat);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
};
#endif

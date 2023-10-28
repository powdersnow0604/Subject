#include "Visualizer.hpp"


namespace Visualizer {

	map<std::string, Plot_Info> PLOT_INFO;

	void title(const string& WinName, const string& title) {
		if (PLOT_INFO.find(WinName) != PLOT_INFO.end()) PLOT_INFO[WinName].title = title;
	}


	void figsize(const string& WinName, int plot_height_, int plot_width_)
	{
		if (plot_height_ <= 0 || plot_width_ <= 0) return;
		if (PLOT_INFO.find(WinName) != PLOT_INFO.end()) {
			PLOT_INFO[WinName].plot_height = plot_height_;
			PLOT_INFO[WinName].plot_width = plot_width_;
		}
	}


	void show(const string& WinName)
	{
		//init
		double size_mul = 1.4;
		int tick_num = 5, tick_length = 10;

		string title;
		Mat print_mat;


		if (PLOT_INFO.find(WinName) != PLOT_INFO.end()) {

			int plot_height = PLOT_INFO[WinName].plot_height, plot_width = PLOT_INFO[WinName].plot_height;
			int mat_height = (int)(plot_height * size_mul), mat_width = (int)(plot_width * size_mul);
			int margin_height = (mat_height - plot_height) / 2, margin_width = (mat_width - plot_width) / 2;

			print_mat = Mat(mat_height, mat_width, CV_8UC3, Scalar(255, 255, 255));

			if (PLOT_INFO[WinName].title == "__default__") title = WinName;
			else title = PLOT_INFO[WinName].title;

			vector<double> range_pointArr = pointArrRange(PLOT_INFO[WinName].points); //vector(4) xmin, xmax, ymin, ymax
			vector<double> range_out = { 0, (double)((long long)plot_width - 1), 0, (double)((long long)plot_height - 1) };


			//graph format

			////////////////////////////////////////////////
			//											  //
			//				margin_y*1					  //
			//											  //
			//      //////////////////////////////////    // 
			//		//		margin_y * 0.25			//	  //
			//margin//marg//////////////////////	//marg//  plot_height * size_mul
			//_x*1	//in_x//	plot_width	  //	//in_x//
			//		//*	  //				  //	//*0.5//
			//		//0.25//				  //	//	  //
			//		//	  //plot_height		  //	//	  //
			//		//	  //				  //	//	  //
			//		//	  //   <real plot>	  //	//	  //
			//		//	  //////////////////////	//	  //
			//		//								//	  //
			//		//////////////////////////////////    //
			//				margin_y*0.5				  //
			////////////////////////////////////////////////
			//         plot_width * size_mul


			Point tl(margin_width, margin_height);
			Point br((int)(mat_width - margin_width / 2), (int)(margin_height * 1.5 + plot_height));
			cv::rectangle(print_mat, Rect(tl, br), Scalar(0, 0, 0)); //?

			for (int i = 0; i < tick_num; ++i) {
				int x = i * plot_width / (tick_num - 1);
				int y = i * plot_height / (tick_num - 1);
				cv::line(print_mat, Point((int)(margin_width * 1.25 + x), (int)(margin_height * 1.5 + plot_height - 1)),
					Point((int)(margin_width * 1.25 + x), (int)(margin_height * 1.5 + plot_height - 1 + tick_length)), Scalar(0, 0, 0));

				cv::line(print_mat, Point(margin_width, (int)(margin_height * 1.25 + plot_height - 1 - y)),
					Point(margin_width - tick_length, (int)(margin_height * 1.25 + plot_height - 1 - y)), Scalar(0, 0, 0));

				string text_x = toStringWithPrecision(range_pointArr[0] + i * (range_pointArr[1] - range_pointArr[0]) / ((long long)tick_num - 1));
				auto text_size_x = cv::getTextSize(text_x, 0, 0.4, 1, 0);
				cv::putText(print_mat, text_x, Point((int)(margin_width * 1.25 + x - text_size_x.width / 2),
					(int)(margin_height * 1.5 + plot_height - 1 + tick_length + text_size_x.height * 1.3)),
					0, 0.4, Scalar(0, 0, 0));

				string text_y = toStringWithPrecision(range_pointArr[2] + i * (range_pointArr[3] - range_pointArr[2]) / ((long long)tick_num - 1));
				auto text_size_y = cv::getTextSize(text_y, 0, 0.4, 1, 0);
				cv::putText(print_mat, text_y, Point((int)(margin_width - tick_length - text_size_y.width * 1.1),
					(int)(margin_height * 1.25 + plot_height - 1 - y + text_size_y.height / 2)),
					0, 0.4, Scalar(0, 0, 0));
			}

			auto title_size = cv::getTextSize(title, 0, 1, 3, 0);
			cv::putText(print_mat, title, Point((int)((mat_width - title_size.width) / 2), (int)((margin_height + title_size.height) / 2)),
				0, 1, Scalar(0, 0, 0), 3);

			//plotting points
			scaler_for_show_2d(PLOT_INFO[WinName].points, range_pointArr, range_out);

			size_t index = 0;
			while (!PLOT_INFO[WinName].partion_size.empty()) {
				size_t num = PLOT_INFO[WinName].partion_size.front();	PLOT_INFO[WinName].partion_size.pop();
				Scalar color = PLOT_INFO[WinName].partion_color.front();	PLOT_INFO[WinName].partion_color.pop();
				int plot_type = PLOT_INFO[WinName].partion_plot_type.front();	PLOT_INFO[WinName].partion_plot_type.pop();
				int thickness = PLOT_INFO[WinName].partion_thickness.front(); PLOT_INFO[WinName].partion_thickness.pop();

				if (plot_type == 1) {
					for (size_t limit = index + num; index < limit; ++index) {
						cv::circle(print_mat, Point((int)(margin_width * 1.25 + PLOT_INFO[WinName].points[index][0]),
							(int)(margin_height * 1.25 + ((long long)plot_height - 1) - PLOT_INFO[WinName].points[index][1])), thickness, color, -1);
					}
				}
				else {
					std::vector<Point> points;
					points.reserve(num);

					for (size_t limit = index + num; index < limit; ++index) {
						points.push_back(cv::Point((int)(margin_width * 1.25 + PLOT_INFO[WinName].points[index][0]),
							(int)(margin_height * 1.25 + ((long long)plot_height - 1) - PLOT_INFO[WinName].points[index][1])));
					}

					cv::polylines(print_mat, points, false, color, thickness);
				}
			}

			PLOT_INFO.erase(WinName);
		}
		else {
			print_mat = Mat(400, 400, CV_8UC3, Scalar(255, 255, 255));
		}

		//show
		cv::imshow(WinName, print_mat);
		cv::waitKey(0);
		cv::destroyWindow(WinName);
	}


	Scalar randomColor()
	{
		static std::mt19937 gen{ std::random_device()() };
		std::uniform_int_distribution<int> idist(0, 255);
		return Scalar(idist(gen), idist(gen), idist(gen));
	}
}
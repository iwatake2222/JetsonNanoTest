#include <stdio.h>
#include <opencv2/opencv.hpp>

static std::string get_tegra_pipeline(int width, int height, int fps);

int main(int argc, char* argv[])
{
	printf("Hello\n");

	int camId = (argc == 2) ? atoi(argv[1]) : 0;
	printf("camId = %d\n", camId);

	cv::Mat image = cv::imread(RESOURCE_DIR"lena.jpg");
	cv::imshow("Display", image);

	cv::VideoCapture cap;
	if (camId == 0) {
		cap.open(get_tegra_pipeline(1280, 720, 60));
	} else {
		cap.open(camId);
	}
	while (cap.read(image)) {
		cv::imshow("DisplayCamera", image);
		int key = cv::waitKey(1);
		if (key >= 0) break;
	}

	cv::destroyAllWindows();
	return 0;
}

// https://devtalk.nvidia.com/default/topic/1049024/what-is-the-defaulat-output-format-of-the-jetson-board-camera/?offset=6#5324356
// sudo apt install v4l-utils
// v4l2-ctl -d /dev/video0 --list-formats-ext
static std::string get_tegra_pipeline(int width, int height, int fps)
{
	return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(width) + ", height=(int)" +
	std::to_string(height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(fps) +
	"/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include "navigation.h"
#include <filesystem>  // Add this at the top
namespace fs = std::filesystem;

using namespace cv;  // cv namespace (using namespace cv), we can access the OpenCV functions directly, without pre-pending cv:: to the function name. 
using namespace std;
using namespace fs;

int main(int argc, char* argv[]) {
    
    const cv::String keys =
        "{help h usage ? |      | Print this message}"
        "{@video v         |      | Path to video file}";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Video Player");

    if (parser.has("help") || !parser.has("@video")) {
        parser.printMessage();
        return 0;
    }

    if (!parser.check()) {
        parser.printErrors();
        return -1;
    }

    std::string videoPath = parser.get<std::string>("@video");

    run_navigation(videoPath);

}
#include "navigation.h"
#include <iostream>
#include <filesystem>  // Add this at the top
namespace fs = std::filesystem;

using namespace cv;  // cv namespace (using namespace cv), we can access the OpenCV functions directly, without pre-pending cv:: to the function name. 
using namespace std;
using namespace fs;


void run_navigation(const std::string& videoPath) {
    // cv::VideoCapture cap(videoPath);
    
    // if (!cap.isOpened()) {
    //     std::cerr << "Error: Could not open the video file: " << videoPath << std::endl;
    // }

    // int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    // int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    // int fps = cap.get(cv::CAP_PROP_FPS);

    //string videoPath = "/home/luke/Gaimersheim_a2d2/optical_flow/data/gaimersheim_drive.mp4";
    path inputPath(videoPath);
    string fileNameNoExt = inputPath.stem(); // "gaimersheim_drive"
    string outputPath = "/home/luke/Gaimersheim_a2d2/optical_flow/data/" + fileNameNoExt + "_out.mp4";

    // use the VideoCapture() class to create a VideoCapture object, which we will then use to read the video file
    VideoCapture vid_capture(videoPath);
    VideoWriter output;
    bool saveVideo = true;

    // Obtain video properties using get() method and print
    int fps = vid_capture.get(CAP_PROP_FPS);
    cout << "Frames per second: " << fps << endl;
    int frame_count = vid_capture.get(CAP_PROP_FRAME_COUNT);
    cout << "Frame count: " << frame_count << endl;

    // int codec = static_cast<int>(vid_capture.get(CAP_PROP_FOURCC));
    int codec = VideoWriter::fourcc('m', 'p', '4', 'v');


    // Obtain frame size information using get() method
    int frame_width = static_cast<int>(vid_capture.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(vid_capture.get(CAP_PROP_FRAME_HEIGHT));
    Size frame_size(frame_width, frame_height);
    cout << "Frame Size: " << frame_size << endl;

    if (!vid_capture.isOpened())
    {
        cout << "Error opening video stream or file" << endl;
    }
    else 
    {
        cout << "Video file found at " << videoPath << " and opened successfully" << endl;
    }

    if (saveVideo == true)
    {
        output.open(outputPath, codec, fps, frame_size);
    }

    while (vid_capture.isOpened())
    {
            // Initialize frame matrix
            Mat frame;
            // Initialize a boolean to check if frames are there or not
            bool isSuccess = vid_capture.read(frame);
            // If frames are present, show it
            namedWindow("Frame", cv::WINDOW_AUTOSIZE);
            moveWindow("Frame", 0, 0);  // Set position to 100,100 (X, Y)
            //cv::resizeWindow("Frame", frame_width, frame_height);  // Or any size that fits your screen
            if(isSuccess == true)
            {
                //display frames
                imshow("Frame", frame);
            }
    
            // If frames are not there, close it
            if (isSuccess == false)
            {
                cout << "Video camera is disconnected" << endl;
                break;
            }        
            //wait 20 ms between successive frames and break the loop if key q is pressed
            int key = waitKey(20);
                if (key == 'q')
            {
                cout << "q key is pressed by the user. Stopping the video" << endl;
                break;
            }

            if (saveVideo == true)
            {
            // write the video 
            //Initialize video writer object
            output.write(frame);
            cout << "Writing frame..." << endl;
            }

    }
        
    // Release the objects
    vid_capture.release();
    output.release();    
    
    }


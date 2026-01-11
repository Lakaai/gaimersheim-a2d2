import cv2
import numpy as np

from gaussian import Gaussian


def run_visual_navigation():
    cap = cv2.VideoCapture("data/gaimersheim_drive.mp4")

    # Initiate ORB detector
    orb = cv2.ORB_create()

    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    mu0 = []
    sigma0 = []

    initial_density = Gaussian.from_moment(mu0, sigma0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break  # No more frames therefore end of video

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # keypoints, descriptors = orb.detectAndCompute(gray_image, None)
        corners = cv2.goodFeaturesToTrack(gray_image, maxCorners=100, qualityLevel=0.1, minDistance=7, blockSize=7)
        # annotated_frame = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)
        # cv2.imshow("Features", annotated_frame)
        corners = np.int32(corners)

        for i in corners:
            x, y = i.ravel()
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        # 
        cv2.imshow("Features", frame)
        # Press Q to quit
        if cv2.waitKey(33) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


run_visual_navigation()

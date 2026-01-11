# import numpy as np
# import cv2 as cv
# import glob

# """

# Coordinate grid for chessboard layout
# (10x7 internal corners):

# (0,0)---(1,0)---(2,0)---...---(9,0)
#   |       |       |             |
# (0,1)---(1,1)---(2,1)---...---(9,1)
#   |       |       |             |
# (0,2)---(1,2)---(2,2)---...---(9,2)
#   |       |       |             |
#  ...     ...     ...           ...
#   |       |       |             |
# (0,6)---(1,6)---(2,6)---...---(9,6)

# """
# visualise = False
# export = False 

# pattern_size = (10, 7)
# # square_size = 1.0 # Size of squares in your units (mm, cm, etc.) 

# # Termination criteria for sub-pixel refinement
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

# # Note corner position are not vectors from camera to points, they are corner positions relative to a coordinate system attached to the chessboard.
# corner_position = np.zeros((pattern_size[1]*pattern_size[0], 3), np.float32) # n x m corners x 3 coordinates (X, Y, Z)

# # Create coordinate grids, i.e two 2D arrays for X coordinates and Y coordinates to represent each corner 
# grid = np.mgrid[0:pattern_size[0], 0:pattern_size[1]]   # Each array has shape pattern_size[0] x pattern_size[1] 
# grid_transposed = grid.T  # Transpose to get [Y coordinates (grid), X coodinates (grid)]
# grid_reshaped = grid_transposed.reshape(-1, 2) # Set number of columns to 2 and automatically determine what the number of rows will be 

# corner_position[:, :2] = grid_reshaped

# # Arrays to store object points and image points from all the images
# object_points = [] # 3D point in space
# image_points = [] # 2D points in image plane

# chessboard_data = glob.glob('data/images/*.JPG')

# for chessboard_image in chessboard_data:
#     print("Processing image:", chessboard_image)
#     image = cv.imread(chessboard_image)
#     gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#     # Find the chess board corners
#     ret, corners = cv.findChessboardCorners(gray, pattern_size, None)

#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         object_points.append(corner_position)

#         refined_corners = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
#         image_points.append(refined_corners)

#         if visualise:
#             # Draw and display the corners
#             cv.drawChessboardCorners(image, pattern_size, refined_corners, ret)

#             cv.namedWindow('Annotated Chessboard', cv.WINDOW_NORMAL)
#             cv.resizeWindow('Annotated Chessboard', 1280, 720)
#             cv.imshow('Annotated Chessboard', image)
            
#             if cv.waitKey(500) & 0xFF == ord('q'):          
#                 break

# cv.destroyAllWindows()

# # Perform camera calibration
# ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

# print("Camera Matrix:")
# print(camera_matrix)
# print("Distortion Coefficients:")
# print(dist_coeffs)

# # # Undistortion
# # image = cv.imread(chessboard_data[0])
# # h,  w = image.shape[:2]
# # new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
# # undistored_image = cv.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
 
# # # Crop the image
# # x, y, w, h = roi
# # undistored_image = undistored_image[y:y+h, x:x+w]

# # Compute reprojection error
# mean_error = 0
# for i in range(len(object_points)):
#     projected_image_points, _ = cv.projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
#     error = cv.norm(image_points[i], projected_image_points, cv.NORM_L2)/len(projected_image_points)
#     mean_error += error
 
# print( "Total reprojection error: {} (pixels)".format(mean_error/len(object_points)))

# if export:
#   import json
#   data = {
#       'camera_matrix': camera_matrix.tolist(),
#       'dist_coeffs': dist_coeffs.tolist(),
#       'reprojection_error': mean_error/len(object_points)
#   }
#   with open('camera.json', 'w') as f:
#       json.dump(data, f)

from src.camera import calibrate_camera

data_path = 'data/images/*.JPG'
calibrate_camera(data_path, visualise=True, export=True, pattern_size=(10,7))
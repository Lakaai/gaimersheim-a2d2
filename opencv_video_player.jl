using PyCall

cv2 = pyimport("cv2")
cap = cv2.VideoCapture("/home/luke/Gaimersheim_a2d2/visual-navigation/data/gaimersheim_drive.mp4")
fast = cv2.FastFeatureDetector_create()
width = Int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = Int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

resolution = (width, height)
println("Resolution: ", resolution)

# Check if CUDA is available in your OpenCV build
println("CUDA devices: ", cv2.cuda.getCudaEnabledDeviceCount())

if !cap.isOpened()
    print("Error opening video file")
end     

while true
    
    ret, frame = cap.read()
    
    if !ret
        break
    end 
    
    keypoints = fast.detect(frame)
    annotated_frame = cv2.drawKeypoints(frame, keypoints, nothing, color=(0, 255, 0))
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Display the resulting frame
    cv2.imshow("gray", annotated_frame)

    key = cv2.waitKey(30) & 0xFF  # 30ms delay
    if key == Int('q')            # Convert q to ASCII value using Int() 
        break
    end
end 

cap.release()
cv2.destroyAllWindows()
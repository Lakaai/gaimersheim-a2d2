using PyCall

cv2 = pyimport("cv2")
cap = cv2.VideoCapture("/home/luke/Gaimersheim_a2d2/visual-navigation/data/gaimersheim_drive.mp4")
fast = cv2.FastFeatureDetector_create()
width = Int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = Int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Check if CUDA is available in your OpenCV build
println("CUDA devices: ", cv2.cuda.getCudaEnabledDeviceCount())

resolution = (width, height)
println("Resolution: ", resolution)

if !cap.isOpened()
    print("Error opening video file")
end     
num_frames = Int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
println("Frames: ", num_frames)

function play(cap)
    frame_count = 0
    while true
        frame_count += 1
        ret, frame = cap.read()
            if !ret
                break
            end 
        if frame_count % 1 == 0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                                          # Convert to grayscale
            keypoints = fast.detect(gray)                                                           # Detect keypoints
            annotated_frame = cv2.drawKeypoints(frame, keypoints, nothing, color=(0, 255, 0))       # Draw keypoints on the frame
            cv2.imshow("Gaimersheim Navigation", annotated_frame)                                   # Display the resulting frame
        else 
            cv2.imshow("Gaimersheim Navigation", frame)                                             # Display the original frame
        end     

        key = cv2.waitKey(30) & 0xFF  # 30ms delay
        if key == Int('q')            # Convert q to ASCII value using Int() 
            break
        end
    end 
    cap.release()
    cv2.destroyAllWindows()
end 

play(cap)
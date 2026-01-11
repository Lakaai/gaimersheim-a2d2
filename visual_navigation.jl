using PyCall
using Dates
using Printf
using BenchmarkTools

cv2 = pyimport("cv2")
cap = cv2.VideoCapture("data/gaimersheim_drive.mp4")
fast = cv2.FastFeatureDetector_create(
    threshold=33,
    nonmaxSuppression=true
)

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
    last_time = time()
    while true
        frame_count += 1
        ret, frame = cap.read()
            if !ret
                break
            end 
        
        # Measure FPS
        now = time()
        fps = 1 / (now - last_time)
        
        # keypoints = fast_detect(frame)                                                           # Detect keypoints
        
        # annotated_frame = annotate_frame(frame, keypoints)      # Draw keypoints on the original frame    

        

        keypoints = fast_detect(frame)
   
        # annotated_frame = annotate_frame(frame, keypoints)
        # t2 = time()

        # println("Annotation time: ", t2 - t1, " s")
        
        # if frame_count % 10 == 0
        #     frame = annotate_frame(frame, keypoints)
        # end
        
        last_time = now

        cv2.imshow("Gaimersheim Navigation", frame)                                   # Display the resulting frame
    
        if frame_count % 30 == 0
            println("FPS: ", fps)
            println("Keypoints: ", length(keypoints))
        end

        key = cv2.waitKey(1) & 0xFF  # 30ms delay
        if key == Int('q')            # Convert q to ASCII value using Int() 
            break
        end
    end 
    cap.release()
    cv2.destroyAllWindows()
end 

function fast_detect(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                                          # Convert to grayscale
    keypoints = fast.detect(gray, nothing)                                                  # Detect keypoints
    return keypoints
end
    
function annotate_frame(frame, keypoints)
    annotated_frame = cv2.drawKeypoints(frame, keypoints, nothing, color=(0, 255, 0))       # Draw keypoints on the frame
    return annotated_frame
end

play(cap)
using VideoIO
using GLMakie

container = VideoIO.open("data/gaimersheim_drive.mp4")      # Construct a AVInput object to access the video and audio streams in a video container
reader = VideoIO.openvideo(container)                       # Access the video stream in an AVInput, and return a VideoReader object

frame = read(reader)                                        # Read first frame to get dimensions
height, width = size(frame)[1:2]

figure = Figure(size = (width, height))                     # Create figure in external window
ax = Axis(figure[1, 1], aspect = DataAspect())

frame_observable = Observable(rotr90(frame))                # Create observable for the image
image!(ax, frame_observable)

GLMakie.activate!()                                         # Show in external window
display(GLMakie.Screen(), figure)

seekstart(reader)                                           # Reset to beginning and play
frame = read(reader)

while !eof(reader)
    read!(reader, frame)
    frame_observable[] = rotr90(frame)                      # Update the frame using observable object
    sleep(0.013) 
end

close(f)


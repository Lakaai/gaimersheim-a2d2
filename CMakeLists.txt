cmake_minimum_required(VERSION 3.5.0)
project(a2d2_navigation_core VERSION 0.1.0 LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Print OpenCV details for debugging
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Add both source files
add_executable(${PROJECT_NAME} 
    src/main.cpp
    src/navigation.cpp
)

# Link OpenCV libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

target_include_directories(${PROJECT_NAME} 
    PRIVATE 
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include 
)

# Copy data directory to build directory to ensure video file is accessible
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})
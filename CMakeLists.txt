cmake_minimum_required(VERSION 3.22)
project(Project1)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

set(Project1_LIBRARIES
        ${OpenCV_LIBS}
        )

set(Project1_INCLUDE_DIRS
        ${OpenCV_INCLUDE_DIRS}
        )

set(Project1_SOURCE_FILES
        main.cpp
        spline.h Interpolator.h Interpolator.cpp SGSmooth.hpp SGSmooth.cpp)

add_executable(Project1 ${Project1_SOURCE_FILES})
target_include_directories(Project1 PRIVATE ${Project1_INCLUDE_DIRS})
target_link_libraries(Project1 ${Project1_LIBRARIES})

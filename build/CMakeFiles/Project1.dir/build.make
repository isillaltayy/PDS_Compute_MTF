# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/isil/CLionProjects/Project1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/isil/CLionProjects/Project1/build

# Include any dependencies generated for this target.
include CMakeFiles/Project1.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Project1.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Project1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Project1.dir/flags.make

CMakeFiles/Project1.dir/main.cpp.o: CMakeFiles/Project1.dir/flags.make
CMakeFiles/Project1.dir/main.cpp.o: ../main.cpp
CMakeFiles/Project1.dir/main.cpp.o: CMakeFiles/Project1.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/isil/CLionProjects/Project1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Project1.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Project1.dir/main.cpp.o -MF CMakeFiles/Project1.dir/main.cpp.o.d -o CMakeFiles/Project1.dir/main.cpp.o -c /home/isil/CLionProjects/Project1/main.cpp

CMakeFiles/Project1.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Project1.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/isil/CLionProjects/Project1/main.cpp > CMakeFiles/Project1.dir/main.cpp.i

CMakeFiles/Project1.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Project1.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/isil/CLionProjects/Project1/main.cpp -o CMakeFiles/Project1.dir/main.cpp.s

CMakeFiles/Project1.dir/Interpolator.cpp.o: CMakeFiles/Project1.dir/flags.make
CMakeFiles/Project1.dir/Interpolator.cpp.o: ../Interpolator.cpp
CMakeFiles/Project1.dir/Interpolator.cpp.o: CMakeFiles/Project1.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/isil/CLionProjects/Project1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/Project1.dir/Interpolator.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Project1.dir/Interpolator.cpp.o -MF CMakeFiles/Project1.dir/Interpolator.cpp.o.d -o CMakeFiles/Project1.dir/Interpolator.cpp.o -c /home/isil/CLionProjects/Project1/Interpolator.cpp

CMakeFiles/Project1.dir/Interpolator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Project1.dir/Interpolator.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/isil/CLionProjects/Project1/Interpolator.cpp > CMakeFiles/Project1.dir/Interpolator.cpp.i

CMakeFiles/Project1.dir/Interpolator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Project1.dir/Interpolator.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/isil/CLionProjects/Project1/Interpolator.cpp -o CMakeFiles/Project1.dir/Interpolator.cpp.s

CMakeFiles/Project1.dir/SGSmooth.cpp.o: CMakeFiles/Project1.dir/flags.make
CMakeFiles/Project1.dir/SGSmooth.cpp.o: ../SGSmooth.cpp
CMakeFiles/Project1.dir/SGSmooth.cpp.o: CMakeFiles/Project1.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/isil/CLionProjects/Project1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/Project1.dir/SGSmooth.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Project1.dir/SGSmooth.cpp.o -MF CMakeFiles/Project1.dir/SGSmooth.cpp.o.d -o CMakeFiles/Project1.dir/SGSmooth.cpp.o -c /home/isil/CLionProjects/Project1/SGSmooth.cpp

CMakeFiles/Project1.dir/SGSmooth.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Project1.dir/SGSmooth.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/isil/CLionProjects/Project1/SGSmooth.cpp > CMakeFiles/Project1.dir/SGSmooth.cpp.i

CMakeFiles/Project1.dir/SGSmooth.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Project1.dir/SGSmooth.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/isil/CLionProjects/Project1/SGSmooth.cpp -o CMakeFiles/Project1.dir/SGSmooth.cpp.s

# Object files for target Project1
Project1_OBJECTS = \
"CMakeFiles/Project1.dir/main.cpp.o" \
"CMakeFiles/Project1.dir/Interpolator.cpp.o" \
"CMakeFiles/Project1.dir/SGSmooth.cpp.o"

# External object files for target Project1
Project1_EXTERNAL_OBJECTS =

Project1: CMakeFiles/Project1.dir/main.cpp.o
Project1: CMakeFiles/Project1.dir/Interpolator.cpp.o
Project1: CMakeFiles/Project1.dir/SGSmooth.cpp.o
Project1: CMakeFiles/Project1.dir/build.make
Project1: /usr/local/lib/libopencv_gapi.so.4.6.0
Project1: /usr/local/lib/libopencv_highgui.so.4.6.0
Project1: /usr/local/lib/libopencv_ml.so.4.6.0
Project1: /usr/local/lib/libopencv_objdetect.so.4.6.0
Project1: /usr/local/lib/libopencv_photo.so.4.6.0
Project1: /usr/local/lib/libopencv_stitching.so.4.6.0
Project1: /usr/local/lib/libopencv_video.so.4.6.0
Project1: /usr/local/lib/libopencv_videoio.so.4.6.0
Project1: /usr/local/lib/libopencv_imgcodecs.so.4.6.0
Project1: /usr/local/lib/libopencv_dnn.so.4.6.0
Project1: /usr/local/lib/libopencv_calib3d.so.4.6.0
Project1: /usr/local/lib/libopencv_features2d.so.4.6.0
Project1: /usr/local/lib/libopencv_flann.so.4.6.0
Project1: /usr/local/lib/libopencv_imgproc.so.4.6.0
Project1: /usr/local/lib/libopencv_core.so.4.6.0
Project1: CMakeFiles/Project1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/isil/CLionProjects/Project1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable Project1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Project1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Project1.dir/build: Project1
.PHONY : CMakeFiles/Project1.dir/build

CMakeFiles/Project1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Project1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Project1.dir/clean

CMakeFiles/Project1.dir/depend:
	cd /home/isil/CLionProjects/Project1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/isil/CLionProjects/Project1 /home/isil/CLionProjects/Project1 /home/isil/CLionProjects/Project1/build /home/isil/CLionProjects/Project1/build /home/isil/CLionProjects/Project1/build/CMakeFiles/Project1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Project1.dir/depend


# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hanc/Code/opencvcode/lpr/Chapter5_NumberPlateRecognition

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hanc/Code/opencvcode/lpr/Chapter5_NumberPlateRecognition/build

# Include any dependencies generated for this target.
include CMakeFiles/trainOCR.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/trainOCR.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/trainOCR.dir/flags.make

CMakeFiles/trainOCR.dir/trainOCR.cpp.o: CMakeFiles/trainOCR.dir/flags.make
CMakeFiles/trainOCR.dir/trainOCR.cpp.o: ../trainOCR.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hanc/Code/opencvcode/lpr/Chapter5_NumberPlateRecognition/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/trainOCR.dir/trainOCR.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/trainOCR.dir/trainOCR.cpp.o -c /home/hanc/Code/opencvcode/lpr/Chapter5_NumberPlateRecognition/trainOCR.cpp

CMakeFiles/trainOCR.dir/trainOCR.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/trainOCR.dir/trainOCR.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hanc/Code/opencvcode/lpr/Chapter5_NumberPlateRecognition/trainOCR.cpp > CMakeFiles/trainOCR.dir/trainOCR.cpp.i

CMakeFiles/trainOCR.dir/trainOCR.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/trainOCR.dir/trainOCR.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hanc/Code/opencvcode/lpr/Chapter5_NumberPlateRecognition/trainOCR.cpp -o CMakeFiles/trainOCR.dir/trainOCR.cpp.s

CMakeFiles/trainOCR.dir/trainOCR.cpp.o.requires:

.PHONY : CMakeFiles/trainOCR.dir/trainOCR.cpp.o.requires

CMakeFiles/trainOCR.dir/trainOCR.cpp.o.provides: CMakeFiles/trainOCR.dir/trainOCR.cpp.o.requires
	$(MAKE) -f CMakeFiles/trainOCR.dir/build.make CMakeFiles/trainOCR.dir/trainOCR.cpp.o.provides.build
.PHONY : CMakeFiles/trainOCR.dir/trainOCR.cpp.o.provides

CMakeFiles/trainOCR.dir/trainOCR.cpp.o.provides.build: CMakeFiles/trainOCR.dir/trainOCR.cpp.o


CMakeFiles/trainOCR.dir/OCR.cpp.o: CMakeFiles/trainOCR.dir/flags.make
CMakeFiles/trainOCR.dir/OCR.cpp.o: ../OCR.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hanc/Code/opencvcode/lpr/Chapter5_NumberPlateRecognition/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/trainOCR.dir/OCR.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/trainOCR.dir/OCR.cpp.o -c /home/hanc/Code/opencvcode/lpr/Chapter5_NumberPlateRecognition/OCR.cpp

CMakeFiles/trainOCR.dir/OCR.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/trainOCR.dir/OCR.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hanc/Code/opencvcode/lpr/Chapter5_NumberPlateRecognition/OCR.cpp > CMakeFiles/trainOCR.dir/OCR.cpp.i

CMakeFiles/trainOCR.dir/OCR.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/trainOCR.dir/OCR.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hanc/Code/opencvcode/lpr/Chapter5_NumberPlateRecognition/OCR.cpp -o CMakeFiles/trainOCR.dir/OCR.cpp.s

CMakeFiles/trainOCR.dir/OCR.cpp.o.requires:

.PHONY : CMakeFiles/trainOCR.dir/OCR.cpp.o.requires

CMakeFiles/trainOCR.dir/OCR.cpp.o.provides: CMakeFiles/trainOCR.dir/OCR.cpp.o.requires
	$(MAKE) -f CMakeFiles/trainOCR.dir/build.make CMakeFiles/trainOCR.dir/OCR.cpp.o.provides.build
.PHONY : CMakeFiles/trainOCR.dir/OCR.cpp.o.provides

CMakeFiles/trainOCR.dir/OCR.cpp.o.provides.build: CMakeFiles/trainOCR.dir/OCR.cpp.o


# Object files for target trainOCR
trainOCR_OBJECTS = \
"CMakeFiles/trainOCR.dir/trainOCR.cpp.o" \
"CMakeFiles/trainOCR.dir/OCR.cpp.o"

# External object files for target trainOCR
trainOCR_EXTERNAL_OBJECTS =

trainOCR: CMakeFiles/trainOCR.dir/trainOCR.cpp.o
trainOCR: CMakeFiles/trainOCR.dir/OCR.cpp.o
trainOCR: CMakeFiles/trainOCR.dir/build.make
trainOCR: /usr/local/lib/libopencv_stitching.so.3.3.0
trainOCR: /usr/local/lib/libopencv_superres.so.3.3.0
trainOCR: /usr/local/lib/libopencv_videostab.so.3.3.0
trainOCR: /usr/local/lib/libopencv_aruco.so.3.3.0
trainOCR: /usr/local/lib/libopencv_bgsegm.so.3.3.0
trainOCR: /usr/local/lib/libopencv_bioinspired.so.3.3.0
trainOCR: /usr/local/lib/libopencv_ccalib.so.3.3.0
trainOCR: /usr/local/lib/libopencv_dpm.so.3.3.0
trainOCR: /usr/local/lib/libopencv_face.so.3.3.0
trainOCR: /usr/local/lib/libopencv_freetype.so.3.3.0
trainOCR: /usr/local/lib/libopencv_fuzzy.so.3.3.0
trainOCR: /usr/local/lib/libopencv_hdf.so.3.3.0
trainOCR: /usr/local/lib/libopencv_img_hash.so.3.3.0
trainOCR: /usr/local/lib/libopencv_line_descriptor.so.3.3.0
trainOCR: /usr/local/lib/libopencv_optflow.so.3.3.0
trainOCR: /usr/local/lib/libopencv_reg.so.3.3.0
trainOCR: /usr/local/lib/libopencv_rgbd.so.3.3.0
trainOCR: /usr/local/lib/libopencv_saliency.so.3.3.0
trainOCR: /usr/local/lib/libopencv_stereo.so.3.3.0
trainOCR: /usr/local/lib/libopencv_structured_light.so.3.3.0
trainOCR: /usr/local/lib/libopencv_surface_matching.so.3.3.0
trainOCR: /usr/local/lib/libopencv_tracking.so.3.3.0
trainOCR: /usr/local/lib/libopencv_xfeatures2d.so.3.3.0
trainOCR: /usr/local/lib/libopencv_ximgproc.so.3.3.0
trainOCR: /usr/local/lib/libopencv_xobjdetect.so.3.3.0
trainOCR: /usr/local/lib/libopencv_xphoto.so.3.3.0
trainOCR: /usr/local/lib/libopencv_shape.so.3.3.0
trainOCR: /usr/local/lib/libopencv_photo.so.3.3.0
trainOCR: /usr/local/lib/libopencv_calib3d.so.3.3.0
trainOCR: /usr/local/lib/libopencv_phase_unwrapping.so.3.3.0
trainOCR: /usr/local/lib/libopencv_dnn.so.3.3.0
trainOCR: /usr/local/lib/libopencv_video.so.3.3.0
trainOCR: /usr/local/lib/libopencv_datasets.so.3.3.0
trainOCR: /usr/local/lib/libopencv_plot.so.3.3.0
trainOCR: /usr/local/lib/libopencv_text.so.3.3.0
trainOCR: /usr/local/lib/libopencv_features2d.so.3.3.0
trainOCR: /usr/local/lib/libopencv_flann.so.3.3.0
trainOCR: /usr/local/lib/libopencv_highgui.so.3.3.0
trainOCR: /usr/local/lib/libopencv_ml.so.3.3.0
trainOCR: /usr/local/lib/libopencv_videoio.so.3.3.0
trainOCR: /usr/local/lib/libopencv_imgcodecs.so.3.3.0
trainOCR: /usr/local/lib/libopencv_objdetect.so.3.3.0
trainOCR: /usr/local/lib/libopencv_imgproc.so.3.3.0
trainOCR: /usr/local/lib/libopencv_core.so.3.3.0
trainOCR: CMakeFiles/trainOCR.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hanc/Code/opencvcode/lpr/Chapter5_NumberPlateRecognition/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable trainOCR"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/trainOCR.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/trainOCR.dir/build: trainOCR

.PHONY : CMakeFiles/trainOCR.dir/build

CMakeFiles/trainOCR.dir/requires: CMakeFiles/trainOCR.dir/trainOCR.cpp.o.requires
CMakeFiles/trainOCR.dir/requires: CMakeFiles/trainOCR.dir/OCR.cpp.o.requires

.PHONY : CMakeFiles/trainOCR.dir/requires

CMakeFiles/trainOCR.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/trainOCR.dir/cmake_clean.cmake
.PHONY : CMakeFiles/trainOCR.dir/clean

CMakeFiles/trainOCR.dir/depend:
	cd /home/hanc/Code/opencvcode/lpr/Chapter5_NumberPlateRecognition/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hanc/Code/opencvcode/lpr/Chapter5_NumberPlateRecognition /home/hanc/Code/opencvcode/lpr/Chapter5_NumberPlateRecognition /home/hanc/Code/opencvcode/lpr/Chapter5_NumberPlateRecognition/build /home/hanc/Code/opencvcode/lpr/Chapter5_NumberPlateRecognition/build /home/hanc/Code/opencvcode/lpr/Chapter5_NumberPlateRecognition/build/CMakeFiles/trainOCR.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/trainOCR.dir/depend


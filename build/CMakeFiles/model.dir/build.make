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
CMAKE_COMMAND = /opt/local/bin/cmake

# The command to remove a file.
RM = /opt/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/CPneumesh

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/build

# Include any dependencies generated for this target.
include CMakeFiles/model.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/model.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/model.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/model.dir/flags.make

CMakeFiles/model.dir/Model.cpp.o: CMakeFiles/model.dir/flags.make
CMakeFiles/model.dir/Model.cpp.o: /Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/CPneumesh/Model.cpp
CMakeFiles/model.dir/Model.cpp.o: CMakeFiles/model.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/model.dir/Model.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/model.dir/Model.cpp.o -MF CMakeFiles/model.dir/Model.cpp.o.d -o CMakeFiles/model.dir/Model.cpp.o -c /Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/CPneumesh/Model.cpp

CMakeFiles/model.dir/Model.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/model.dir/Model.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/CPneumesh/Model.cpp > CMakeFiles/model.dir/Model.cpp.i

CMakeFiles/model.dir/Model.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/model.dir/Model.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/CPneumesh/Model.cpp -o CMakeFiles/model.dir/Model.cpp.s

# Object files for target model
model_OBJECTS = \
"CMakeFiles/model.dir/Model.cpp.o"

# External object files for target model
model_EXTERNAL_OBJECTS =

model.cpython-37m-darwin.so: CMakeFiles/model.dir/Model.cpp.o
model.cpython-37m-darwin.so: CMakeFiles/model.dir/build.make
model.cpython-37m-darwin.so: CMakeFiles/model.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library model.cpython-37m-darwin.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/model.dir/link.txt --verbose=$(VERBOSE)
	/Library/Developer/CommandLineTools/usr/bin/strip -x /Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/build/model.cpython-37m-darwin.so

# Rule to build all files generated by this target.
CMakeFiles/model.dir/build: model.cpython-37m-darwin.so
.PHONY : CMakeFiles/model.dir/build

CMakeFiles/model.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/model.dir/cmake_clean.cmake
.PHONY : CMakeFiles/model.dir/clean

CMakeFiles/model.dir/depend:
	cd /Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/CPneumesh /Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/CPneumesh /Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/build /Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/build /Users/Roll/Desktop/pyPneuMesh-dev/pyPneuMesh/build/CMakeFiles/model.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/model.dir/depend


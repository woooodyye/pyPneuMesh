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
CMAKE_SOURCE_DIR = /Users/Roll/desktop/eigen-3.4.0

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/Roll/desktop/eigen-3.4.0/build

# Include any dependencies generated for this target.
include doc/examples/CMakeFiles/class_Reshaped.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include doc/examples/CMakeFiles/class_Reshaped.dir/compiler_depend.make

# Include the progress variables for this target.
include doc/examples/CMakeFiles/class_Reshaped.dir/progress.make

# Include the compile flags for this target's objects.
include doc/examples/CMakeFiles/class_Reshaped.dir/flags.make

doc/examples/CMakeFiles/class_Reshaped.dir/class_Reshaped.cpp.o: doc/examples/CMakeFiles/class_Reshaped.dir/flags.make
doc/examples/CMakeFiles/class_Reshaped.dir/class_Reshaped.cpp.o: ../doc/examples/class_Reshaped.cpp
doc/examples/CMakeFiles/class_Reshaped.dir/class_Reshaped.cpp.o: doc/examples/CMakeFiles/class_Reshaped.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Roll/desktop/eigen-3.4.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object doc/examples/CMakeFiles/class_Reshaped.dir/class_Reshaped.cpp.o"
	cd /Users/Roll/desktop/eigen-3.4.0/build/doc/examples && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT doc/examples/CMakeFiles/class_Reshaped.dir/class_Reshaped.cpp.o -MF CMakeFiles/class_Reshaped.dir/class_Reshaped.cpp.o.d -o CMakeFiles/class_Reshaped.dir/class_Reshaped.cpp.o -c /Users/Roll/desktop/eigen-3.4.0/doc/examples/class_Reshaped.cpp

doc/examples/CMakeFiles/class_Reshaped.dir/class_Reshaped.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/class_Reshaped.dir/class_Reshaped.cpp.i"
	cd /Users/Roll/desktop/eigen-3.4.0/build/doc/examples && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/Roll/desktop/eigen-3.4.0/doc/examples/class_Reshaped.cpp > CMakeFiles/class_Reshaped.dir/class_Reshaped.cpp.i

doc/examples/CMakeFiles/class_Reshaped.dir/class_Reshaped.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/class_Reshaped.dir/class_Reshaped.cpp.s"
	cd /Users/Roll/desktop/eigen-3.4.0/build/doc/examples && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/Roll/desktop/eigen-3.4.0/doc/examples/class_Reshaped.cpp -o CMakeFiles/class_Reshaped.dir/class_Reshaped.cpp.s

# Object files for target class_Reshaped
class_Reshaped_OBJECTS = \
"CMakeFiles/class_Reshaped.dir/class_Reshaped.cpp.o"

# External object files for target class_Reshaped
class_Reshaped_EXTERNAL_OBJECTS =

doc/examples/class_Reshaped: doc/examples/CMakeFiles/class_Reshaped.dir/class_Reshaped.cpp.o
doc/examples/class_Reshaped: doc/examples/CMakeFiles/class_Reshaped.dir/build.make
doc/examples/class_Reshaped: doc/examples/CMakeFiles/class_Reshaped.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/Roll/desktop/eigen-3.4.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable class_Reshaped"
	cd /Users/Roll/desktop/eigen-3.4.0/build/doc/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/class_Reshaped.dir/link.txt --verbose=$(VERBOSE)
	cd /Users/Roll/desktop/eigen-3.4.0/build/doc/examples && ./class_Reshaped >/Users/Roll/desktop/eigen-3.4.0/build/doc/examples/class_Reshaped.out

# Rule to build all files generated by this target.
doc/examples/CMakeFiles/class_Reshaped.dir/build: doc/examples/class_Reshaped
.PHONY : doc/examples/CMakeFiles/class_Reshaped.dir/build

doc/examples/CMakeFiles/class_Reshaped.dir/clean:
	cd /Users/Roll/desktop/eigen-3.4.0/build/doc/examples && $(CMAKE_COMMAND) -P CMakeFiles/class_Reshaped.dir/cmake_clean.cmake
.PHONY : doc/examples/CMakeFiles/class_Reshaped.dir/clean

doc/examples/CMakeFiles/class_Reshaped.dir/depend:
	cd /Users/Roll/desktop/eigen-3.4.0/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/Roll/desktop/eigen-3.4.0 /Users/Roll/desktop/eigen-3.4.0/doc/examples /Users/Roll/desktop/eigen-3.4.0/build /Users/Roll/desktop/eigen-3.4.0/build/doc/examples /Users/Roll/desktop/eigen-3.4.0/build/doc/examples/CMakeFiles/class_Reshaped.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : doc/examples/CMakeFiles/class_Reshaped.dir/depend


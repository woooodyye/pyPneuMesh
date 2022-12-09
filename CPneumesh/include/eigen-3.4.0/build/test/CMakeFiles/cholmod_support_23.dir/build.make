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
include test/CMakeFiles/cholmod_support_23.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/cholmod_support_23.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/cholmod_support_23.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/cholmod_support_23.dir/flags.make

test/CMakeFiles/cholmod_support_23.dir/cholmod_support.cpp.o: test/CMakeFiles/cholmod_support_23.dir/flags.make
test/CMakeFiles/cholmod_support_23.dir/cholmod_support.cpp.o: ../test/cholmod_support.cpp
test/CMakeFiles/cholmod_support_23.dir/cholmod_support.cpp.o: test/CMakeFiles/cholmod_support_23.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Roll/desktop/eigen-3.4.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/cholmod_support_23.dir/cholmod_support.cpp.o"
	cd /Users/Roll/desktop/eigen-3.4.0/build/test && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/cholmod_support_23.dir/cholmod_support.cpp.o -MF CMakeFiles/cholmod_support_23.dir/cholmod_support.cpp.o.d -o CMakeFiles/cholmod_support_23.dir/cholmod_support.cpp.o -c /Users/Roll/desktop/eigen-3.4.0/test/cholmod_support.cpp

test/CMakeFiles/cholmod_support_23.dir/cholmod_support.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cholmod_support_23.dir/cholmod_support.cpp.i"
	cd /Users/Roll/desktop/eigen-3.4.0/build/test && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/Roll/desktop/eigen-3.4.0/test/cholmod_support.cpp > CMakeFiles/cholmod_support_23.dir/cholmod_support.cpp.i

test/CMakeFiles/cholmod_support_23.dir/cholmod_support.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cholmod_support_23.dir/cholmod_support.cpp.s"
	cd /Users/Roll/desktop/eigen-3.4.0/build/test && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/Roll/desktop/eigen-3.4.0/test/cholmod_support.cpp -o CMakeFiles/cholmod_support_23.dir/cholmod_support.cpp.s

# Object files for target cholmod_support_23
cholmod_support_23_OBJECTS = \
"CMakeFiles/cholmod_support_23.dir/cholmod_support.cpp.o"

# External object files for target cholmod_support_23
cholmod_support_23_EXTERNAL_OBJECTS =

test/cholmod_support_23: test/CMakeFiles/cholmod_support_23.dir/cholmod_support.cpp.o
test/cholmod_support_23: test/CMakeFiles/cholmod_support_23.dir/build.make
test/cholmod_support_23: /opt/local/lib/libcholmod.dylib
test/cholmod_support_23: /opt/local/lib/libamd.dylib
test/cholmod_support_23: /opt/local/lib/libcolamd.dylib
test/cholmod_support_23: /opt/local/lib/libcamd.dylib
test/cholmod_support_23: /opt/local/lib/libccolamd.dylib
test/cholmod_support_23: /opt/local/lib/libmetis.dylib
test/cholmod_support_23: lapack/libeigen_lapack.dylib
test/cholmod_support_23: blas/libeigen_blas.dylib
test/cholmod_support_23: test/CMakeFiles/cholmod_support_23.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/Roll/desktop/eigen-3.4.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable cholmod_support_23"
	cd /Users/Roll/desktop/eigen-3.4.0/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cholmod_support_23.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/cholmod_support_23.dir/build: test/cholmod_support_23
.PHONY : test/CMakeFiles/cholmod_support_23.dir/build

test/CMakeFiles/cholmod_support_23.dir/clean:
	cd /Users/Roll/desktop/eigen-3.4.0/build/test && $(CMAKE_COMMAND) -P CMakeFiles/cholmod_support_23.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/cholmod_support_23.dir/clean

test/CMakeFiles/cholmod_support_23.dir/depend:
	cd /Users/Roll/desktop/eigen-3.4.0/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/Roll/desktop/eigen-3.4.0 /Users/Roll/desktop/eigen-3.4.0/test /Users/Roll/desktop/eigen-3.4.0/build /Users/Roll/desktop/eigen-3.4.0/build/test /Users/Roll/desktop/eigen-3.4.0/build/test/CMakeFiles/cholmod_support_23.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/cholmod_support_23.dir/depend


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
include test/CMakeFiles/bdcsvd_3.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/bdcsvd_3.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/bdcsvd_3.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/bdcsvd_3.dir/flags.make

test/CMakeFiles/bdcsvd_3.dir/bdcsvd.cpp.o: test/CMakeFiles/bdcsvd_3.dir/flags.make
test/CMakeFiles/bdcsvd_3.dir/bdcsvd.cpp.o: ../test/bdcsvd.cpp
test/CMakeFiles/bdcsvd_3.dir/bdcsvd.cpp.o: test/CMakeFiles/bdcsvd_3.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Roll/desktop/eigen-3.4.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/bdcsvd_3.dir/bdcsvd.cpp.o"
	cd /Users/Roll/desktop/eigen-3.4.0/build/test && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/bdcsvd_3.dir/bdcsvd.cpp.o -MF CMakeFiles/bdcsvd_3.dir/bdcsvd.cpp.o.d -o CMakeFiles/bdcsvd_3.dir/bdcsvd.cpp.o -c /Users/Roll/desktop/eigen-3.4.0/test/bdcsvd.cpp

test/CMakeFiles/bdcsvd_3.dir/bdcsvd.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bdcsvd_3.dir/bdcsvd.cpp.i"
	cd /Users/Roll/desktop/eigen-3.4.0/build/test && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/Roll/desktop/eigen-3.4.0/test/bdcsvd.cpp > CMakeFiles/bdcsvd_3.dir/bdcsvd.cpp.i

test/CMakeFiles/bdcsvd_3.dir/bdcsvd.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bdcsvd_3.dir/bdcsvd.cpp.s"
	cd /Users/Roll/desktop/eigen-3.4.0/build/test && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/Roll/desktop/eigen-3.4.0/test/bdcsvd.cpp -o CMakeFiles/bdcsvd_3.dir/bdcsvd.cpp.s

# Object files for target bdcsvd_3
bdcsvd_3_OBJECTS = \
"CMakeFiles/bdcsvd_3.dir/bdcsvd.cpp.o"

# External object files for target bdcsvd_3
bdcsvd_3_EXTERNAL_OBJECTS =

test/bdcsvd_3: test/CMakeFiles/bdcsvd_3.dir/bdcsvd.cpp.o
test/bdcsvd_3: test/CMakeFiles/bdcsvd_3.dir/build.make
test/bdcsvd_3: test/CMakeFiles/bdcsvd_3.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/Roll/desktop/eigen-3.4.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bdcsvd_3"
	cd /Users/Roll/desktop/eigen-3.4.0/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bdcsvd_3.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/bdcsvd_3.dir/build: test/bdcsvd_3
.PHONY : test/CMakeFiles/bdcsvd_3.dir/build

test/CMakeFiles/bdcsvd_3.dir/clean:
	cd /Users/Roll/desktop/eigen-3.4.0/build/test && $(CMAKE_COMMAND) -P CMakeFiles/bdcsvd_3.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/bdcsvd_3.dir/clean

test/CMakeFiles/bdcsvd_3.dir/depend:
	cd /Users/Roll/desktop/eigen-3.4.0/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/Roll/desktop/eigen-3.4.0 /Users/Roll/desktop/eigen-3.4.0/test /Users/Roll/desktop/eigen-3.4.0/build /Users/Roll/desktop/eigen-3.4.0/build/test /Users/Roll/desktop/eigen-3.4.0/build/test/CMakeFiles/bdcsvd_3.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/bdcsvd_3.dir/depend


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
include test/CMakeFiles/nomalloc_7.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/nomalloc_7.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/nomalloc_7.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/nomalloc_7.dir/flags.make

test/CMakeFiles/nomalloc_7.dir/nomalloc.cpp.o: test/CMakeFiles/nomalloc_7.dir/flags.make
test/CMakeFiles/nomalloc_7.dir/nomalloc.cpp.o: ../test/nomalloc.cpp
test/CMakeFiles/nomalloc_7.dir/nomalloc.cpp.o: test/CMakeFiles/nomalloc_7.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Roll/desktop/eigen-3.4.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/nomalloc_7.dir/nomalloc.cpp.o"
	cd /Users/Roll/desktop/eigen-3.4.0/build/test && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/nomalloc_7.dir/nomalloc.cpp.o -MF CMakeFiles/nomalloc_7.dir/nomalloc.cpp.o.d -o CMakeFiles/nomalloc_7.dir/nomalloc.cpp.o -c /Users/Roll/desktop/eigen-3.4.0/test/nomalloc.cpp

test/CMakeFiles/nomalloc_7.dir/nomalloc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/nomalloc_7.dir/nomalloc.cpp.i"
	cd /Users/Roll/desktop/eigen-3.4.0/build/test && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/Roll/desktop/eigen-3.4.0/test/nomalloc.cpp > CMakeFiles/nomalloc_7.dir/nomalloc.cpp.i

test/CMakeFiles/nomalloc_7.dir/nomalloc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/nomalloc_7.dir/nomalloc.cpp.s"
	cd /Users/Roll/desktop/eigen-3.4.0/build/test && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/Roll/desktop/eigen-3.4.0/test/nomalloc.cpp -o CMakeFiles/nomalloc_7.dir/nomalloc.cpp.s

# Object files for target nomalloc_7
nomalloc_7_OBJECTS = \
"CMakeFiles/nomalloc_7.dir/nomalloc.cpp.o"

# External object files for target nomalloc_7
nomalloc_7_EXTERNAL_OBJECTS =

test/nomalloc_7: test/CMakeFiles/nomalloc_7.dir/nomalloc.cpp.o
test/nomalloc_7: test/CMakeFiles/nomalloc_7.dir/build.make
test/nomalloc_7: test/CMakeFiles/nomalloc_7.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/Roll/desktop/eigen-3.4.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable nomalloc_7"
	cd /Users/Roll/desktop/eigen-3.4.0/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/nomalloc_7.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/nomalloc_7.dir/build: test/nomalloc_7
.PHONY : test/CMakeFiles/nomalloc_7.dir/build

test/CMakeFiles/nomalloc_7.dir/clean:
	cd /Users/Roll/desktop/eigen-3.4.0/build/test && $(CMAKE_COMMAND) -P CMakeFiles/nomalloc_7.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/nomalloc_7.dir/clean

test/CMakeFiles/nomalloc_7.dir/depend:
	cd /Users/Roll/desktop/eigen-3.4.0/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/Roll/desktop/eigen-3.4.0 /Users/Roll/desktop/eigen-3.4.0/test /Users/Roll/desktop/eigen-3.4.0/build /Users/Roll/desktop/eigen-3.4.0/build/test /Users/Roll/desktop/eigen-3.4.0/build/test/CMakeFiles/nomalloc_7.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/nomalloc_7.dir/depend


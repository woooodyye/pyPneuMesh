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
include test/CMakeFiles/stdvector_2.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/stdvector_2.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/stdvector_2.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/stdvector_2.dir/flags.make

test/CMakeFiles/stdvector_2.dir/stdvector.cpp.o: test/CMakeFiles/stdvector_2.dir/flags.make
test/CMakeFiles/stdvector_2.dir/stdvector.cpp.o: ../test/stdvector.cpp
test/CMakeFiles/stdvector_2.dir/stdvector.cpp.o: test/CMakeFiles/stdvector_2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Roll/desktop/eigen-3.4.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/stdvector_2.dir/stdvector.cpp.o"
	cd /Users/Roll/desktop/eigen-3.4.0/build/test && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/stdvector_2.dir/stdvector.cpp.o -MF CMakeFiles/stdvector_2.dir/stdvector.cpp.o.d -o CMakeFiles/stdvector_2.dir/stdvector.cpp.o -c /Users/Roll/desktop/eigen-3.4.0/test/stdvector.cpp

test/CMakeFiles/stdvector_2.dir/stdvector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stdvector_2.dir/stdvector.cpp.i"
	cd /Users/Roll/desktop/eigen-3.4.0/build/test && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/Roll/desktop/eigen-3.4.0/test/stdvector.cpp > CMakeFiles/stdvector_2.dir/stdvector.cpp.i

test/CMakeFiles/stdvector_2.dir/stdvector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stdvector_2.dir/stdvector.cpp.s"
	cd /Users/Roll/desktop/eigen-3.4.0/build/test && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/Roll/desktop/eigen-3.4.0/test/stdvector.cpp -o CMakeFiles/stdvector_2.dir/stdvector.cpp.s

# Object files for target stdvector_2
stdvector_2_OBJECTS = \
"CMakeFiles/stdvector_2.dir/stdvector.cpp.o"

# External object files for target stdvector_2
stdvector_2_EXTERNAL_OBJECTS =

test/stdvector_2: test/CMakeFiles/stdvector_2.dir/stdvector.cpp.o
test/stdvector_2: test/CMakeFiles/stdvector_2.dir/build.make
test/stdvector_2: test/CMakeFiles/stdvector_2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/Roll/desktop/eigen-3.4.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable stdvector_2"
	cd /Users/Roll/desktop/eigen-3.4.0/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stdvector_2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/stdvector_2.dir/build: test/stdvector_2
.PHONY : test/CMakeFiles/stdvector_2.dir/build

test/CMakeFiles/stdvector_2.dir/clean:
	cd /Users/Roll/desktop/eigen-3.4.0/build/test && $(CMAKE_COMMAND) -P CMakeFiles/stdvector_2.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/stdvector_2.dir/clean

test/CMakeFiles/stdvector_2.dir/depend:
	cd /Users/Roll/desktop/eigen-3.4.0/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/Roll/desktop/eigen-3.4.0 /Users/Roll/desktop/eigen-3.4.0/test /Users/Roll/desktop/eigen-3.4.0/build /Users/Roll/desktop/eigen-3.4.0/build/test /Users/Roll/desktop/eigen-3.4.0/build/test/CMakeFiles/stdvector_2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/stdvector_2.dir/depend


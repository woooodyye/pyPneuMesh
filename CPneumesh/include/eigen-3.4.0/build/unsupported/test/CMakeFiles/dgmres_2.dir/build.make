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
include unsupported/test/CMakeFiles/dgmres_2.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include unsupported/test/CMakeFiles/dgmres_2.dir/compiler_depend.make

# Include the progress variables for this target.
include unsupported/test/CMakeFiles/dgmres_2.dir/progress.make

# Include the compile flags for this target's objects.
include unsupported/test/CMakeFiles/dgmres_2.dir/flags.make

unsupported/test/CMakeFiles/dgmres_2.dir/dgmres.cpp.o: unsupported/test/CMakeFiles/dgmres_2.dir/flags.make
unsupported/test/CMakeFiles/dgmres_2.dir/dgmres.cpp.o: ../unsupported/test/dgmres.cpp
unsupported/test/CMakeFiles/dgmres_2.dir/dgmres.cpp.o: unsupported/test/CMakeFiles/dgmres_2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Roll/desktop/eigen-3.4.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object unsupported/test/CMakeFiles/dgmres_2.dir/dgmres.cpp.o"
	cd /Users/Roll/desktop/eigen-3.4.0/build/unsupported/test && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT unsupported/test/CMakeFiles/dgmres_2.dir/dgmres.cpp.o -MF CMakeFiles/dgmres_2.dir/dgmres.cpp.o.d -o CMakeFiles/dgmres_2.dir/dgmres.cpp.o -c /Users/Roll/desktop/eigen-3.4.0/unsupported/test/dgmres.cpp

unsupported/test/CMakeFiles/dgmres_2.dir/dgmres.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dgmres_2.dir/dgmres.cpp.i"
	cd /Users/Roll/desktop/eigen-3.4.0/build/unsupported/test && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/Roll/desktop/eigen-3.4.0/unsupported/test/dgmres.cpp > CMakeFiles/dgmres_2.dir/dgmres.cpp.i

unsupported/test/CMakeFiles/dgmres_2.dir/dgmres.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dgmres_2.dir/dgmres.cpp.s"
	cd /Users/Roll/desktop/eigen-3.4.0/build/unsupported/test && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/Roll/desktop/eigen-3.4.0/unsupported/test/dgmres.cpp -o CMakeFiles/dgmres_2.dir/dgmres.cpp.s

# Object files for target dgmres_2
dgmres_2_OBJECTS = \
"CMakeFiles/dgmres_2.dir/dgmres.cpp.o"

# External object files for target dgmres_2
dgmres_2_EXTERNAL_OBJECTS =

unsupported/test/dgmres_2: unsupported/test/CMakeFiles/dgmres_2.dir/dgmres.cpp.o
unsupported/test/dgmres_2: unsupported/test/CMakeFiles/dgmres_2.dir/build.make
unsupported/test/dgmres_2: unsupported/test/CMakeFiles/dgmres_2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/Roll/desktop/eigen-3.4.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable dgmres_2"
	cd /Users/Roll/desktop/eigen-3.4.0/build/unsupported/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dgmres_2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
unsupported/test/CMakeFiles/dgmres_2.dir/build: unsupported/test/dgmres_2
.PHONY : unsupported/test/CMakeFiles/dgmres_2.dir/build

unsupported/test/CMakeFiles/dgmres_2.dir/clean:
	cd /Users/Roll/desktop/eigen-3.4.0/build/unsupported/test && $(CMAKE_COMMAND) -P CMakeFiles/dgmres_2.dir/cmake_clean.cmake
.PHONY : unsupported/test/CMakeFiles/dgmres_2.dir/clean

unsupported/test/CMakeFiles/dgmres_2.dir/depend:
	cd /Users/Roll/desktop/eigen-3.4.0/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/Roll/desktop/eigen-3.4.0 /Users/Roll/desktop/eigen-3.4.0/unsupported/test /Users/Roll/desktop/eigen-3.4.0/build /Users/Roll/desktop/eigen-3.4.0/build/unsupported/test /Users/Roll/desktop/eigen-3.4.0/build/unsupported/test/CMakeFiles/dgmres_2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : unsupported/test/CMakeFiles/dgmres_2.dir/depend


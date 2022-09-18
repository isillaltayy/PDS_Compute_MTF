# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/isil/CLionProjects/Project1/cmake-build-debug/_deps/argparse-src"
  "/home/isil/CLionProjects/Project1/cmake-build-debug/_deps/argparse-build"
  "/home/isil/CLionProjects/Project1/cmake-build-debug/_deps/argparse-subbuild/argparse-populate-prefix"
  "/home/isil/CLionProjects/Project1/cmake-build-debug/_deps/argparse-subbuild/argparse-populate-prefix/tmp"
  "/home/isil/CLionProjects/Project1/cmake-build-debug/_deps/argparse-subbuild/argparse-populate-prefix/src/argparse-populate-stamp"
  "/home/isil/CLionProjects/Project1/cmake-build-debug/_deps/argparse-subbuild/argparse-populate-prefix/src"
  "/home/isil/CLionProjects/Project1/cmake-build-debug/_deps/argparse-subbuild/argparse-populate-prefix/src/argparse-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/isil/CLionProjects/Project1/cmake-build-debug/_deps/argparse-subbuild/argparse-populate-prefix/src/argparse-populate-stamp/${subDir}")
endforeach()

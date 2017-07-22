cmake_minimum_required (VERSION 2.6)
project (HDG)
enable_language (Fortran)

if(NOT DEFINED ENV{PETSC_DIR})
  message(FATAL_ERROR "You must the environment variable PETSC_DIR for compilation")
endif()

link_directories ($ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib)

# Stuff from CMake example
# FFLAGS depend on the compiler
get_filename_component (Fortran_COMPILER_NAME ${CMAKE_Fortran_COMPILER} NAME)

if (Fortran_COMPILER_NAME MATCHES "gfortran.*")
  # gfortran
  set (CMAKE_Fortran_FLAGS_RELEASE "-funroll-all-loops -fno-f2c -O3")
  set (CMAKE_Fortran_FLAGS_DEBUG   "-fno-f2c -O0 -g")
elseif (Fortran_COMPILER_NAME MATCHES "ifort.*")
  # ifort (untested)
  set (CMAKE_Fortran_FLAGS_RELEASE "-f77rtl -O3")
  set (CMAKE_Fortran_FLAGS_DEBUG   "-f77rtl -O0 -g")
elseif (Fortran_COMPILER_NAME MATCHES "g77")
  # g77
  set (CMAKE_Fortran_FLAGS_RELEASE "-funroll-all-loops -fno-f2c -O3 -m32")
  set (CMAKE_Fortran_FLAGS_DEBUG   "-fno-f2c -O0 -g -m32")
else (Fortran_COMPILER_NAME MATCHES "gfortran.*")
  message ("CMAKE_Fortran_COMPILER full path: " ${CMAKE_Fortran_COMPILER})
  message ("Fortran compiler: " ${Fortran_COMPILER_NAME})
  message ("No optimized Fortran compiler flags are known, we just try -O2...")
  set (CMAKE_Fortran_FLAGS_RELEASE "-O2")
  set (CMAKE_Fortran_FLAGS_DEBUG   "-O0 -g")
endif (Fortran_COMPILER_NAME MATCHES "gfortran.*")


#
# begin of my programs
# 
set (PROGRAMS "exvec" "primal-hdg-dc")

foreach (p ${PROGRAMS})
  add_executable (${p} "${p}.F")
endforeach (p)

message(STATUS "PETSC_ARCH $ENV{PETSC_ARCH}")
message(STATUS "PETSC_DIR $ENV{PETSC_DIR}")

target_include_directories (exvec PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
			   $ENV{PETSC_DIR}/include
			   $ENV{PETSC_DIR}/include/mpiuni
			   $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/include
			   )
target_link_libraries (exvec LINK_PUBLIC petsc lapack blas)


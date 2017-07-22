cmake_minimum_required (VERSION 2.6)
project (HDG)
enable_language (Fortran)

if(NOT DEFINED ENV{PETSC_DIR})
  message(FATAL_ERROR "You must set PETSC_DIR for compilation")
endif()

message(STATUS "PETSC_ARCH $ENV{PETSC_ARCH}")
message(STATUS "PETSC_DIR $ENV{PETSC_DIR}")


link_directories ($ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib)

# Stuff from CMake example
# FFLAGS depend on the compiler
get_filename_component (Fortran_COMPILER_NAME ${CMAKE_Fortran_COMPILER} NAME)

if (Fortran_COMPILER_NAME MATCHES "gfortran.*")
  # gfortran
  set (CMAKE_Fortran_FLAGS_RELEASE "-mcmodel=large -funroll-all-loops -fno-f2c -O3")
  set (CMAKE_Fortran_FLAGS_DEBUG   "-mcmodel=large -fno-f2c -O0 -g")
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
set(PROGRAMS exvec exksp primal-hdg-dc)
#add_executable (exvec "exvec.F")
#add_executable (primal-hdg-dc "primal-hdg-dc.F")

foreach(p ${PROGRAMS})
  add_executable (${p} "${p}.F")
  target_include_directories (${p} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    $ENV{PETSC_DIR}/include
    $ENV{PETSC_DIR}/include/mpiuni
    $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/include )
  target_link_libraries (${p} LINK_PUBLIC petsc lapack blas)
endforeach(p)


#target_include_directories (primal-hdg-dc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
#			   $ENV{PETSC_DIR}/include
#			   $ENV{PETSC_DIR}/include/mpiuni
#			   $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/include
#			   )
#target_link_libraries (primal-hdg-dc LINK_PUBLIC petsc)




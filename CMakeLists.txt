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

# make sure that the default is a RELEASE
if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE RELEASE CACHE STRING
      "Choose the type of build, options are: None Debug Release."
      FORCE)
endif (NOT CMAKE_BUILD_TYPE)

if (Fortran_COMPILER_NAME MATCHES "gfortran.*")
  # gfortran
  set (CMAKE_Fortran_FLAGS_RELEASE "-w -Wfatal-errors -ffixed-form -mcmodel=large -fno-f2c -O0 -g -fbacktrace -ffpe-trap=zero,overflow,underflow")
  #set (CMAKE_Fortran_FLAGS_RELEASE "-w -mcmodel=large -funroll-all-loops -fno-f2c -O3")
  #set (CMAKE_Fortran_FLAGS_DEBUG   "-mcmodel=large -fno-f2c -O0 -g")
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

# show some messages
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Build flags: ${CMAKE_Fortran_FLAGS_RELEASE}")
message(STATUS "Fortran compiler: ${Fortran_COMPILER_NAME}")

#
# Begin of my programs (single code-single program)
#
set(PROGRAMS
  exvec
  exksp
  teste
  primal-hdg-dc
  primal-hdg-cm
  primal-hdg-dc-monomial
  primal-hdg-dc-parabolic
)

#
# Program with PETSc (HDG and tests)
#
foreach(p ${PROGRAMS})
  add_executable (${p} "src/${p}.F90")
  target_include_directories (${p} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    $ENV{PETSC_DIR}/include
    $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/include )
  target_link_libraries (${p} LINK_PUBLIC petsc mpifort lapack blas)
endforeach(p)

#
# Program to test cell model (TT2006)
#
add_executable (cellmodel "src/cellmodel.F90" "src/tt2006.F90")

#
# Programa to monodomain solver - HDG
#
add_executable (primal-monodomain "src/primal-hdg-dc-monodomain.F90" "src/tt2006.F90" "src/ms.F90")
target_include_directories (primal-monodomain PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    $ENV{PETSC_DIR}/include
    $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/include )
target_link_libraries (primal-monodomain LINK_PUBLIC petsc mpifort lapack blas)


add_executable (primal-monodomain-cells "src/primal-hdg-dc-monodomain-cells.F90" "src/tt2006.F90" "src/ms.F90")
target_include_directories (primal-monodomain-cells PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    $ENV{PETSC_DIR}/include
    $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/include )
target_link_libraries (primal-monodomain-cells LINK_PUBLIC petsc mpifort lapack blas)

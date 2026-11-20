cmake_minimum_required(VERSION 3.18)   # find_* REQUIRED needs 3.18

# --- Compiler: use mpicc ---
# The compiler must be chosen before project(), otherwise CMake detects the
# system cc first and only swaps in mpicc at generate time.
if(DEFINED ENV{Aarambam_CC})
  set(CMAKE_C_COMPILER $ENV{Aarambam_CC})
else()
  find_program(MPI_C_COMPILER NAMES mpicc REQUIRED)
  set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
endif()

project(Aarambam C)

# --- Match Makefile flags: -O3 -Wall and GNU extensions (like drand48, M_PI) ---
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)           # => gnu99, like GCC default in 2LPTPNG
add_compile_options(-O3 -Wall -g)

# Optional: compile for the CPU of the build machine (-march=native: AVX2 / AVX-512 and fused multiply-add).
# A few per cent faster, but the executables then only run on CPUs with the same instruction sets (leave it
# OFF if login and compute nodes differ), and results change at the rounding level (fused multiply-adds).
# Aarambam_NATIVE=ON as an environment variable or -D.
if(DEFINED ENV{Aarambam_NATIVE})
  set(_AARAMBAM_NATIVE_DEFAULT "$ENV{Aarambam_NATIVE}")
else()
  set(_AARAMBAM_NATIVE_DEFAULT OFF)
endif()
option(Aarambam_NATIVE "Compile the 2LPT executables for the build machine's CPU (-march=native)" ${_AARAMBAM_NATIVE_DEFAULT})
message(STATUS "Aarambam: -march=native: ${Aarambam_NATIVE}")
if(Aarambam_NATIVE)
  add_compile_options(-march=native)
endif()

# --- Dependencies of the 2LPT executables ---
# GSL
if(DEFINED ENV{Aarambam_GSL_ROOT})
  set(GSL_INCLUDE_DIR "$ENV{Aarambam_GSL_ROOT}/include" CACHE PATH "GSL include directory" FORCE)
else()
  if(DEFINED ENV{GSL_DIR})
    set(_GSL_HINT $ENV{GSL_DIR})
  elseif(DEFINED ENV{GSL_ROOT})
    set(_GSL_HINT $ENV{GSL_ROOT})
  endif()

  find_path(GSL_INCLUDE_DIR
    NAMES        gsl/gsl_math.h
    HINTS        ${_GSL_HINT}
    PATH_SUFFIXES include
    REQUIRED
  )
endif()

get_filename_component(_GSL_PREFIX "${GSL_INCLUDE_DIR}" DIRECTORY)
find_library(GSL_LIBRARY
  NAMES        gsl
  HINTS        ${_GSL_PREFIX}
  PATH_SUFFIXES lib lib64
  REQUIRED
)
get_filename_component(GSL_LIB_DIR ${GSL_LIBRARY} DIRECTORY)

# FFTW3: double precision, built with MPI (fftw3-mpi.h, libfftw3_mpi) against the same MPI as mpicc.
# Aarambam_FFTW3_ROOT is the --prefix of that build and is then the only place looked in; without it,
# FFTW3_ROOT / FFTW_DIR / FFTW_ROOT and the system paths are searched for fftw3-mpi.h.
if(DEFINED ENV{Aarambam_FFTW2_ROOT} AND NOT DEFINED ENV{Aarambam_FFTW3_ROOT})
  message(WARNING "Aarambam_FFTW2_ROOT is no longer used: Aarambam now uses FFTW3 with MPI (set Aarambam_FFTW3_ROOT)")
endif()

if(DEFINED ENV{Aarambam_FFTW3_ROOT})
  set(_FFTW3_HINT "$ENV{Aarambam_FFTW3_ROOT}")
  set(_FFTW3_PATHS NO_DEFAULT_PATH)
else()
  foreach(_var FFTW3_ROOT FFTW_DIR FFTW_ROOT)
    if(DEFINED ENV{${_var}})
      list(APPEND _FFTW3_HINT "$ENV{${_var}}")
    endif()
  endforeach()
endif()

find_path(FFTW3_INCLUDE_DIR
  NAMES        fftw3-mpi.h
  HINTS        ${_FFTW3_HINT}
  PATH_SUFFIXES include
  ${_FFTW3_PATHS}
)
if(NOT FFTW3_INCLUDE_DIR)
  message(FATAL_ERROR "FFTW3 with MPI support (fftw3-mpi.h) was not found. Build FFTW3 with --enable-mpi (see "
                      "the installation docs) and set Aarambam_FFTW3_ROOT to its --prefix.")
endif()

# the libraries of that same build, next to its include/ directory
get_filename_component(_FFTW3_PREFIX "${FFTW3_INCLUDE_DIR}" DIRECTORY)
find_library(FFTW3_MPI_LIBRARY NAMES fftw3_mpi HINTS ${_FFTW3_PREFIX} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
find_library(FFTW3_LIBRARY     NAMES fftw3     HINTS ${_FFTW3_PREFIX} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
if(NOT FFTW3_MPI_LIBRARY OR NOT FFTW3_LIBRARY)
  message(FATAL_ERROR "Found ${FFTW3_INCLUDE_DIR}/fftw3-mpi.h, but not libfftw3_mpi and libfftw3 under "
                      "${_FFTW3_PREFIX}/lib. Was FFTW3 built and installed with --enable-mpi?")
endif()
get_filename_component(FFTW3_LIB_DIR ${FFTW3_LIBRARY} DIRECTORY)
message(STATUS "Aarambam: FFTW3 ${FFTW3_MPI_LIBRARY}")

# MPI
if(DEFINED ENV{Aarambam_OPENMPI_ROOT})
  set(OPENMPI_LIB_DIR "$ENV{Aarambam_OPENMPI_ROOT}/lib" CACHE PATH "OpenMPI library directory" FORCE)
else()
  # split the colon‐delimited env vars into a CMake list and combine
  string(REPLACE ":" ";" _LD_PATHS  "$ENV{LD_LIBRARY_PATH}")
  string(REPLACE ":" ";" _LIB_PATHS "$ENV{LIBRARY_PATH}")
  list(APPEND _SEARCH_PATHS ${_LD_PATHS} ${_LIB_PATHS})

  # look for libmpi.so (or .dylib/.a) next to mpicc, then on your LD_LIBRARY_PATH or LIBRARY_PATH
  get_filename_component(_MPICC_BIN_DIR "${CMAKE_C_COMPILER}" DIRECTORY)
  find_library(OPENMPI_MPI_LIB
    NAMES mpi
    HINTS "${_MPICC_BIN_DIR}/../lib" "${_MPICC_BIN_DIR}/../lib64"
    PATHS ${_SEARCH_PATHS}
    REQUIRED
  )

  # strip off the filename to get the lib directory
  get_filename_component(OPENMPI_LIB_DIR ${OPENMPI_MPI_LIB} DIRECTORY)
endif()

# --- 2LPT executables: src/lpt/common (shared by all) + one directory per method ---
set(LPT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/lpt)
set(LPT_COMMON_SRC
  ${LPT_DIR}/common/allvars.c
  ${LPT_DIR}/common/read_param.c
  ${LPT_DIR}/common/power.c
  ${LPT_DIR}/common/lpt_core.c
  ${LPT_DIR}/common/lpt_fft.c
  ${LPT_DIR}/common/save.c
  ${LPT_DIR}/common/read_glass.c
  ${LPT_DIR}/common/nrsrc/nrutil.c
  ${LPT_DIR}/common/nrsrc/qromb.c
  ${LPT_DIR}/common/nrsrc/polint.c
  ${LPT_DIR}/common/nrsrc/trapzd.c
)

# Preprocessor options (OPT / OPTIONS in the original 2LPTic Makefile)
# Particle IDs: 32-bit by default (the executables stop if TotNumPart > 2^31-1). Set Aarambam_64BIT_IDS=ON
# (environment variable or -D) to write 8-byte IDs; the snapshot readers then need to expect 64-bit IDs.
if(DEFINED ENV{Aarambam_64BIT_IDS})
  set(_AARAMBAM_64BIT_IDS_DEFAULT "$ENV{Aarambam_64BIT_IDS}")
else()
  set(_AARAMBAM_64BIT_IDS_DEFAULT OFF)
endif()
option(Aarambam_64BIT_IDS "Write 64-bit particle IDs in the snapshots" ${_AARAMBAM_64BIT_IDS_DEFAULT})
message(STATUS "Aarambam: 64-bit particle IDs: ${Aarambam_64BIT_IDS}")

set(LPT_DEFINITIONS
  CORRECT_CIC=1
  # PRODUCEGAS=1   # uncomment if you want this default ON
  # ONLY_ZA=1      # uncomment if you want this default ON
)
if(NOT Aarambam_64BIT_IDS)
  list(APPEND LPT_DEFINITIONS NO64BITID=1)
endif()
set(LPT_INCLUDE_DIRS ${LPT_DIR}/common ${LPT_DIR}/common/nrsrc ${GSL_INCLUDE_DIR} ${FFTW3_INCLUDE_DIR})

# Code shared by all methods, compiled once
add_library(lptcore OBJECT ${LPT_COMMON_SRC})
target_include_directories(lptcore PRIVATE ${LPT_INCLUDE_DIRS})
target_compile_definitions(lptcore PRIVATE ${LPT_DEFINITIONS})

function(aarambam_lpt_executable name method_dir)
  add_executable(${name} ${ARGN} $<TARGET_OBJECTS:lptcore>)
  target_include_directories(${name} PRIVATE ${method_dir} ${LPT_INCLUDE_DIRS})
  target_compile_definitions(${name} PRIVATE ${LPT_DEFINITIONS})
  target_link_directories(${name} PRIVATE ${OPENMPI_LIB_DIR} ${GSL_LIB_DIR} ${FFTW3_LIB_DIR})
  target_link_libraries(${name} PRIVATE ${FFTW3_MPI_LIBRARY} ${FFTW3_LIBRARY} gsl gslcblas m mpi)
  # Installed binaries must find libmpi / libgsl / libfftw3 without LD_LIBRARY_PATH
  set_target_properties(${name} PROPERTIES INSTALL_RPATH "${OPENMPI_LIB_DIR};${GSL_LIB_DIR};${FFTW3_LIB_DIR}")
  install(TARGETS ${name} RUNTIME DESTINATION Aarambam/bin)
endfunction()

# Modal (Legendre basis) method; the former ResBasis variant is this executable with WhichSpectrum 1/2
aarambam_lpt_executable(2LPTBasis ${LPT_DIR}/modal
  ${LPT_DIR}/modal/main_modal.c ${LPT_DIR}/modal/params_modal.c)
aarambam_lpt_executable(2LPTBinned ${LPT_DIR}/binned
  ${LPT_DIR}/binned/main.c ${LPT_DIR}/binned/params_binned.c)

# ─── CMB-BEST Python Extension ──────────────────────────────────────────────

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(OpenMP REQUIRED COMPONENTS C)  # tetrapyd.c includes omp.h unconditionally

# paths
set(CMB_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Aarambam/basis/legendre)
set(CMB_PYX    ${CMB_SRC_DIR}/cmbbest.pyx)
set(CMB_C      ${CMAKE_CURRENT_BINARY_DIR}/Aarambam/basis/legendre/cmbbest.c)

# 1 generate cmbbest.c (the Makefile generator does not create the output dir)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Aarambam/basis/legendre)
add_custom_command(
  OUTPUT ${CMB_C}
  COMMAND ${Python3_EXECUTABLE} -m cython -3 --fast-fail -o ${CMB_C} ${CMB_PYX}
  DEPENDS ${CMB_PYX} ${CMB_SRC_DIR}/tetrapyd.c ${CMB_SRC_DIR}/arrays.c
  COMMENT "Cythonizing ${CMB_PYX}"
  VERBATIM
)

# 2 build the extension
if(COMMAND Python3_add_library)
  Python3_add_library(cmbbest MODULE WITH_SOABI
    ${CMB_C}
    ${CMB_SRC_DIR}/tetrapyd.c
    ${CMB_SRC_DIR}/arrays.c
  )
else()
  add_library(cmbbest MODULE
    ${CMB_C}
    ${CMB_SRC_DIR}/tetrapyd.c
    ${CMB_SRC_DIR}/arrays.c
  )
  # fallback suffix
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
      "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so', end='')"
    OUTPUT_VARIABLE _ext
  )
  set_target_properties(cmbbest PROPERTIES SUFFIX "${_ext}")
endif()

# 3 include & link
target_include_directories(cmbbest PRIVATE
  ${Python3_NumPy_INCLUDE_DIRS}
  ${CMB_SRC_DIR}
)
target_link_libraries(cmbbest PRIVATE OpenMP::OpenMP_C)

# 4 install into your package
install(TARGETS cmbbest
  RUNTIME DESTINATION Aarambam/basis/legendre      # no-op on MODULE, but fine
  LIBRARY DESTINATION Aarambam/basis/legendre      # .so goes here under site-packages/Aarambam/basis
)

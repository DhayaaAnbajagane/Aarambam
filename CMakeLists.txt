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

# FFTW2
if(DEFINED ENV{Aarambam_FFTW2_ROOT})
  set(_FFTW2_HINT "$ENV{Aarambam_FFTW2_ROOT}")
else()
  message(FATAL_ERROR "Environment variable Aarambam_FFTW2_ROOT must be set to your FFTW2 install/build root")
endif()

find_path(FFTW2_INCLUDE_DIR
  NAMES        dfftw.h
  HINTS        ${_FFTW2_HINT}
  PATH_SUFFIXES include
  REQUIRED
)

find_library(_FFTW2_LIB
  NAMES        fftw  dfftw
  HINTS        ${_FFTW2_HINT}
  PATH_SUFFIXES lib
  REQUIRED
)
get_filename_component(FFTW2_LIB_DIR ${_FFTW2_LIB} DIRECTORY)

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
  ${LPT_DIR}/common/save.c
  ${LPT_DIR}/common/read_glass.c
  ${LPT_DIR}/common/nrsrc/nrutil.c
  ${LPT_DIR}/common/nrsrc/qromb.c
  ${LPT_DIR}/common/nrsrc/polint.c
  ${LPT_DIR}/common/nrsrc/trapzd.c
)

# Preprocessor options (OPT / OPTIONS in the original 2LPTic Makefile)
set(LPT_DEFINITIONS
  NO64BITID=1
  CORRECT_CIC=1
  # PRODUCEGAS=1   # uncomment if you want this default ON
  # ONLY_ZA=1      # uncomment if you want this default ON
)
set(LPT_INCLUDE_DIRS ${LPT_DIR}/common ${LPT_DIR}/common/nrsrc ${GSL_INCLUDE_DIR} ${FFTW2_INCLUDE_DIR})

# Code shared by all methods, compiled once
add_library(lptcore OBJECT ${LPT_COMMON_SRC})
target_include_directories(lptcore PRIVATE ${LPT_INCLUDE_DIRS})
target_compile_definitions(lptcore PRIVATE ${LPT_DEFINITIONS})

function(aarambam_lpt_executable name method_dir)
  add_executable(${name} ${ARGN} $<TARGET_OBJECTS:lptcore>)
  target_include_directories(${name} PRIVATE ${method_dir} ${LPT_INCLUDE_DIRS})
  target_compile_definitions(${name} PRIVATE ${LPT_DEFINITIONS})
  target_link_directories(${name} PRIVATE ${OPENMPI_LIB_DIR} ${GSL_LIB_DIR} ${FFTW2_LIB_DIR})
  target_link_libraries(${name} PRIVATE m mpi drfftw_mpi dfftw_mpi drfftw dfftw gsl gslcblas)
  # Installed binaries must find libmpi / libgsl without LD_LIBRARY_PATH
  set_target_properties(${name} PROPERTIES INSTALL_RPATH "${OPENMPI_LIB_DIR};${GSL_LIB_DIR};${FFTW2_LIB_DIR}")
  install(TARGETS ${name} RUNTIME DESTINATION Aarambam/bin)
endfunction()

aarambam_lpt_executable(2LPTBasis ${LPT_DIR}/basis
  ${LPT_DIR}/basis/main.c ${LPT_DIR}/basis/power.c ${LPT_DIR}/basis/params_basis.c)
aarambam_lpt_executable(2LPTResBasis ${LPT_DIR}/resbasis
  ${LPT_DIR}/resbasis/main.c ${LPT_DIR}/resbasis/power.c ${LPT_DIR}/resbasis/params_resbasis.c)
aarambam_lpt_executable(2LPTBinned ${LPT_DIR}/binned
  ${LPT_DIR}/binned/main.c ${LPT_DIR}/binned/power.c ${LPT_DIR}/binned/params_binned.c)

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

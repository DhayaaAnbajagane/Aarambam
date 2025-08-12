cmake_minimum_required(VERSION 3.16)
project(Aarambam C)

# --- Compiler: use mpicc ---
if(DEFINED ENV{Aarambam_CC})
  set(CMAKE_C_COMPILER $ENV{Aarambam_CC} CACHE FILEPATH "C compiler" FORCE)
else()
  find_program(MPI_C_COMPILER NAMES mpicc REQUIRED)
  set(CMAKE_C_COMPILER ${MPI_C_COMPILER} CACHE FILEPATH "MPI C compiler" FORCE)
endif()

# --- Match Makefile flags: -O3 -Wall and GNU extensions (like drand48, M_PI) ---
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)           # => gnu99, like GCC default in 2LPTPNG
add_compile_options(-O3 -Wall -g)

# --- Sources (OBJS without .o) ---
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Aarambam/LPT/Basis/)
add_executable(2LPTBasis
  ${SRC_DIR}/main.c
  ${SRC_DIR}/power.c
  ${SRC_DIR}/checkchoose.c
  ${SRC_DIR}/allvars.c
  ${SRC_DIR}/save.c
  ${SRC_DIR}/read_param.c
  ${SRC_DIR}/read_glass.c
  ${SRC_DIR}/nrsrc/nrutil.c
  ${SRC_DIR}/nrsrc/qromb.c
  ${SRC_DIR}/nrsrc/polint.c
  ${SRC_DIR}/nrsrc/trapzd.c
)

# --- Include paths ---
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
  )
endif()
target_include_directories(2LPTBasis PRIVATE ${GSL_INCLUDE_DIR})
set(GSL_LIB_DIR "${GSL_INCLUDE_DIR}/../lib" CACHE PATH "GSL library directory" FORCE)

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
target_include_directories(2LPTBasis PRIVATE ${FFTW2_INCLUDE_DIR})

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
  string(REPLACE ":" ";" _LD_PATHS  $ENV{LD_LIBRARY_PATH})
  string(REPLACE ":" ";" _LIB_PATHS "$ENV{LIBRARY_PATH}")
  list(APPEND _SEARCH_PATHS ${_LD_PATHS} ${_LIB_PATHS})

  # look for libmpi.so (or .dylib/.a) on your LD_LIBRARY_PATH or LIBRARY_PATH
  find_library(OPENMPI_MPI_LIB
    NAMES mpi
    PATHS ${_SEARCH_PATHS}
    REQUIRED
  )

  # strip off the filename to get the lib directory
  get_filename_component(OPENMPI_LIB_DIR ${OPENMPI_MPI_LIB} DIRECTORY)
endif()

# Local headers
target_include_directories(2LPTBasis PRIVATE
  ${SRC_DIR}
  ${SRC_DIR}/nrsrc
)

# --- Preprocessor options (OPT / OPTIONS in the Makefile) ---
# Default Makefile switches:
target_compile_definitions(2LPTBasis PRIVATE
  NO64BITID=1
  CORRECT_CIC=1
  # PRODUCEGAS=1   # uncomment if you want this default ON
  # ONLY_ZA=1      # uncomment if you want this default ON
)

# --- Link directories (LIBS part: -L...) ---
target_link_directories(2LPTBasis PRIVATE
  ${OPENMPI_LIB_DIR}
  ${GSL_LIB_DIR}
  ${FFTW2_LIB_DIR}
)

# --- Link libraries ---
target_link_libraries(2LPTBasis PRIVATE
  m
  mpi
  drfftw_mpi dfftw_mpi drfftw dfftw
  gsl gslcblas
)
install(TARGETS 2LPTBasis RUNTIME DESTINATION Aarambam/bin)


# ─── CMB-BEST Python Extension ──────────────────────────────────────────────

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(OpenMP)  # for -fopenmp

# paths
set(CMB_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Aarambam/basis)
set(CMB_PYX    ${CMB_SRC_DIR}/cmbbest.pyx)
set(CMB_C      ${CMAKE_CURRENT_BINARY_DIR}/Aarambam/basis/cmbbest.c)

# 1 generate cmbbest.c
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
if(OpenMP_C_FOUND)
  target_link_libraries(cmbbest PRIVATE OpenMP::OpenMP_C)
  target_compile_options(cmbbest PRIVATE -fopenmp)
  target_link_options(cmbbest PRIVATE -fopenmp)
endif()

# 4 install into your package
install(TARGETS cmbbest
  RUNTIME DESTINATION Aarambam/basis      # no-op on MODULE, but fine
  LIBRARY DESTINATION Aarambam/basis      # .so goes here under site-packages/Aarambam/basis
)
